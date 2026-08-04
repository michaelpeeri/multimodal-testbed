import csv
import math
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Synthetic data for gene expression imputation
def make_synthetic_data(
    n_cells: int   = 800,
    n_genes: int   = 1000,
    n_types: int   = 5,
    missing_rate: float = 0.3,
    seed: int      = 42,
):
  """
  Simulate a log-normalised gene-expression matrix with:
    - n_types latent cell-type programmes
    - ~missing_rate dropout (set to NaN)

  Masking (type 1): missing as np.nan
  """
  rng = np.random.default_rng(seed)

  # Low-rank structure: cells × types  ×  types × genes
  programs    = rng.standard_normal((n_types, n_genes)).clip(0)    # cell types/patterns
  expr_sets = []
  for _ in ('train', 'test'):
    assignments = rng.dirichlet(np.ones(n_types), size=n_cells)      # type composition of each sample
    expr        = assignments @ programs
    expr        = np.log1p(expr + rng.exponential(0.1, expr.shape))

    # Introduce dropout (MCAR for simplicity)
    dropout = rng.random(expr.shape) < missing_rate
    expr[dropout] = float('nan')

    expr_sets.append( expr.astype(np.float32) )

  return expr_sets[0], expr_sets[1]



def build_modular_cov(
    N:int,
    K:int,
    generator,
    rho_between:float=0.05,
    rho_within:float=0.5,
    allow_negative=True,
):
  block_bounds = np.linspace(0,N,K+1).round().astype(int)
  Sigma = torch.full((N,N), rho_between, requires_grad=False)
  for k in range(K):
    s = block_bounds[k]
    e = block_bounds[k+1]
    assert(e>s)
    block = torch.full((e-s, e-s), rho_within, requires_grad=False)
    block.fill_diagonal_(1.0)

    if allow_negative:
      signs = torch.randint(0, 2, (e-s,), generator=generator) * 2 - 1
      sign_mat = signs.unsqueeze(0) * signs.unsqueeze(1)
      block = block * sign_mat.float()
      block.fill_diagonal_(1.0)  # diagonal stays 1

    Sigma[s:e,s:e] = block

  # nudge towards positive semi-definitiveness if needed
  min_eig = torch.linalg.eigvalsh(Sigma).min()
  if min_eig <= 0:
    while True:
      jitter = (-min_eig + 1e-6) * torch.eye(N)
      Sigma = Sigma + jitter
      # Re-normalise to keep unit diagonal (correlation matrix)
      d = Sigma.diagonal().sqrt()
      Sigma = Sigma / d.unsqueeze(0) / d.unsqueeze(1)
      min_eig = torch.linalg.eigvalsh(Sigma).min()

      if min_eig > 0: break

  return Sigma, block_bounds

def make_synthetic_data2(
    n_cells: int        = 800,
    n_genes: int        = 200,
    n_blocks: int       = 5,
    n_types: int        = 1,
    missing_rate: float = 0.1,
    rho_within:float    = 0.5,
    rho_between:float   = 0.05,
    means:torch.Tensor|None= None,
    stds:torch.Tensor|None= None,
    allow_negative:bool = True,
    seed: int           = 42,
    device:str          = None,
    latent_clamp: float = 15.0):
  """
  Masking (type 1): missing as np.nan
  """
  generator = torch.Generator(device=device)
  if seed is not None:
    generator.manual_seed(seed)

  #if means is not None or stds is not None:
  dtype = torch.float32
  mu    = means.to(dtype=dtype, device=device) if means is not None else torch.zeros(n_genes, dtype=dtype, device=device)
  sigma = stds.to( dtype=dtype, device=device) if stds  is not None else torch.ones( n_genes, dtype=dtype, device=device)


  Sigma = []
  block_bounds = []
  L = []
  for _ in range(n_types):
    _Sigma, _block_bounds = build_modular_cov(
      N=n_genes,
      K=n_blocks,
      rho_between=rho_between,
      rho_within=rho_within,
      allow_negative=True,
      generator=generator)

    _L = torch.linalg.cholesky(_Sigma)  # Sigma = L @ L.T

    Sigma.append( _Sigma )
    block_bounds.append( _block_bounds)
    L.append( _L )

  out = []
  for _ in [0,1]:
    Xcandidates = []

    for t in range(n_types):
      # Sample standard normals, then rotate with Cholesky factor
      Z = torch.randn(n_cells, n_genes, device=device, generator=generator)
      _Xt = Z @ L[t].T  # shape (n_cells, N)
      Xcandidates.append(_Xt)
    Xcandidates = torch.stack( Xcandidates )
    print(f'Xcandidates:{Xcandidates.shape}')

    # pick single 'type' for each sample; data will be an even mixture of the types
    x_types = torch.randint(0,n_types,size=(n_cells,))
    print(f'x_types:{x_types.shape}')
    # for each sample, fetch the data generated using the chosen type
    X = Xcandidates[x_types,torch.arange(n_cells),:]
    print(f'X:{X.shape}')

    # Map latent normal values to NB means and sample counts.
    # Use the logits parameterisation (rather than probs = r/(r+mean_nb)) to
    # avoid probs saturating to exactly 1.0 when mean_nb underflows to 0 in
    # float32 -- that violates NegativeBinomial's half-open probs constraint.
    # The exponent is also clamped to keep mean_nb in a numerically sane
    # range (heavy-tailed latents can otherwise blow up exp()).
    r = torch.tensor(10.0, device=device)
    log_mean_nb = (X * sigma.unsqueeze(0) + mu.unsqueeze(0)).clamp(-latent_clamp, latent_clamp)
    logits = torch.log(r) - log_mean_nb
    counts = torch.distributions.NegativeBinomial(total_count=r, logits=logits).sample()
    X = torch.log1p(counts)

    # Introduce dropout (MCAR for simplicity)
    dropout = torch.rand(X.shape, generator=generator) < missing_rate
    X[dropout] = float('nan')

    out.append((X, x_types))


  #modules = [
  #    list(range(k * vars_per_module, (k + 1) * vars_per_module))
  #    for k in range(K)
  #]

  return (out[0][0], out[0][1],
          out[1][0], out[1][1],
          block_bounds)


def _random_fill(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Fill `x` at positions where `mask` is True with per-gene random noise
  drawn from N(nanmean, nanstd) for that gene.

  This is statistically similar to a real observation of that gene (unlike a
  fixed mean or zero-fill sentinel), so the filled value itself gives no cue
  about missingness -- the model must rely on the explicit mask channel.
  """
  gene_mean = x.nanmean(dim=0)
  gene_var  = (x - gene_mean).pow(2).nanmean(dim=0)
  gene_std  = gene_var.clamp(min=1e-6).sqrt()
  return x.where(~mask, gene_mean + gene_std * torch.randn_like(x))

def get_random_mask(x:torch.Tensor, masked_fraction:float) -> torch.Tensor:
  """
  """
  type1_mask = torch.isnan(x)

  type2_mask = torch.rand(x.shape) < masked_fraction
  type2_mask = torch.logical_and( type2_mask, ~type1_mask )
  mask  = type2_mask | type1_mask

  x_masked = _random_fill(x, mask)
  # Augmentation: add noise to genuinely observed positions only -- masked
  # positions already received fresh per-gene random fill above.
  x_masked = x_masked + torch.randn_like(x_masked) * 0.05 * (~mask).float()
  return x_masked, mask, type2_mask

class GeneExpressionDataset(Dataset):
  def __init__(self, matrix:np.ndarray|torch.Tensor):
    if isinstance(matrix, np.ndarray):
      self.data = torch.tensor(matrix, dtype=torch.float32)
    else:
      self.data = matrix.to(dtype=torch.float32)
    self.mask = torch.tensor( ~np.isnan(matrix), dtype=torch.bool)
    self.data[~self.mask] = torch.randn((~self.mask).sum()).abs() * 0.05  # replace masked values with noise

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.mask[idx]

_debug_loss_bits = None
def masked_loss(
    x:            torch.Tensor,
    pred:         torch.Tensor,
    mask:         torch.Tensor,
    mu:           torch.Tensor,
    logvar:       torch.Tensor,
    gate_probs:   torch.Tensor|None = None,
    soft_weights: torch.Tensor|None = None,
    mask_type2:   torch.Tensor|None = None,
    beta:         float=1.0,
    gamma:        float=0.2,
    free_bits:    float=0.1,
    lambda_entropy:float=0.2,
    kl_override: torch.Tensor|None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Masked ELBO
  mask:       0=known, 1=masked/missing
  mask_type2: 0=observed/missing, 1=masked  (only deliberately masked values)
  beta:       VAE balance (low beta: focus on reconstruction quality; high beta: focus on structured latent space representation)
  gamma:      imputation balance (recon loss on known vs. recon loss on imputed positions)

  Handles both standard VAE (mu:(B,D) and MoVE (mu:(K,B,D)))
  """
  global _debug_loss_bits  # DEBUG ONLY

  assert(x.shape[-1]==pred.shape[-1])
  #x = x.reshape((x.shape[0],2,-1))[:,0,:]

  # MSE over observed positions
  _mask = ~mask
  sq_err = (pred - x).pow(2)
  recon_loss = (sq_err * _mask).sum() / _mask.sum().clamp(min=1)

  # Type 2 reconstruction
  if mask_type2 is not None:
    recon_loss_type2 = (sq_err * mask_type2.float()).sum() / mask_type2.sum().clamp(min=1)
    recon_loss += recon_loss_type2 * gamma
  else:
    recon_loss_type2 = torch.tensor(0.0)

  # KL:  -½ Σ (1 + log σ² - μ² - σ²)
  if kl_override is not None:
    # Pre-computed KL (e.g. VampPrior); free_bits clamping is skipped because
    # the VampPrior KL is a difference of log-densities and does not decompose
    # per dimension.
    kl_loss = kl_override

  elif mu.ndim==2:
    #kl_loss   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
    kl_loss    = kl_per_dim.clamp(min=free_bits).sum(dim=1).mean()
    #_debug_loss_bits = kl_per_dim.detach()  # DEBUG ONLY

  elif mu.ndim==3:
    # for MoVE: weighted KL over the components
    assert(soft_weights is not None)
    kl_per_dim  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) # (K,B,D)
    kl_per_comp = kl_per_dim.clamp(min=free_bits).sum(dim=-1)
    kl_loss     = (kl_per_comp.T * soft_weights).sum(dim=0).mean()  # weighted sum over components, then average over batch

  else:
    assert(False)

  # entropy loss - encourage uniform gate assignment
  if gate_probs is not None:
    avg_gate = gate_probs.mean(dim=0) # (K,)
    gating_entropy_loss = (avg_gate * avg_gate.log()).sum() # encourage uniform load
  else:
    gating_entropy_loss = torch.tensor(0.0)

  loss = recon_loss + beta * kl_loss + lambda_entropy * gating_entropy_loss
  return loss, recon_loss, recon_loss_type2, kl_loss


def random_points_on_hypersphere(
    num_points:    int,
    K:             int,
    N:             int,
    concentration: float = 0.0,
    n_clusters:    int   = 1,
    centre:        bool  = True,
    device:        str   = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
  """
  Generate random K-dimensional points constrained to an N-dimensional
  hypersphere (N < K), rotated so all K dimensions carry signal.

  Args:
      num_points:    number of points to generate.
      K:             dimensionality of the output space.
      N:             intrinsic dimensionality of the manifold (N < K).
      concentration: von Mises-Fisher concentration kappa >= 0.
                     0  => uniform on the sphere (original behaviour).
                     >0 => points are pulled towards their cluster centre;
                          higher values produce tighter clusters.
      n_clusters:    number of modes.  Each point is assigned to one cluster
                     uniformly at random, then sampled from a vMF distribution
                     centred on that cluster's mean direction.
      centre:        if True (default), subtract the empirical column mean from
                     the rotated output.  This ensures zero-mean cell states so
                     that non-interacting genes do not acquire a spurious DC
                     correlation from the offset of the cluster centres.
      device:        torch device string.

  Returns:
      rotated : (num_points, K) float tensor of cell-state coordinates.
      labels  : (num_points,)   int tensor of cluster assignments (0-indexed).
  """
  import warnings
  assert N < K, "N must be strictly less than K"

  if n_clusters > 1 and concentration < 1e-6:
    warnings.warn(
        "random_points_on_hypersphere: n_clusters > 1 but concentration ~ 0. "
        "Cluster centres have no effect on the distribution at concentration=0; "
        "all points will be drawn uniformly regardless of cluster assignment.",
        stacklevel=2,
    )

  # --- 1. Sample n_clusters centre directions on S^N ---
  centres = torch.randn(n_clusters, N + 1, device=device)
  centres = centres / torch.norm(centres, dim=-1, keepdim=True)  # (n_clusters, N+1)

  # --- 2. Assign each point to a cluster uniformly at random ---
  labels = torch.randint(0, n_clusters, (num_points,), device=device)  # (num_points,)

  # --- 3. Sample points from vMF(centre[label], concentration) ---
  if concentration < 1e-6:
    # Uniform on S^N — identical to the original implementation
    coords = torch.randn(num_points, N + 1, device=device)
    coords = coords / torch.norm(coords, dim=-1, keepdim=True)
  else:
    mu = centres[labels]  # (num_points, N+1) — each point's cluster centre

    # Tangent-plane perturbation: sample isotropic noise then project out the
    # component along mu so that z lies in the tangent plane at mu.
    z = torch.randn(num_points, N + 1, device=device)
    z = z - (z * mu).sum(dim=-1, keepdim=True) * mu   # project onto tangent plane
    z = z / torch.norm(z, dim=-1, keepdim=True)        # unit tangent vector

    # Mix: high concentration => near mu; concentration -> 0 => dominated by z (uniform)
    coords = mu + z / concentration
    coords = coords / torch.norm(coords, dim=-1, keepdim=True)  # renormalise to sphere

  # --- 4. Embed in K-dimensional space ---
  embedded = torch.zeros(num_points, K, device=device)
  embedded[:, :N+1] = coords

  # --- 5. Random rotation in R^K (Haar-uniform via QR) ---
  random_matrix = torch.randn(K, K, device=device)
  Q, R = torch.linalg.qr(random_matrix)
  signs = torch.sign(torch.diag(R))   # fix sign ambiguity -> uniform over SO(K)
  Q = Q * signs

  rotated = embedded @ Q.T  # (num_points, K)

  # --- 6. Optionally remove empirical mean ---
  if centre:
    rotated = rotated - rotated.mean(dim=0, keepdim=True)

  return rotated, labels


def make_synthetic_data4(
    block_state:    torch.Tensor,
    n_genes:        int = 200,
    n_blocks:       int = 5,
    rho_within:     float = 0.5,
    means:          torch.Tensor|None = None,
    stds:           torch.Tensor|None = None,
    allow_negative: bool = False,
    missing_rate:   float = 0.1,
    seed:           int = 42,
    device:         str|None = None,
    latent_clamp:   float = 15.0,
) -> tuple[torch.Tensor, np.ndarray]:
  """
  Explicit-factor variant of make_synthetic_data2/3 with caller-supplied block states.

  Each sample receives its block-state vector from `block_state`; gene-specific noise
  is sampled internally.  There is no global shared factor.

  For block k spanning genes [s, e):

      X[:, s:e] = sqrt(rho_within) * block_state[:, k:k+1]
                  + sqrt(1 - rho_within) * noise[:, s:e]

  If block_state[:, k] ~ N(0,1), the within-block correlation is rho_within.
  Between-block correlation is fully determined by the correlation structure the
  caller puts into block_state (no n_genes^2 covariance matrix is formed).

  Note: This version has no explicit support for discrete cell types (n_types==1)

  Masking (type 1): missing as np.nan.

  Returns:
      data: torch.Tensor(n_cells, n_genes)
      block_bounds
  """
  generator = torch.Generator(device=device)
  if seed is not None:
      generator.manual_seed(seed)

  #if means is not None or stds is not None:
  dtype = torch.float32
  mu    = means.to(dtype=dtype, device=device) if means is not None else torch.zeros(n_genes, dtype=dtype, device=device)
  sigma = stds.to( dtype=dtype, device=device) if stds  is not None else torch.ones( n_genes, dtype=dtype, device=device)

  assert block_state.shape[1] == n_blocks, (
      f"block_state has {block_state.shape[1]} columns but n_blocks={n_blocks}"
  )
  n_cells = block_state.shape[0]
  block_state = block_state.to(device=device)

  block_bounds = np.linspace(0, n_genes, n_blocks + 1).round().astype(int)

  std_block = np.sqrt(rho_within)
  std_noise = np.sqrt(max(1.0 - rho_within, 0.0))

  noise = torch.randn(n_cells, n_genes, device=device, generator=generator)
  X = torch.zeros(n_cells, n_genes, device=device)

  for k in range(n_blocks):
      s = block_bounds[k]
      e = block_bounds[k + 1]

      # Common component for this block (broadcast over genes)
      X[:, s:e] = std_block * block_state[:, k : k + 1] + std_noise * noise[:, s:e]

      if allow_negative:
          signs = torch.randint(0, 2, (e - s,), generator=generator, device=device) * 2 - 1
          X[:, s:e] = X[:, s:e] * signs.unsqueeze(0)

  # Map latent normal values to NB means and sample counts.
  # Use the logits parameterisation (rather than probs = r/(r+mean_nb)) to
  # avoid probs saturating to exactly 1.0 when mean_nb underflows to 0 in
  # float32 -- that violates NegativeBinomial's half-open probs constraint.
  # The exponent is also clamped to keep mean_nb in a numerically sane
  # range (heavy-tailed latents can otherwise blow up exp()).
  r = torch.tensor(10.0, device=device)  # use constant over-dispersion factor for all genes
  log_mean_nb = (X * sigma.unsqueeze(0) + mu.unsqueeze(0)).clamp(-latent_clamp, latent_clamp)
  logits = torch.log(r) - log_mean_nb
  counts = torch.distributions.NegativeBinomial(total_count=r, logits=logits).sample()
  X = torch.log1p(counts)

  dropout = torch.rand(X.shape, generator=generator, device=device) < missing_rate
  X[dropout] = float('nan')

  return X, block_bounds



# Define group hierarchy, with a 'membership fraction' associated with each group
groups_def = \
     {0: (0.20, { 6:(0.7,{23:(0.4,{})}),
                  7:(0.5,{24:(0.7,{})})}),
      1: (0.15, { 8:(0.6,{25:(0.7,{})}), 
                  9:(0.7,{26:(0.5,{})})}),
      2: (0.30, {10:(0.8,{27:(0.5,{})}), 
                 11:(0.4,{28:(0.7,{})}), 
                 12:(0.4,{})}),
      3: (0.30, {13:(0.4,{}), 
                 14:(0.3,{29:(0.6,{})}), 
                 15:(0.5,{})}),
      4: (0.35, {16:(0.6,{30:(0.5,{})}), 
                 17:(0.2,{})}),
      5: (0.20, {18:(0.3,{}),
                 19:(0.6,{}),
                 20:(0.4,{31:(0.6,{})}),
                 21:(0.4,{}),
                 22:(0.4,{})})
      }

def _draw_group_membership(g:dict, device=None):
  ret = []
  for gid, ginfo in g.items():
    membership_prob, descendants = ginfo

    if torch.rand(1, device=device).item() >= membership_prob: continue

    ret.append(gid)
    ret.extend( _draw_group_membership(descendants, device=device) )
  return tuple(ret)

def make_synthetic_data5(
    cell_state:     torch.Tensor,
    n_genes:        int = 200,
    corr_scale:     float = 1.0,
    noise_scale:    float = 0.001,
    n_gene_groups:  int = 32,  # must match groups_def
    means:          torch.Tensor|None = None,
    stds:           torch.Tensor|None = None,
    allow_negative: bool = False,
    missing_rate:   float = 0.1,
    seed:           int = 42,
    device:         str|None = None,
    groups_def:     dict = groups_def, # must match n_gene_groups
    influence_gamma:tuple[float,float] = (10.0,10.0),
    latent_clamp:   float = 15.0,
) -> tuple[torch.Tensor, np.ndarray]:
  """
  Hierarchical sparse-interaction synthetic gene-count data.

  Genes are stochastically assigned to groups in a 3-level hierarchy
  (groups_def).  Each cell's expression is driven by a cell_state vector
  that lives on a low-dimensional manifold (typically produced by
  random_points_on_hypersphere).

  Signal model (latent space, before NB sampling):

      group_membership : (n_genes, n_gene_groups)  -- sparse binary
      cell_state       : (n_cells, n_gene_groups)  -- manifold coords

      latent[:, g] = corr_scale * (group_membership @ cell_state.T)[g, :]
                     * influence_weight[g]           -- Gamma(10,10) ~ 1
                     + noise                         -- N(0, noise_scale)

  Then latent values are mapped to NegativeBinomial counts and log1p
  transformed, exactly as in make_synthetic_data4.

  Args:
      cell_state:    (n_cells, n_gene_groups) tensor of cell states.
      n_genes:       number of genes to simulate.
      corr_scale:    scales cell_state before the projection; controls the
                     overall signal amplitude / within-module correlation.
      noise_scale:   std of gene-level Gaussian noise added before NB sampling.
      n_gene_groups: number of groups (must match the IDs in groups_def).
      means:         optional per-gene offset (n_genes,) applied before exp().
      stds:          optional per-gene scale  (n_genes,) applied before exp().
      allow_negative: if True, each gene gets an independent ±1 sign flip.
      missing_rate:  fraction of entries set to NaN (MCAR dropout).
      seed:          RNG seed.
      device:        torch device string.
      groups_def:    hierarchy definition dict; must cover exactly n_gene_groups IDs.
      latent_clamp:  clamp applied to the pre-exp() latent exponent (in either
                     direction) before mapping to the NB mean. Protects against
                     heavy-tailed `influence_gamma` settings (e.g. low
                     shape/rate) producing extreme latents that would
                     otherwise overflow/underflow exp() in float32 and crash
                     NegativeBinomial's probs constraint.

  Returns:
      X             : (n_cells, n_genes) float32 tensor, log1p counts, NaN for missing.
      group_membership_np : (n_genes, n_gene_groups) numpy bool array.
  """
  generator = torch.Generator(device=device)
  if seed is not None:
      generator.manual_seed(seed)

  # n_cells is authoritative from the supplied tensor
  n_cells = cell_state.shape[0]
  assert cell_state.shape[1] == n_gene_groups, (
      f"cell_state has {cell_state.shape[1]} columns but n_gene_groups={n_gene_groups}"
  )

  # --- sparse gene-group membership matrix ---
  # group_membership[i, k] = 1 iff gene i belongs to group k
  group_membership = torch.zeros((n_genes, n_gene_groups), requires_grad=False)
  for gene_idx in range(n_genes):
    m = _draw_group_membership(groups_def)
    for mi in m:
      group_membership[gene_idx, mi] = 1.0
  group_membership = group_membership.to(device=device)

  ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
  #block_state_both = block_state_both[0,:].unsqueeze(0).expand(block_state_both.shape[0],block_state_both.shape[1])  # copy the first cell's state for all cells
  #block_state_both[:,:2] = 1 + torch.randn(10_000,2)*0.1
  #block_state_both[:,1:] = 0   # only group 1 has any genes
  ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##

  # Scale cell state by corr_scale so the caller controls signal amplitude
  cell_state = cell_state.to(device=device) * corr_scale  # (n_cells, n_gene_groups)

  # Per-(gene, cell) interaction-strength weights drawn from Gamma with mean~1
  # Gamma(concentration=10, rate=10) => mean=1, std~0.32; tight enough to be
  # roughly uniform but non-degenerate.
  # Note: changed to a more challenging Gamma(1.0,1.0) - The ideas is to make some genes highly dependent on group membership, and others to be nearly silent.
  # Consider changing to Gamma(0.5,0.5).
  influence_dist = torch.distributions.Gamma(
      torch.tensor([influence_gamma[0]], device=device),
      torch.tensor([influence_gamma[1]], device=device),
  )

  # (n_genes, n_cells): project cell states through sparse membership
  out = group_membership @ cell_state.T                   # (n_genes, n_cells)
  out *= influence_dist.sample(out.shape).squeeze(-1)     # per-entry strength
  out += torch.randn(out.shape, device=device,
                     generator=generator) * noise_scale   # gene-level noise

  # Transpose to (n_cells, n_genes) -- codebase convention
  X = out.T.contiguous()   # (n_cells, n_genes)

  # Optional per-gene sign flips (makes some genes anti-correlated within a group)
  if allow_negative:
      signs = (torch.randint(0, 2, (n_genes,),
                             generator=generator, device=device) * 2 - 1).float()
      X = X * signs.unsqueeze(0)

  # Per-gene offset / scale (consistent with make_synthetic_data4)
  dtype = torch.float32
  mu    = means.to(dtype=dtype, device=device) if means is not None \
          else torch.zeros(n_genes, dtype=dtype, device=device)
  sigma = stds.to( dtype=dtype, device=device) if stds  is not None \
          else torch.ones( n_genes, dtype=dtype, device=device)

  # Map latent values to NegativeBinomial counts, then log1p-normalise.
  # Use the logits parameterisation (rather than probs = r/(r+mean_nb)) to
  # avoid probs saturating to exactly 1.0 when mean_nb underflows to 0 in
  # float32 -- that violates NegativeBinomial's half-open probs constraint.
  # The exponent is also clamped to keep mean_nb in a numerically sane range,
  # since heavy-tailed influence weights (e.g. low influence_gamma) can
  # otherwise blow up the latent and overflow/underflow exp().
  r           = torch.tensor(10.0, device=device)
  log_mean_nb = (X * sigma.unsqueeze(0) + mu.unsqueeze(0)).clamp(-latent_clamp, latent_clamp)
  logits      = torch.log(r) - log_mean_nb
  counts      = torch.distributions.NegativeBinomial(total_count=r, logits=logits).sample()
  X           = torch.log1p(counts)

  # MCAR dropout -> NaN
  dropout = torch.rand(X.shape, generator=generator, device=device) < missing_rate
  X[dropout] = float('nan')

  return X, group_membership.cpu().numpy()


# --------------------------------------------------------------------------
# SERGIO-backed synthetic data (make_synthetic_data6)
# --------------------------------------------------------------------------
# SERGIO (https://github.com/PayamDiba/SERGIO, Dibaeinia & Sinha 2020) is a
# gene-regulatory-network (GRN) simulator: given a fixed GRN structure and a
# vector of master-regulator (MR) basal production rates per cell type
# ("bin"), it integrates a stochastic differential equation per gene and
# returns realistic steady-state single-cell expression profiles. It is not
# a dependency of this repo (not installed in this dev environment) -- import
# is deferred to inside make_synthetic_data6, with a clear error otherwise.

def _parse_sergio_targets_file(path: str) -> tuple[int, list[int]]:
  """
  Scan a SERGIO-format `input_file_targets` CSV (see SERGIO's build_graph
  documentation) to determine the total gene count and the set of
  master-regulator gene IDs, without depending on SERGIO's own parser (which
  uses the `np.int`/`np.float` aliases removed in numpy>=1.24).

  Row format (one row per *target* gene):
      target_id, n_regs, reg_id_1,...,reg_id_n, K_1,...,K_n[, coop_1,...,coop_n]
  Master regulators never appear as a target_id (column 0) in this file --
  they are exactly the regulator IDs that are never a target.

  Returns:
    n_genes: total number of genes (targets + master regulators). Gene IDs
             are asserted to form a contiguous zero-based range [0, n_genes),
             as required by SERGIO.
    mr_ids:  sorted list of inferred master-regulator gene IDs.
  """
  target_ids = set()
  reg_ids    = set()

  with open(path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      row = [c.strip() for c in row if c.strip() != '']
      if not row:
        continue
      target_id = int(float(row[0]))
      n_regs    = int(float(row[1]))
      target_ids.add(target_id)
      for r in row[2 : 2 + n_regs]:
        reg_ids.add(int(float(r)))

  mr_ids  = sorted(reg_ids - target_ids)
  all_ids = target_ids | reg_ids
  n_genes = len(all_ids)

  assert all_ids == set(range(n_genes)), (
      f"{path}: gene IDs must be a contiguous zero-based range [0, {n_genes}), "
      f"as required by SERGIO's build_graph."
  )
  return n_genes, mr_ids


def _write_sergio_regs_file(mr_state: torch.Tensor, mr_ids: list[int]) -> str:
  """
  Write a SERGIO-format `input_file_regs` CSV to a fresh temp file: one row
  per master regulator, `mr_id, rate_cluster_0, ..., rate_cluster_{K-1}`.

  Args:
    mr_state: (n_clusters, n_mrs) tensor of basal production rates; column i
              holds the rates (across clusters) for mr_ids[i].
    mr_ids:   ordered list of master-regulator gene IDs, len == n_mrs.

  Returns:
    Path to the written temp file. Caller is responsible for deleting it.
  """
  mr_state_np = mr_state.detach().cpu().numpy()
  fd, path = tempfile.mkstemp(suffix='.csv', prefix='sergio_regs_')
  with os.fdopen(fd, 'w', newline='') as f:
    writer = csv.writer(f)
    for i, mr_id in enumerate(mr_ids):
      writer.writerow([mr_id] + mr_state_np[:, i].tolist())
  return path


def _sample_cluster_sizes(
    n_cells:         int,
    n_clusters:      int,
    concentration:   float,
    min_per_cluster: int,
    rng:             np.random.Generator,
) -> np.ndarray:
  """
  Split n_cells among n_clusters using Dirichlet-sampled proportions (higher
  `concentration` => more even split), rounded to integers that sum exactly
  to n_cells and are each >= min_per_cluster.
  """
  assert n_cells >= n_clusters * min_per_cluster, (
      f"n_cells={n_cells} cannot satisfy min_cells_per_cluster={min_per_cluster} "
      f"across n_clusters={n_clusters}"
  )

  proportions = rng.dirichlet(np.full(n_clusters, concentration))
  raw         = proportions * n_cells
  counts      = np.maximum(np.floor(raw).astype(int), min_per_cluster)

  # Fix up rounding so counts sum exactly to n_cells (largest-remainder
  # method), while respecting the min_per_cluster floor.
  deficit = n_cells - counts.sum()
  if deficit > 0:
    remainder_order = np.argsort(-(raw - np.floor(raw)))
    for i in range(deficit):
      counts[remainder_order[i % n_clusters]] += 1
  elif deficit < 0:
    order = np.argsort(-counts)
    i = 0
    while deficit < 0:
      idx = order[i % n_clusters]
      if counts[idx] > min_per_cluster:
        counts[idx] -= 1
        deficit += 1
      i += 1

  assert counts.sum() == n_cells
  return counts


# --------------------------------------------------------------------------
# Building a SERGIO `input_file_targets` GRN-structure file from a reference
# regulatory network (e.g. TRRUST), rather than synthesizing topology from
# scratch: a connected subgraph of the reference network is sampled, made
# acyclic (SERGIO's layering algorithm requires a DAG), and written out with
# randomly-parameterised K/Hill coefficients.
# --------------------------------------------------------------------------

def _sample_from_dist(spec: tuple, size: int, rng: np.random.Generator) -> np.ndarray:
  """
  Sample `size` iid values from a small named distribution spec, used for
  both `k_dist` (interaction-strength magnitudes) and `hill_coeff_dist`
  (per-edge Hill/cooperativity coefficients):
    ('constant', v)
    ('uniform', low, high)
    ('lognormal', mu, sigma)
    ('normal', mu, sigma)
    ('choice', values[, weights])
  """
  kind = spec[0]
  if kind == 'constant':
    _, v = spec
    return np.full(size, v, dtype=np.float64)
  elif kind == 'uniform':
    _, low, high = spec
    return rng.uniform(low, high, size=size)
  elif kind == 'lognormal':
    _, mu, sigma = spec
    return rng.lognormal(mu, sigma, size=size)
  elif kind == 'normal':
    _, mu, sigma = spec
    return rng.normal(mu, sigma, size=size)
  elif kind == 'choice':
    values  = np.asarray(spec[1], dtype=np.float64)
    weights = spec[2] if len(spec) > 2 else None
    p = None
    if weights is not None:
      w = np.asarray(weights, dtype=np.float64)
      p = w / w.sum()
    return rng.choice(values, size=size, p=p)
  else:
    raise ValueError(f"unrecognized distribution spec kind: {kind!r}")


def _parse_reference_grn_edges(
    path:               str,
    delimiter:          str,
    regulator_col:      int,
    target_col:         int,
    mode_col:           int|None,
    activation_labels:  frozenset,
    repression_labels:  frozenset,
) -> list[tuple[str, str, str]]:
  """
  Parse a generic reference-GRN edge-list file (e.g. TRRUST's rawdata TSV)
  into deduplicated (regulator, target, sign_category) triples, where
  sign_category is one of 'activation', 'repression', 'ambiguous'
  ('ambiguous' covers unknown/unlabeled mode, and pairs with conflicting
  mode labels across duplicate rows -- e.g. TRRUST has 845 (regulator,
  target) pairs annotated as both Activation and Repression by different
  papers). Self-loops (regulator == target) are dropped, since SERGIO's
  input format cannot represent autoregulation.
  """
  needed_cols = [regulator_col, target_col] + ([mode_col] if mode_col is not None else [])
  max_col = max(needed_cols)

  pair_modes: dict[tuple[str, str], set[str]] = {}
  with open(path, 'r', newline='') as f:
    reader = csv.reader(f, delimiter=delimiter)
    for row in reader:
      if len(row) <= max_col:
        continue
      reg = row[regulator_col].strip()
      tgt = row[target_col].strip()
      if not reg or not tgt or reg == tgt:
        continue
      mode = row[mode_col].strip() if mode_col is not None else ''
      pair_modes.setdefault((reg, tgt), set()).add(mode)

  edges = []
  for (reg, tgt), modes in pair_modes.items():
    if modes <= activation_labels:
      sign_category = 'activation'
    elif modes <= repression_labels:
      sign_category = 'repression'
    else:
      sign_category = 'ambiguous'
    edges.append((reg, tgt, sign_category))
  return edges


def _sample_connected_subgraph(
    edges:             list[tuple[str, str, str]],
    n_genes:            int,
    rng:                np.random.Generator,
    max_seed_attempts:  int,
) -> set[str]:
  """
  Build an undirected adjacency from `edges` (ignoring sign) and
  snowball-sample a connected set of exactly `n_genes` node symbols via
  randomized BFS from a random seed node, retrying from different seed
  nodes (up to `max_seed_attempts`) if the first seed's connected component
  turns out to be smaller than n_genes.
  """
  adj: dict[str, set[str]] = {}
  for reg, tgt, _ in edges:
    adj.setdefault(reg, set()).add(tgt)
    adj.setdefault(tgt, set()).add(reg)

  nodes = list(adj.keys())
  assert len(nodes) >= n_genes, (
      f"reference GRN has only {len(nodes)} nodes (after dropping self-loops "
      f"and isolated-by-self-loop-only nodes), cannot sample a connected "
      f"subgraph of n_genes={n_genes}"
  )

  best_component_size = 0
  for _ in range(max_seed_attempts):
    seed_node = nodes[rng.integers(len(nodes))]
    visited  = {seed_node}
    order    = [seed_node]
    frontier = [seed_node]
    while frontier and len(visited) < n_genes:
      cur = frontier.pop(int(rng.integers(len(frontier))))
      neighbors = list(adj[cur])
      rng.shuffle(neighbors)
      for nb in neighbors:
        if nb not in visited:
          visited.add(nb)
          order.append(nb)
          frontier.append(nb)
          if len(visited) == n_genes:
            break
    best_component_size = max(best_component_size, len(visited))
    if len(visited) >= n_genes:
      return set(order[:n_genes])

  raise ValueError(
      f"could not find a connected subgraph of n_genes={n_genes} within "
      f"max_seed_attempts={max_seed_attempts}; largest reachable component "
      f"found was {best_component_size} nodes. Try a different seed, more "
      f"max_seed_attempts, or a smaller n_genes."
  )


def _eades_dag_order(
    nodes:            list[str],
    directed_edges:   list[tuple[str, str]],
) -> list[str]:
  """
  Greedy minimum-feedback-arc-set heuristic (Eades, Lin & Smyth 1993):
  repeatedly strip sinks (no remaining out-edges; prepend to the right
  sequence) and sources (no remaining in-edges; append to the left
  sequence); when neither exists (i.e. only cycles remain), remove the
  vertex maximizing (out-degree - in-degree) and append it to the left
  sequence. The concatenated order minimizes (heuristically) the number of
  edges that end up pointing "backward" and therefore have to be dropped to
  make the graph acyclic, as SERGIO's layering algorithm requires.

  Returns: `nodes` permuted into this order.
  """
  out_adj: dict[str, set[str]] = {n: set() for n in nodes}
  in_adj:  dict[str, set[str]] = {n: set() for n in nodes}
  for u, v in directed_edges:
    out_adj[u].add(v)
    in_adj[v].add(u)

  remaining = set(nodes)
  s1: list[str] = []
  s2: list[str] = []

  def remove(n):
    remaining.discard(n)
    for v in out_adj[n]:
      in_adj[v].discard(n)
    for u in in_adj[n]:
      out_adj[u].discard(n)
    out_adj[n].clear()
    in_adj[n].clear()

  while remaining:
    progressed = True
    while progressed:
      progressed = False
      for n in [n for n in remaining if not out_adj[n]]:
        if n in remaining:
          s2.insert(0, n)
          remove(n)
          progressed = True
      for n in [n for n in remaining if not in_adj[n]]:
        if n in remaining:
          s1.append(n)
          remove(n)
          progressed = True

    if remaining:
      u = max(remaining, key=lambda n: len(out_adj[n]) - len(in_adj[n]))
      s1.append(u)
      remove(u)

  return s1 + s2


def generate_sergio_grn_from_reference(
    reference_grn_path:            str,
    n_genes:                       int,
    output_path:                   str,
    delimiter:                     str        = '\t',
    regulator_col:                 int        = 0,
    target_col:                    int        = 1,
    mode_col:                      int|None   = 2,
    activation_labels:             frozenset  = frozenset({'Activation'}),
    repression_labels:             frozenset  = frozenset({'Repression'}),
    unknown_mode_repressor_prob:   float      = 0.5,
    k_dist:                        tuple      = ('uniform', 1.0, 5.0),
    hill_coeff_dist:               tuple      = ('constant', 2.0),
    max_seed_attempts:             int        = 20,
    seed:                          int|None   = 42,
) -> tuple[str, list[int], dict[int, str]]:
  """
  Generate a SERGIO-format `input_file_targets` GRN-structure CSV (consumed
  by make_synthetic_data6) whose *topology* is derived from a connected,
  DAG-ified subgraph of a real reference gene-regulatory network (e.g.
  TRRUST's rawdata TSV), rather than synthesized from scratch. Only the
  quantitative interaction parameters (K magnitude/sign, Hill/cooperativity
  coefficient) are randomly sampled per this function's configuration.

  Per-edge Hill/cooperativity coefficients are always written (one column
  per regulator, alongside the reg-id and K columns) -- there is no
  "shared_coop_state" option here. Correspondingly, the downstream
  make_synthetic_data6(..., shared_coop_state=...) call MUST be given a
  value <= 0, so that SERGIO's sim.build_graph() reads these per-edge
  coop-state columns from the file instead of overriding every interaction
  with a single global coefficient (see sergio.build_graph: coop states in
  the file are only read when shared_coop_state <= 0; otherwise they are
  silently ignored and the file's column layout would in fact be wrong,
  since build_graph then expects only 2 column-blocks per row instead of 3).

  Args:
    reference_grn_path: path to a reference regulatory-network edge-list
                  file, one row per (regulator, target[, mode, ...]) edge.
                  Defaults match TRRUST's rawdata format (tab-delimited,
                  columns: TF, target, mode in {Activation, Repression,
                  Unknown}, PMIDs).
    n_genes:      number of genes in the generated GRN (>= 2). A connected
                  subgraph of exactly this many nodes is sampled from the
                  reference network.
    output_path:  where to write the generated SERGIO-format CSV.
    delimiter/regulator_col/target_col/mode_col: parsing configuration for
                  reference_grn_path. mode_col=None treats every edge as
                  sign-unknown ('ambiguous').
    activation_labels/repression_labels: sets of mode-column string values
                  that mean "activation"/"repression"; any other value (or
                  a (regulator, target) pair with conflicting labels across
                  duplicate rows) is treated as sign-ambiguous.
    unknown_mode_repressor_prob: P(repressor) applied to sign-ambiguous
                  edges (known Activation/Repression edges always use their
                  literal sign).
    k_dist:       dist-spec (see _sample_from_dist) for interaction-strength
                  magnitude (sign is applied separately, per edge).
    hill_coeff_dist: dist-spec (see _sample_from_dist) for the per-edge
                  Hill/cooperativity coefficient.
    max_seed_attempts: number of random seed nodes to try when sampling a
                  connected subgraph of size n_genes before giving up.
    seed:         seeds all randomness in this function (subgraph sampling,
                  K/Hill/sign sampling).

  Returns:
    output_path:  same as the input arg, for convenience.
    mr_ids:       sorted list of inferred master-regulator gene ids (nodes
                  with zero in-degree after DAG-ification) -- pass directly
                  as make_synthetic_data6's `mr_gene_ids` (or leave that
                  None there, since it infers the same ids from the file).
    gene_id_to_symbol: dict mapping each generated gene id (0..n_genes-1) to
                  the reference network's original gene symbol, so results
                  can be related back to real genes (e.g. in a reference
                  scRNA dataset).

  Note: the reference network's directed edges are generally cyclic (e.g.
  TRRUST has mutual-regulation pairs and longer feedback loops), but
  SERGIO's layering algorithm requires an acyclic graph. Edges that violate
  the DAG order chosen by the min-feedback-arc-set heuristic are dropped
  entirely (not flipped), since flipping a literature-curated
  activation/repression edge's direction isn't supported by evidence. As a
  result, some nodes may lose all their edges; any such isolated node has
  exactly one of its original (pre-DAG-ification) edges force-restored,
  flipped if necessary to respect the DAG order (labeled 'ambiguous' sign in
  that case), purely to satisfy SERGIO's requirement that every gene
  participate in at least one edge.
  """
  assert n_genes >= 2, "n_genes must be >= 2"

  rng = np.random.default_rng(seed)

  edges = _parse_reference_grn_edges(
      reference_grn_path, delimiter, regulator_col, target_col, mode_col,
      frozenset(activation_labels), frozenset(repression_labels),
  )
  assert edges, f"no usable edges parsed from {reference_grn_path}"

  node_set = _sample_connected_subgraph(edges, n_genes, rng, max_seed_attempts)

  induced = [(reg, tgt, sign) for reg, tgt, sign in edges
             if reg in node_set and tgt in node_set]
  assert induced, (
      f"sampled connected subgraph of {len(node_set)} nodes has no induced "
      f"edges -- this should not happen for a connected sample"
  )

  order = _eades_dag_order(sorted(node_set), [(r, t) for r, t, _ in induced])
  assert len(order) == n_genes
  symbol_to_id = {sym: i for i, sym in enumerate(order)}

  survivors = [(reg, tgt, sign) for reg, tgt, sign in induced
               if symbol_to_id[reg] < symbol_to_id[tgt]]

  # Repair isolated nodes (no surviving in- or out-edge): SERGIO requires
  # every gene to participate in at least one edge.
  touched = set()
  for reg, tgt, _ in survivors:
    touched.add(reg); touched.add(tgt)
  isolated = [sym for sym in order if sym not in touched]
  if isolated:
    by_node: dict[str, list[tuple[str, str, str]]] = {}
    for reg, tgt, sign in induced:
      by_node.setdefault(reg, []).append((reg, tgt, sign))
      by_node.setdefault(tgt, []).append((reg, tgt, sign))
    for sym in isolated:
      candidates = by_node.get(sym, [])
      assert candidates, (
          f"isolated node {sym!r} has no induced edges at all -- "
          f"inconsistent with connected-subgraph sampling"
      )
      reg, tgt, sign = candidates[int(rng.integers(len(candidates)))]
      if symbol_to_id[reg] < symbol_to_id[tgt]:
        survivors.append((reg, tgt, sign))
      else:
        # symbol_to_id[tgt] < symbol_to_id[reg]: original direction violates
        # the DAG order (that's why it was dropped, and why sym ended up
        # isolated). Flip it to satisfy the order; causal direction is no
        # longer supported by evidence, so mark it ambiguous. (Equality is
        # impossible: self-loops were dropped during parsing.)
        reg, tgt = tgt, reg
        survivors.append((reg, tgt, 'ambiguous'))
      touched.add(reg); touched.add(tgt)

  missing = set(order) - touched
  assert not missing, f"failed to connect isolated nodes: {missing}"

  gene_id_to_symbol = {i: sym for i, sym in enumerate(order)}
  targets_regs: dict[int, list[tuple[int, str]]] = {}
  for reg, tgt, sign in survivors:
    reg_id, tgt_id = symbol_to_id[reg], symbol_to_id[tgt]
    targets_regs.setdefault(tgt_id, []).append((reg_id, sign))

  mr_ids = sorted(i for i in range(n_genes) if i not in targets_regs)

  rows = []
  for tgt_id in sorted(targets_regs.keys()):
    reg_list = targets_regs[tgt_id]
    n_regs   = len(reg_list)
    magnitudes  = _sample_from_dist(k_dist, n_regs, rng)
    coop_states = _sample_from_dist(hill_coeff_dist, n_regs, rng)

    reg_ids  = []
    k_values = []
    for (reg_id, sign), mag in zip(reg_list, magnitudes):
      if sign == 'activation':
        is_repressor = False
      elif sign == 'repression':
        is_repressor = True
      else:
        is_repressor = rng.random() < unknown_mode_repressor_prob
      reg_ids.append(reg_id)
      k_values.append(-abs(float(mag)) if is_repressor else abs(float(mag)))

    rows.append([tgt_id, n_regs] + reg_ids + k_values + [float(c) for c in coop_states])

  with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
      writer.writerow(row)

  return output_path, mr_ids, gene_id_to_symbol


def make_synthetic_data6(
    mr_state:               torch.Tensor,
    input_file_targets:     str,
    n_cells:                int = 800,
    mr_gene_ids:            list[int]|None = None,
    shared_coop_state:      float = 2.0,
    noise_params:           float|list = 1.0,
    noise_type:             str   = 'dpd',
    decays:                 float|list = 0.8,
    sampling_state:         int   = 15,
    dt:                     float = 0.01,
    cluster_conc:           float = 20.0,
    min_cells_per_cluster:  int   = 1,
    add_outlier_genes:      bool  = True,
    outlier_prob:           float = 0.01,
    outlier_mean:           float = 0.8,
    outlier_scale:          float = 1.0,
    add_lib_size_effect:    bool  = True,
    lib_size_mean:          float = 4.6,
    lib_size_scale:         float = 0.4,
    add_dropout:            bool  = True,
    dropout_shape:          float = 6.5,
    dropout_percentile:     float = 82.0,
    convert_to_umi_counts:  bool  = True,
    missing_rate:           float = 0.1,
    seed:                   int|None = 42,
    device:                 str|None = None,
) -> tuple[torch.Tensor, np.ndarray]:
  """
  SERGIO-backed synthetic gene-expression data: sample cells from n_clusters
  gene-regulatory-network (GRN) simulations, each driven by a single vector
  of master-regulator (MR) basal production rates.

  This is a thin wrapper around the SERGIO simulator
  (https://github.com/PayamDiba/SERGIO, Dibaeinia & Sinha 2020) that:
    1. treats each row of `mr_state` as the MR production-rate profile of one
       SERGIO "bin" (cell type) -- i.e. n_clusters = mr_state.shape[0]
       simulations sharing one caller-supplied, fixed GRN structure;
    2. splits the requested `n_cells` unevenly-but-not-too-unevenly across
       clusters via a Dirichlet draw (see `cluster_conc`);
    3. runs SERGIO's technical-noise pipeline (outlier genes -> library size
       -> dropout -> UMI counts), each stage independently toggleable;
    4. log1p-transforms the result and applies this codebase's usual MCAR
       `missing_rate` NaN masking (type 1 missing, as in
       make_synthetic_data/2/4/5) on top of SERGIO's own dropout;
    5. shuffles the output cells into random order (NOT grouped by cluster),
       since callers typically split the returned cells into train/test/eval
       groups by contiguous slicing.

  Args:
    mr_state:     (n_clusters, n_mrs) tensor of basal MR production rates.
                  Used as-is (must be >= 0) -- unlike make_synthetic_data4/5,
                  no latent->rate transform is applied here; the caller is
                  responsible for supplying values in a sane range for
                  SERGIO (its own demo data uses production rates in ~[1, 5]).
    input_file_targets: path to a SERGIO-format GRN-structure CSV (see
                  SERGIO's build_graph documentation) -- caller-supplied,
                  fixed regulatory structure (independent of mr_state).
    n_cells:      total number of output cells (across all clusters).
    mr_gene_ids:  optional explicit ordering of master-regulator gene IDs
                  matching mr_state's columns. If None, inferred from
                  `input_file_targets` (regulators that never appear as a
                  target) and sorted ascending.
    shared_coop_state, noise_params, noise_type, decays, sampling_state, dt:
                  passed straight through to SERGIO's sergio()/build_graph()
                  (defaults match SERGIO's own steady-state demo).
    cluster_conc: Dirichlet concentration controlling how evenly n_cells is
                  split across clusters (higher => more even).
    min_cells_per_cluster: minimum cells guaranteed per cluster.
    add_outlier_genes/add_lib_size_effect/add_dropout/convert_to_umi_counts:
                  toggle stages of SERGIO's technical-noise pipeline (applied
                  in this fixed order, per SERGIO's documented usage).
    outlier_prob/outlier_mean/outlier_scale: sim.outlier_effect() params.
    lib_size_mean/lib_size_scale:            sim.lib_size_effect() params.
    dropout_shape/dropout_percentile:        sim.dropout_indicator() params.
    missing_rate: MCAR type-1 missing rate applied after SERGIO's own noise
                  (NaN, consistent with every other make_synthetic_data*).
    seed:         also seeds the *global* numpy RNG before invoking SERGIO,
                  since SERGIO's own methods call `np.random.*` directly
                  rather than accepting a Generator/seed.
    device:       torch device for the returned tensor.

  Returns:
    X:              (n_cells, n_genes) float32 tensor, log1p-transformed,
                    NaN at type-1-missing positions. Cell order is randomly
                    shuffled (not grouped by cluster).
    cluster_labels: (n_cells,) int64 numpy array; cluster_labels[i] gives the
                    mr_state row (0-indexed) that generated cell i.

  Note: requires the `SERGIO` package from
  https://github.com/PayamDiba/SERGIO -- clone it and add the repo root to
  PYTHONPATH. It is NOT pip-installable: the repo has no setup.py/pyproject
  (so `import SERGIO` must resolve to a git checkout on PYTHONPATH, and the
  class must be imported as `from SERGIO.sergio import sergio` since
  SERGIO/__init__.py does not re-export it), and the `sergio-scSim` PyPI
  package is a broken/unrelated distribution: despite its README describing
  this same classic API, the code it actually ships (checked v1.9.1-1.9.3)
  is an incompatible GRN/MR/Noise-class rewrite with a completely different
  interface -- do not use it. It is not installed in this dev environment,
  so this function can only be syntax-checked here, not executed. SERGIO
  v1.0.0's build_graph() also uses the `np.int`/`np.float` numpy aliases
  removed in numpy>=1.24; a narrow, local shim is applied around the SERGIO
  calls below (and reverted immediately after) to work around this without
  patching SERGIO itself.
  """
  try:
    from SERGIO.sergio import sergio
  except ImportError as e:
    raise ImportError(
        "make_synthetic_data6 requires the 'SERGIO' package from "
        "https://github.com/PayamDiba/SERGIO. Clone it and add the repo "
        "root to PYTHONPATH (it is not pip-installable, and the "
        "'sergio-scSim' PyPI package ships an incompatible, unrelated "
        "rewrite -- do not use it)."
    ) from e

  assert mr_state.ndim == 2, "mr_state must be (n_clusters, n_mrs)"
  n_clusters, n_mrs = mr_state.shape
  assert (mr_state >= 0).all(), (
      "mr_state values are used directly as SERGIO basal production rates "
      "and must be non-negative."
  )

  n_genes, mr_ids_inferred = _parse_sergio_targets_file(input_file_targets)
  if mr_gene_ids is None:
    mr_ids = mr_ids_inferred
  else:
    assert set(mr_gene_ids) == set(mr_ids_inferred), (
        f"mr_gene_ids {sorted(mr_gene_ids)} does not match the master "
        f"regulators inferred from {input_file_targets}: {mr_ids_inferred}"
    )
    mr_ids = mr_gene_ids
  assert n_mrs == len(mr_ids), (
      f"mr_state has {n_mrs} columns but the GRN has {len(mr_ids)} master "
      f"regulators"
  )

  rng = np.random.default_rng(seed)
  counts = _sample_cluster_sizes(
      n_cells, n_clusters, cluster_conc, min_cells_per_cluster, rng)
  number_sc = int(counts.max())

  if seed is not None:
    # SERGIO's own methods (simulate/dropout_indicator/convert_to_UMIcounts/
    # outlier_effect/lib_size_effect/...) draw from the *global* numpy RNG
    # rather than accepting a Generator, so this is required for
    # reproducibility of the SERGIO-side randomness.
    np.random.seed(seed)

  # Narrow compatibility shim for SERGIO v1.0.0's use of the numpy np.int/
  # np.float aliases (removed in numpy>=1.24). Scoped tightly around the
  # SERGIO calls and reverted in `finally` below.
  _np_int_bak   = getattr(np, 'int',   None)
  _np_float_bak = getattr(np, 'float', None)
  np.int, np.float = int, float

  regs_path = _write_sergio_regs_file(mr_state, mr_ids)
  try:
    sim = sergio(
        number_genes=n_genes,
        number_bins=n_clusters,
        number_sc=number_sc,
        noise_params=noise_params,
        noise_type=noise_type,
        decays=decays,
        dynamics=False,
        sampling_state=sampling_state,
        dt=dt,
    )
    sim.build_graph(input_file_targets, regs_path, shared_coop_state)
    sim.simulate()
    expr = sim.getExpressions()   # (n_clusters, n_genes, number_sc)

    if add_outlier_genes:
      expr = sim.outlier_effect(expr, outlier_prob, outlier_mean, outlier_scale)
    if add_lib_size_effect:
      _, expr = sim.lib_size_effect(expr, lib_size_mean, lib_size_scale)
    if add_dropout:
      binary_ind = sim.dropout_indicator(expr, dropout_shape, dropout_percentile)
      expr = np.multiply(binary_ind, expr)
    if convert_to_umi_counts:
      expr = sim.convert_to_UMIcounts(expr)

  finally:
    os.remove(regs_path)
    if _np_int_bak is None: del np.int
    else:                   np.int = _np_int_bak
    if _np_float_bak is None: del np.float
    else:                     np.float = _np_float_bak

  # Subsample counts[i] cells (without replacement) from each cluster's
  # number_sc simulated pool, then shuffle so the output isn't grouped by
  # cluster (callers typically slice train/test/eval contiguously).
  cols   = []
  labels = []
  for i in range(n_clusters):
    chosen = rng.choice(number_sc, size=int(counts[i]), replace=False)
    cols.append(expr[i][:, chosen])                      # (n_genes, counts[i])
    labels.append(np.full(int(counts[i]), i, dtype=np.int64))

  X_np           = np.concatenate(cols, axis=1)    # (n_genes, n_cells)
  cluster_labels = np.concatenate(labels, axis=0)  # (n_cells,)

  perm           = rng.permutation(n_cells)
  X_np           = X_np[:, perm]
  cluster_labels = cluster_labels[perm]

  X = torch.tensor(X_np.T, dtype=torch.float32, device=device)  # (n_cells, n_genes)
  X = torch.log1p(X.clamp(min=0))

  # MCAR dropout -> NaN (type-1 missing, applied after SERGIO's own noise)
  generator = torch.Generator(device=device)
  if seed is not None:
    generator.manual_seed(seed)
  dropout = torch.rand(X.shape, generator=generator, device=device) < missing_rate
  X[dropout] = float('nan')

  return X, cluster_labels


# --------------------------------------------------------------------------
# Real-data reference loading & summary statistics, for calibrating
# make_synthetic_data6's SERGIO-derived parameters against a real scRNA-seq
# reference matrix.
# --------------------------------------------------------------------------
#
# Typical usage:
#   target_stats = compute_summary_stats(load_reference_h5ad("ref.h5ad"))
#   X_sim, _     = make_synthetic_data6(..., missing_rate=0.0)
#   sim_stats    = compute_summary_stats(X_sim)
#   # caller compares target_stats vs sim_stats (e.g. weighted L2 over the
#   # scalar entries) as a tuning objective for the SERGIO-side parameters.
#
# missing_rate is deliberately 0.0 above: make_synthetic_data6's MCAR NaN
# mask is a separate, later-added imputation-task artifact (applied on top
# of SERGIO's own dropout) with no counterpart in a real reference matrix,
# so it should be left out when generating candidates to match against
# real-data target stats, and only added afterward when producing actual
# train/eval matrices.

def load_reference_h5ad(
    path:         str,
    layer:        str|None = None,
    n_top_genes:  int|None = None,
    device:       str|None = None,
) -> torch.Tensor:
  """
  Load a real scRNA-seq reference matrix from an .h5ad file and bring it to
  the same (n_cells, n_genes) float32, log1p-transformed tensor format
  produced by make_synthetic_data6 (minus its MCAR NaN mask -- a real
  reference has no such artificial missingness; see the module-level note
  above on how to compare fairly against make_synthetic_data6 output).

  Args:
    path:         path to an .h5ad file (AnnData format).
    layer:        if given, read raw counts from adata.layers[layer]
                  instead of adata.X. If None (default), adata.X is assumed
                  to hold raw/UMI counts (not already normalized or
                  log-transformed) -- matching make_synthetic_data6's own
                  convert_to_UMIcounts -> log1p convention.
    n_top_genes:  if given, keep only the `n_top_genes` genes with the
                  highest mean raw-count expression (a simple,
                  dependency-free stand-in for highly-variable-gene
                  selection). Useful for keeping the reference at a
                  comparable gene-count scale to a small SERGIO GRN. Gene
                  *identity*/order is not otherwise aligned with any
                  synthetic GRN -- compute_summary_stats compares
                  aggregate distributions, not per-gene identity.
    device:       torch device for the returned tensor.

  Returns:
    X: (n_cells, n_genes) float32 tensor, log1p-transformed, no NaNs.
       Genes with zero total raw count across all cells are dropped first
       (they carry no information and would produce degenerate
       mean/variance statistics downstream).

  Note: requires the `anndata` package (`pip install anndata`) -- not
  installed in this dev environment, so this can only be syntax-checked
  here, not executed (same situation as make_synthetic_data6's SERGIO
  dependency).
  """
  try:
    import anndata
  except ImportError as e:
    raise ImportError(
        "load_reference_h5ad requires the 'anndata' package: "
        "pip install anndata"
    ) from e
  import scipy.sparse

  adata = anndata.read_h5ad(path)
  raw = adata.layers[layer] if layer is not None else adata.X
  if scipy.sparse.issparse(raw):
    raw = raw.toarray()
  X_np = np.asarray(raw, dtype=np.float64)
  assert X_np.ndim == 2, f"expected a 2D matrix, got shape {X_np.shape}"

  # Drop genes with zero total count across all cells: undefined
  # variance/log-mean, and uninformative for the summary statistics below.
  gene_totals = X_np.sum(axis=0)
  X_np = X_np[:, gene_totals > 0]

  if n_top_genes is not None and X_np.shape[1] > n_top_genes:
    gene_means = X_np.mean(axis=0)
    top_idx    = np.sort(np.argsort(gene_means)[::-1][:n_top_genes])
    X_np       = X_np[:, top_idx]

  X = torch.tensor(X_np, dtype=torch.float32, device=device)
  X = torch.log1p(X.clamp(min=0))
  return X


def compute_summary_stats(
    X:                 torch.Tensor,
    n_pca_components:  int      = 10,
    n_corr_genes:      int|None = 500,
    percentiles:       tuple    = (5, 25, 50, 75, 95),
    seed:              int|None = 0,
) -> dict:
  """
  Compute a dict of distributional summary statistics describing a
  (n_cells, n_genes) log1p-scale gene-expression matrix -- callable
  identically on make_synthetic_data6's output (X only; ignore its
  cluster_labels) and on load_reference_h5ad's output, so the two can be
  compared as a tuning target/candidate pair. Uses NaN-aware reductions
  throughout, so it is safe to call on MCAR-masked synthetic data too, but
  for a fair comparison against a real reference the synthetic candidate
  should be generated with missing_rate=0.0 (see this module's top-level
  note) since the real reference has no NaNs to match against.

  Args:
    X:                 (n_cells, n_genes) tensor, log1p-scale, NaN at
                       missing (unobserved) positions (or no NaNs at all).
    n_pca_components:  number of leading PCA components to report the
                       explained-variance ratio for (capped at
                       min(n_cells, n_genes) - 1 internally).
    n_corr_genes:      if n_genes exceeds this, a random subsample of this
                       many genes is used for the pairwise gene-gene
                       correlation statistics (for tractability). None
                       disables subsampling (uses all genes).
    percentiles:       percentiles reported for each *_p<pct> distribution
                       summary entry.
    seed:              seeds the gene subsampling for the correlation
                       statistics (does not affect any other statistic).

  Returns:
    dict with (all values are Python floats or 1D numpy arrays):
      n_cells, n_genes
      gene_mean_{mean,std,p<pct>...}: distribution, across genes, of each
        gene's mean (log1p-scale) expression.
      gene_var_{mean,std,p<pct>...}:  same, for each gene's variance.
      mean_var_log_slope, mean_var_log_corr: slope and Pearson correlation
        of log(gene variance) regressed on log(gene mean) across genes
        (the classic mean-variance/overdispersion relationship), computed
        only over genes with strictly positive mean and variance.
      log_lib_size_{mean,std,p<pct>...}: distribution, across cells, of
        log1p(per-cell total raw count) -- directly comparable in scale to
        SERGIO's own lib_size_mean/lib_size_scale parameters.
      zero_frac: overall fraction of observed entries that are exactly 0.
      dropout_curve_bin_edges: bin edges (over per-gene mean log1p
        expression) used for the binned dropout curve below.
      dropout_curve_zero_frac: mean per-gene zero-fraction within each bin
        of dropout_curve_bin_edges (NaN for empty bins) -- a compact,
        gene-count-independent summary of the zero-fraction/dropout curve,
        comparable to SERGIO's dropout_shape/dropout_percentile knobs.
      pca_explained_variance_ratio: (n_pca_components,) array.
      gene_corr_abs_{mean,std,p50,p90}: distribution summary of |pairwise
        gene-gene Pearson correlation| (upper triangle, off-diagonal),
        computed on up to n_corr_genes genes.
  """
  assert X.dim() == 2, f"expected a 2D (n_cells, n_genes) tensor, got shape {tuple(X.shape)}"
  n_cells, n_genes = X.shape
  X_np = X.detach().to(torch.float64).cpu().numpy()
  obs_mask = ~np.isnan(X_np)

  stats: dict = {"n_cells": n_cells, "n_genes": n_genes}

  def _dist_summary(prefix: str, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    stats[f"{prefix}_mean"] = float(np.mean(finite)) if finite.size else float('nan')
    stats[f"{prefix}_std"]  = float(np.std(finite))  if finite.size else float('nan')
    for p in percentiles:
      stats[f"{prefix}_p{p}"] = (
          float(np.percentile(finite, p)) if finite.size else float('nan')
      )

  # --- per-gene mean & variance distributions ---
  with np.errstate(invalid='ignore'):
    gene_mean = np.nanmean(np.where(obs_mask, X_np, np.nan), axis=0)
    gene_var  = np.nanvar(np.where(obs_mask, X_np, np.nan), axis=0)
  _dist_summary("gene_mean", gene_mean)
  _dist_summary("gene_var", gene_var)

  # --- mean-variance relationship (log-log regression across genes) ---
  mv_mask = np.isfinite(gene_mean) & np.isfinite(gene_var) & (gene_mean > 0) & (gene_var > 0)
  if mv_mask.sum() >= 2:
    log_mean = np.log(gene_mean[mv_mask])
    log_var  = np.log(gene_var[mv_mask])
    slope, _ = np.polyfit(log_mean, log_var, 1)
    corr     = float(np.corrcoef(log_mean, log_var)[0, 1])
  else:
    slope, corr = float('nan'), float('nan')
  stats["mean_var_log_slope"] = float(slope)
  stats["mean_var_log_corr"]  = corr

  # --- per-cell library size (back out of log1p to raw-count scale) ---
  raw = np.expm1(np.where(obs_mask, X_np, 0.0))
  raw[~obs_mask] = 0.0
  lib_size = raw.sum(axis=1)
  _dist_summary("log_lib_size", np.log1p(lib_size))

  # --- overall zero fraction (among observed entries) ---
  n_obs = obs_mask.sum()
  stats["zero_frac"] = (
      float(np.sum((X_np == 0) & obs_mask) / n_obs) if n_obs > 0 else float('nan')
  )

  # --- binned dropout curve: per-gene zero-frac vs per-gene mean expression ---
  gene_obs_count = obs_mask.sum(axis=0)
  gene_zero_frac = np.divide(
      np.sum((X_np == 0) & obs_mask, axis=0), np.maximum(gene_obs_count, 1),
      out=np.full(n_genes, np.nan), where=gene_obs_count > 0,
  )
  valid_genes = np.isfinite(gene_mean) & np.isfinite(gene_zero_frac)
  n_bins = min(10, int(valid_genes.sum()))
  if n_bins >= 1:
    bin_edges = np.quantile(gene_mean[valid_genes], np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    n_bins    = len(bin_edges) - 1
  if n_bins >= 1:
    bin_idx = np.clip(np.digitize(gene_mean[valid_genes], bin_edges[1:-1], right=True), 0, n_bins - 1)
    curve = np.full(n_bins, np.nan)
    zf_valid = gene_zero_frac[valid_genes]
    for b in range(n_bins):
      sel = bin_idx == b
      if sel.any():
        curve[b] = float(np.mean(zf_valid[sel]))
    stats["dropout_curve_bin_edges"]   = bin_edges
    stats["dropout_curve_zero_frac"]   = curve
  else:
    stats["dropout_curve_bin_edges"] = np.array([])
    stats["dropout_curve_zero_frac"] = np.array([])

  # --- global structure: PCA explained-variance ratio, gene-gene corr ---
  # NaNs filled with the per-gene mean for PCA/correlation only (SVD and
  # corrcoef require fully-observed input); the nan-aware stats above are
  # unaffected by this.
  fill = np.where(np.isfinite(gene_mean), gene_mean, 0.0)
  X_filled = np.where(obs_mask, X_np, fill[np.newaxis, :])

  k = max(0, min(n_pca_components, n_cells - 1, n_genes - 1))
  if k >= 1:
    Xc = X_filled - X_filled.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    explained_var = s ** 2
    total = explained_var.sum()
    ratio = explained_var / total if total > 0 else np.zeros_like(explained_var)
    stats["pca_explained_variance_ratio"] = ratio[:k]
  else:
    stats["pca_explained_variance_ratio"] = np.array([])

  rng = np.random.default_rng(seed)
  if n_corr_genes is not None and n_genes > n_corr_genes:
    gene_idx = np.sort(rng.choice(n_genes, size=n_corr_genes, replace=False))
  else:
    gene_idx = np.arange(n_genes)
  if len(gene_idx) >= 2:
    sub = X_filled[:, gene_idx]
    with np.errstate(invalid='ignore'):
      corr_matrix = np.corrcoef(sub, rowvar=False)
    iu = np.triu_indices_from(corr_matrix, k=1)
    abs_corr = np.abs(corr_matrix[iu])
    abs_corr = abs_corr[np.isfinite(abs_corr)]
  else:
    abs_corr = np.array([])
  if abs_corr.size:
    stats["gene_corr_abs_mean"] = float(np.mean(abs_corr))
    stats["gene_corr_abs_std"]  = float(np.std(abs_corr))
    stats["gene_corr_abs_p50"]  = float(np.percentile(abs_corr, 50))
    stats["gene_corr_abs_p90"]  = float(np.percentile(abs_corr, 90))
  else:
    stats["gene_corr_abs_mean"] = float('nan')
    stats["gene_corr_abs_std"]  = float('nan')
    stats["gene_corr_abs_p50"]  = float('nan')
    stats["gene_corr_abs_p90"]  = float('nan')

  return stats

