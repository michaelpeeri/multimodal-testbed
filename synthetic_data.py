import math
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

