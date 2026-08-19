import argparse
import csv
import json
import math
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    coherency_bias:                float      = 0.0,
    canalization_strength:         float      = 0.0,
    balancing_strength:            float      = 0.0,
    path_decay:                    float      = 0.9,
    max_seed_attempts:             int        = 20,
    seed:                          int|None   = 42,
    diagnostics:                   dict|None  = None,
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
    coherency_bias: in [0, 1]. Biases sign-ambiguous edges' repressor/
                  activator draw toward agreeing with the "sign" (relative
                  to this target's winning master regulator, see below) of
                  the regulator carrying that edge, instead of the plain
                  i.i.d. unknown_mode_repressor_prob coin flip.
                  0.0 (default) reproduces the unbiased draw exactly (same
                  rng.random() call, same call order, byte-identical
                  output to omitting this parameter entirely); 1.0 always
                  aligns (deterministically, no rng.random() consumed).
                  Literal Activation/Repression edges are NEVER flipped by
                  this or any other new parameter below -- only sign-
                  ambiguous edges are affected, since flipping a
                  literature-curated edge's direction isn't supported by
                  evidence (same principle as unknown_mode_repressor_prob
                  above).
    canalization_strength: in [0, 1]. Reweights every edge's sampled K
                  magnitude by (1 + canalization_strength) if that edge's
                  regulator's dominant master-regulator (and its
                  contribution's sign) matches this target's winning
                  master regulator/sign, else by (1 - canalization_strength)
                  (weight clamped to >= 0.05). Applies to ALL edges
                  (literal-sign and ambiguous alike), since K-magnitude was
                  always a free/synthetic quantity regardless of the
                  edge's literature-derived sign. 0.0 (default) leaves
                  every magnitude's weight at exactly 1.0 (no-op,
                  byte-identical to omitting this parameter). Each target's
                  full K-vector is then rescaled so sum(|K_i|) (SERGIO's
                  production-rate ceiling for that target) matches what it
                  would have been without this reweighting -- i.e. this
                  knob only reallocates a target's fixed production budget
                  toward/away from specific regulators, it never changes
                  the target's own overall expression scale on average
                  (see the main loop's own comment for the mechanism this
                  guards against: uncompensated reweighting otherwise
                  mechanically links "more canalization" to "lower
                  gene_mean/gene_var/library size"). Also exactly a no-op
                  at canalization_strength=0.0 (scale is always 1.0 then).
    balancing_strength: in [0, inf). Non-negative fairness penalty applied
                  when picking each target's "winning" master regulator
                  (see below): winner = argmax_mr(vote[mr] -
                  balancing_strength * mr_load[mr]), where mr_load[mr]
                  accumulates as mr "wins" targets. Prevents whichever
                  master regulator already has the largest topological
                  fan-out from entrenching further (which would only
                  concentrate more variance into a single PCA component,
                  the opposite of this mechanism's purpose). 0.0 (default)
                  is a pure unbalanced argmax (still deterministic -- has
                  no effect on output when coherency_bias=
                  canalization_strength=0.0, since the winner is then
                  unused).
    path_decay:   in (0, 1]. Per-hop attenuation applied when propagating a
                  gene's "dominant master regulator" signal strength
                  forward to its own downstream targets, modeling the
                  intuition that a master regulator's influence weakens
                  (in terms of how reliably its sign propagates) with
                  topological distance. Has no effect on output unless
                  coherency_bias > 0 or canalization_strength > 0.

    Together, coherency_bias/canalization_strength/balancing_strength/
    path_decay implement an opt-in "coherent feed-forward loop" /
    canalizing-function bias (see Alon, "An Introduction to Systems
    Biology": coherent vs. incoherent feed-forward loops; Kauffman 1969:
    canalizing Boolean functions) on top of the otherwise-i.i.d. per-edge
    parameter sampling: each target gene picks a single "winning" master
    regulator (by fairness-adjusted vote among its regulators' own
    dominant master regulators -- a hard, propagated winner-take-all
    assignment, not a soft blend across multiple master regulators), and
    edges that agree with that winner get biased sign (if ambiguous) and
    boosted magnitude, while disagreeing edges get shrunk magnitude. This
    is a single left-to-right pass merged into the main loop below
    (requires no separate path-enumeration/BFS step), exploiting the fact
    that `survivors` only contains edges with reg_id < tgt_id (see below),
    so every regulator's own winning-master-regulator assignment is always
    already computed by the time a target that depends on it is reached.
    All four parameters default to values that reproduce today's
    unbiased/i.i.d. behavior exactly.
    max_seed_attempts: number of random seed nodes to try when sampling a
                  connected subgraph of size n_genes before giving up.
    seed:         seeds all randomness in this function (subgraph sampling,
                  K/Hill/sign sampling).
    diagnostics:  optional dict; if given, populated in-place (this
                  function's own return value/signature is unaffected --
                  pass a fresh {} and read it back afterward) with
                  per-target-gene and summary information about the
                  coherency_bias/canalization_strength/balancing_strength
                  mechanism's behavior, for offline analysis (e.g. "did
                  balancing_strength actually spread out which master
                  regulators win, or is one MR still entrenching most
                  targets?"). Zero-cost when None (the default) -- no
                  extra computation is performed, only bookkeeping of
                  values already computed for the main algorithm. Adds
                  the following keys (all NaN/empty-safe even if every
                  coherency/canalization/balancing param is 0.0, i.e. this
                  is meaningful to inspect even for "unbiased" runs):
                    "tgt_ids":            (n_targets,) list, sorted target
                                          gene ids, parallel to every other
                                          per-target array below.
                    "winner_mr":          (n_targets,) list, this target's
                                          winning master regulator id.
                    "vote_margin":        (n_targets,) list, winner's vote
                                          minus runner-up's vote (BEFORE
                                          the balancing_strength penalty is
                                          applied) -- large margin means
                                          the "natural" (unbalanced) winner
                                          was clear-cut; near-zero means
                                          balancing_strength (if nonzero)
                                          had a real say in the outcome, or
                                          the contest was close regardless.
                                          NaN if the target has only one
                                          distinct voting MR (no contest).
                    "n_regs":             (n_targets,) list, number of
                                          regulators for this target.
                    "n_ambiguous":        (n_targets,) list, number of
                                          sign-ambiguous edges among this
                                          target's regulators.
                    "n_ambiguous_flipped": (n_targets,) list, of those
                                          ambiguous edges, how many drew a
                                          sign DIFFERENT from what a plain
                                          unknown_mode_repressor_prob coin
                                          flip's expectation would have
                                          been biased toward (i.e. how many
                                          were actually swayed by
                                          coherency_bias -- 0 whenever
                                          coherency_bias == 0.0, since then
                                          p_repressor is never altered).
                    "n_aligned_edges":    (n_targets,) list, number of this
                                          target's edges whose (dominant
                                          MR, sign) matches (winner_mr,
                                          this target's own net sign) --
                                          the same "aligned" condition
                                          canalization_strength rewards.
                    "propagated_strength": (n_targets,) list, the signed
                                          strength stored in mr_profile for
                                          this target (path_decay-scaled
                                          mean of aligned incoming
                                          strengths; 0.0 if no aligned
                                          edges).
                    "mr_load":            dict[mr_id, float], final
                                          accumulated |strength| "load" per
                                          master regulator (same dict the
                                          algorithm itself uses for the
                                          balancing penalty).
                    "n_distinct_winners": int, number of distinct master
                                          regulators that won at least one
                                          target (out of len(mr_ids)
                                          possible) -- a direct, single-
                                          number summary of winner-take-all
                                          concentration vs. spread.
                    "top1_winner_share":  float in [0, 1], the single
                                          largest number of targets won by
                                          any one master regulator, divided
                                          by the total number of targets --
                                          1.0 means total entrenchment by
                                          one MR, low values mean winning
                                          is spread out. NaN if there are
                                          no targets (n_genes == len(mr_ids)).

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

  # mr_profile[gene_id] = (dominant_mr_id, signed_strength): each master
  # regulator is its own dominant MR with unit strength; every other gene
  # gets this assigned when it's processed as a target below (always
  # possible -- see mr_profile[reg_id] access below -- since `survivors`
  # only contains edges with reg_id < tgt_id, so every regulator of a
  # target has necessarily already been processed, either as an MR seed or
  # as an earlier target, by the time that target is reached in
  # sorted(targets_regs.keys()) order).
  mr_profile: dict[int, tuple[int, float]] = {mr: (mr, 1.0) for mr in mr_ids}
  mr_load:    dict[int, float]             = {mr: 0.0 for mr in mr_ids}

  if diagnostics is not None:
    diag_tgt_ids:              list[int]   = []
    diag_winner_mr:             list[int]   = []
    diag_vote_margin:           list[float] = []
    diag_n_regs:                list[int]   = []
    diag_n_ambiguous:           list[int]   = []
    diag_n_ambiguous_flipped:   list[int]   = []
    diag_n_aligned_edges:       list[int]   = []
    diag_propagated_strength:   list[float] = []

  rows = []
  for tgt_id in sorted(targets_regs.keys()):
    reg_list = targets_regs[tgt_id]
    n_regs   = len(reg_list)
    magnitudes  = _sample_from_dist(k_dist, n_regs, rng)
    coop_states = _sample_from_dist(hill_coeff_dist, n_regs, rng)

    reg_profiles = [mr_profile[reg_id] for reg_id, _ in reg_list]

    votes: dict[int, float] = {}
    for dom_mr, strength in reg_profiles:
      votes[dom_mr] = votes.get(dom_mr, 0.0) + abs(strength)
    winner_mr = max(votes, key=lambda mr: votes[mr] - balancing_strength * mr_load[mr])

    if diagnostics is not None:
      sorted_votes = sorted(votes.values(), reverse=True)
      vote_margin = (sorted_votes[0] - sorted_votes[1]) if len(sorted_votes) >= 2 else float('nan')
      n_ambiguous = sum(1 for _, sign in reg_list if sign not in ('activation', 'repression'))
      n_ambiguous_flipped = 0

    reg_ids  = []
    k_values = []
    edge_signs: list[int] = []
    for (reg_id, sign), mag, (dom_mr, dom_strength) in zip(reg_list, magnitudes, reg_profiles):
      if sign == 'activation':
        is_repressor = False
      elif sign == 'repression':
        is_repressor = True
      else:
        p_repressor = unknown_mode_repressor_prob
        if coherency_bias > 0.0 and dom_mr == winner_mr:
          # bias toward whatever sign makes this edge AGREE with the
          # regulator's own net effect on the winning MR (dom_strength's
          # sign): a repressor edge flips the propagated sign, so a
          # negative dom_strength (regulator is net-repressed by its own
          # winning MR) wants a repressor edge here to end up net-positive
          # (two sign flips), and vice versa.
          want_repressor = dom_strength < 0.0
          p_repressor = (1.0 - coherency_bias) * unknown_mode_repressor_prob \
                      + coherency_bias * float(want_repressor)
        draw = rng.random()
        is_repressor = draw < p_repressor
        if diagnostics is not None and (draw < p_repressor) != (draw < unknown_mode_repressor_prob):
          # counts this draw as "flipped" only if coherency_bias's altered
          # p_repressor actually changed the outcome relative to what the
          # plain unbiased unknown_mode_repressor_prob coin flip would have
          # given for this SAME draw -- always 0 when coherency_bias == 0.0
          # (p_repressor == unknown_mode_repressor_prob exactly, so the two
          # comparisons can never disagree).
          n_ambiguous_flipped += 1
      edge_sign = -1 if is_repressor else 1
      edge_signs.append(edge_sign)
      reg_ids.append(reg_id)

      weight = 1.0
      if canalization_strength > 0.0:
        aligned = (dom_mr == winner_mr) and (np.sign(dom_strength) == edge_sign or dom_strength == 0.0)
        weight = max(1.0 + canalization_strength, 0.05) if aligned else max(1.0 - canalization_strength, 0.05)
      k_values.append(edge_sign * abs(float(mag)) * weight)

    # Renormalize this target's K-vector so its total production-rate
    # "budget" (sum of |K_i|, i.e. SERGIO's calculate_prod_rate_ ceiling:
    # rate = sum_i |K_i| * hill_i(...), each hill_i in [0, 1]) matches what
    # it would have been WITHOUT canalization's per-edge reweighting above.
    # Without this, canalization_strength systematically shrinks a target's
    # total achievable production rate whenever it has any "disaligned"
    # edges (weight as low as 0.05x, uncompensated by the aligned edges'
    # weight, which only goes up to 2x) -- a real, mechanical link from
    # "more canalization/balancing -> more PC2+ structure" to "lower
    # gene_mean/gene_var/library size", since SERGIO's lib_size_effect then
    # renormalizes each CELL's total to a fixed drawn target, forcing every
    # other gene to compete for whatever budget this shrinkage left behind.
    # Preserving sum(|K_i|) here removes that particular coupling: what
    # canalization does with the (now fixed) budget is purely reallocate
    # it toward whichever regulator tracks the winning master regulator,
    # which is what should create between-cluster (PCA-relevant) variance,
    # rather than shrinking the target's overall expression level.
    # Exactly a no-op (scale == 1.0) whenever canalization_strength == 0.0,
    # since weight == 1.0 for every edge then (weighted_abs_sum == raw_sum
    # exactly) -- byte-identical to omitting this step entirely, same as
    # every other canalization_strength/coherency_bias/balancing_strength/
    # path_decay knob's own "0.0 reproduces prior behavior" guarantee.
    raw_sum          = sum(abs(float(mag)) for mag in magnitudes)
    weighted_abs_sum = sum(abs(k) for k in k_values)
    if weighted_abs_sum > 0.0:
      budget_scale = raw_sum / weighted_abs_sum
      k_values = [k * budget_scale for k in k_values]

    aligned_vals = [dom_strength * es for (dom_mr, dom_strength), es in zip(reg_profiles, edge_signs)
                     if dom_mr == winner_mr]
    strength = path_decay * (sum(aligned_vals) / len(aligned_vals)) if aligned_vals else 0.0
    mr_profile[tgt_id] = (winner_mr, strength)
    mr_load[winner_mr] = mr_load.get(winner_mr, 0.0) + abs(strength)

    if diagnostics is not None:
      diag_tgt_ids.append(tgt_id)
      diag_winner_mr.append(winner_mr)
      diag_vote_margin.append(vote_margin)
      diag_n_regs.append(n_regs)
      diag_n_ambiguous.append(n_ambiguous)
      diag_n_ambiguous_flipped.append(n_ambiguous_flipped)
      diag_n_aligned_edges.append(len(aligned_vals))
      diag_propagated_strength.append(strength)

    rows.append([tgt_id, n_regs] + reg_ids + k_values + [float(c) for c in coop_states])

  with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
      writer.writerow(row)

  if diagnostics is not None:
    n_targets = len(diag_tgt_ids)
    winner_counts: dict[int, int] = {}
    for mr in diag_winner_mr:
      winner_counts[mr] = winner_counts.get(mr, 0) + 1
    diagnostics.update({
        "tgt_ids":              diag_tgt_ids,
        "winner_mr":            diag_winner_mr,
        "vote_margin":          diag_vote_margin,
        "n_regs":               diag_n_regs,
        "n_ambiguous":          diag_n_ambiguous,
        "n_ambiguous_flipped":  diag_n_ambiguous_flipped,
        "n_aligned_edges":      diag_n_aligned_edges,
        "propagated_strength":  diag_propagated_strength,
        "mr_load":              dict(mr_load),
        "n_distinct_winners":   len(winner_counts),
        "top1_winner_share":    (max(winner_counts.values()) / n_targets) if n_targets else float('nan'),
    })

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
# MR overlap tree + tree-correlated mr_state sampling.
# --------------------------------------------------------------------------
# _sample_mr_state (tune_synthetic_data.py) draws every (cluster, MR) entry
# i.i.d. from Uniform(mr_rate_low, mr_rate_high) -- no MR ever gets a rate
# correlated with any other MR's, regardless of how much regulatory overlap
# they share. The functions below build a hierarchical clustering of a
# GRN's master regulators by the overlap of the gene sets they directly
# regulate (build_mr_overlap_tree), then sample MR states whose cross-MR
# *correlation* (not marginal distribution) is driven by that tree
# (sample_mr_state_from_tree): MRs that regulate highly overlapping gene
# sets get correlated basal rates across states, so their (also overlapping)
# target genes tend to respond coherently -- inducing exactly the kind of
# co-regulated-gene correlation structure i.i.d. mr_state sampling cannot.
#
# Typical usage (see also run_mr_tree_experiment below for a full paired
# tree-vs-i.i.d. comparison harness):
#   grn_path, mr_ids, _ = generate_sergio_grn_from_reference(..., seed=42)
#   tree     = build_mr_overlap_tree(grn_path, mr_ids)
#   mr_state = sample_mr_state_from_tree(
#       tree, n_states=n_clusters, low=1.0, high=5.0, seed=0, tree_strength=1.0)
#   X, cluster_labels = make_synthetic_data6(
#       mr_state=torch.tensor(mr_state, dtype=torch.float32),
#       input_file_targets=grn_path, mr_gene_ids=mr_ids, ...)

@dataclass(frozen=True)
class MRTree:
  """
  A rooted tree over master-regulator (MR) gene ids, as built by
  build_mr_overlap_tree() (or any other tree-construction strategy sharing
  this same interface) and consumed by sample_mr_state_from_tree() /
  mr_tree_pairwise_covariance(). Deliberately minimal so the sampler is
  independent of *how* the tree was built (Jaccard overlap + average
  linkage, or any future alternative strategy).

  mr_ids:       ordered list of leaf master-regulator gene ids, len n_mrs.
                Leaf node ids 0..n_mrs-1 correspond index-for-index to this
                list (mr_ids[i] is the gene id of leaf node i) --
                sample_mr_state_from_tree's output columns follow this
                same order.
  children:     dict mapping each internal node id (>= n_mrs) to a
                (left_child_id, right_child_id) 2-tuple of node ids (leaf
                ids in [0, n_mrs) and/or other internal node ids). Leaf
                node ids are never keys of this dict.
  edge_length:  dict mapping every non-root node id (leaf or internal) to
                the non-negative length of the edge connecting it to its
                parent. The root has no entry (it has no parent).
  root:         node id of the root. If n_mrs == 1, root == 0 (the single
                leaf) and both children/edge_length are empty.
  """
  mr_ids:      list
  children:    dict
  edge_length: dict
  root:        int

  @property
  def n_mrs(self) -> int:
    return len(self.mr_ids)


def build_mr_target_sets(input_file_targets: str, mr_ids: list) -> dict:
  """
  Parse a SERGIO-format `input_file_targets` CSV (as written by
  generate_sergio_grn_from_reference / consumed by make_synthetic_data6,
  see _parse_sergio_targets_file's row-format docstring) and return, for
  each gene id in `mr_ids`, the set of target gene ids it *directly*
  regulates in that file's final, already-DAG-ified graph (one hop --
  not the transitive closure of its downstream regulatory cascade).

  This deliberately reads the *generated* GRN file rather than the
  original (possibly cyclic) reference network passed to generate_sergio_
  grn_from_reference: the generated file is the topology make_synthetic_
  data6 actually simulates over (post subgraph-sampling, DAG-ification,
  and isolated-node repair), so it is what build_mr_overlap_tree's
  clustering should reflect.

  Args:
    input_file_targets: path to a SERGIO-format targets CSV (row format:
                  target_id, n_regs, reg_id_1..reg_id_n, K_1..K_n[,
                  coop_1..coop_n]).
    mr_ids:       gene ids to build target sets for (typically every
                  master regulator in the file, e.g. generate_sergio_grn_
                  from_reference's own returned mr_ids -- but any subset
                  of regulator ids present in the file also works).

  Returns:
    dict mapping each id in `mr_ids` to a frozenset of target gene ids it
    directly regulates. An id with no surviving out-edges maps to an
    empty frozenset rather than raising -- should not happen for a true
    master regulator in a well-formed generated GRN (see generate_sergio_
    grn_from_reference's isolated-node-repair docstring note: every gene
    ends up with at least one edge, and an MR's zero in-degree forces any
    such edge to be outgoing), but tolerated here for an arbitrary
    caller-supplied id.
  """
  mr_id_set = set(mr_ids)
  target_sets = {mr_id: set() for mr_id in mr_id_set}
  with open(input_file_targets, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      row = [c.strip() for c in row if c.strip() != '']
      if not row:
        continue
      target_id = int(float(row[0]))
      n_regs    = int(float(row[1]))
      for r in row[2 : 2 + n_regs]:
        reg_id = int(float(r))
        if reg_id in mr_id_set:
          target_sets[reg_id].add(target_id)
  return {mr_id: frozenset(s) for mr_id, s in target_sets.items()}


def _jaccard_distance_matrix(target_sets: list) -> np.ndarray:
  """
  Pairwise Jaccard *distance* matrix (1 - |A∩B| / |A∪B|) over a list of
  gene-id sets (one per MR, in the caller's mr_ids order). Two MRs with
  identical nonempty target sets get distance 0.0; two MRs where the
  union is empty (i.e. at least one has an empty target set, since
  self-unions of an empty set are empty) get distance 1.0 by convention
  -- |A∩B|/|A∪B| is 0/0 there, treated as "maximally dissimilar" (an
  empty target set carries no overlap information to cluster on) rather
  than raising. Diagonal is exactly 0.0.
  """
  n = len(target_sets)
  dist = np.zeros((n, n), dtype=np.float64)
  for i in range(n):
    si = target_sets[i]
    for j in range(i + 1, n):
      sj = target_sets[j]
      union = len(si | sj)
      d = 1.0 if union == 0 else 1.0 - len(si & sj) / union
      dist[i, j] = dist[j, i] = d
  return dist


def _average_linkage_tree(dist: np.ndarray) -> tuple:
  """
  Deterministic UPGMA (average-linkage) agglomerative clustering over a
  precomputed (n, n) symmetric distance matrix. Leaves are node ids
  0..n-1 (row/column index into `dist`); each merge creates a new
  internal node id n, n+1, .... Self-contained (no scipy dependency) and
  deterministic -- ties are broken by lowest (a, b) active-node-id pair,
  which is itself fully determined by `dist` and the merge history.

  Returns (children, height, root):
    children: dict[internal_node_id] = (a, b), the two node ids merged to
              create it (a < b is NOT guaranteed -- only that they're the
              two chosen active clusters at that step).
    height:   dict[node_id] -> float, 0.0 for every leaf (0..n-1) and the
              UPGMA merge distance for every internal node -- always
              non-decreasing along any root-ward path (a valid
              dendrogram, i.e. safe to derive edge lengths from via
              height(parent) - height(child)).
    root:     the final (top-level) node id (2 * n - 2 if n >= 2, else 0).
  """
  n = dist.shape[0]
  assert n >= 1, "need at least one leaf"
  if n == 1:
    return {}, {0: 0.0}, 0

  size:   dict = {i: 1   for i in range(n)}
  height: dict = {i: 0.0 for i in range(n)}
  children: dict = {}
  # D[a][b]: current inter-cluster distance between active clusters a, b.
  D: dict = {i: {j: float(dist[i, j]) for j in range(n) if j != i} for i in range(n)}
  active   = list(range(n))
  next_id  = n

  while len(active) > 1:
    best_key = None
    best_pair = None
    for ai in range(len(active)):
      a = active[ai]
      for b in active[ai + 1:]:
        key = (D[a][b], a, b)
        if best_key is None or key < best_key:
          best_key, best_pair = key, (a, b)
    d_ab, a, b = best_key[0], best_pair[0], best_pair[1]

    new_id = next_id
    next_id += 1
    children[new_id] = (a, b)
    height[new_id]   = d_ab
    size[new_id]     = size[a] + size[b]

    merged = {}
    for c in active:
      if c == a or c == b:
        continue
      merged[c] = (size[a] * D[a][c] + size[b] * D[b][c]) / (size[a] + size[b])
    del D[a], D[b]
    for c in D:
      D[c].pop(a, None)
      D[c].pop(b, None)
    D[new_id] = merged
    for c, d in merged.items():
      D[c][new_id] = d

    active = [c for c in active if c != a and c != b] + [new_id]

  return children, height, active[0]


def build_mr_overlap_tree(
    input_file_targets: str,
    mr_ids:             list,
    target_sets:        dict | None = None,
) -> MRTree:
  """
  Build an MRTree clustering `mr_ids` by the overlap of the gene sets
  they directly regulate in `input_file_targets` (see
  build_mr_target_sets): MRs regulating highly overlapping gene sets
  merge low (near the leaves) and end up sharing a long portion of their
  root-to-leaf path; MRs regulating disjoint gene sets only merge near
  the root and share almost none of it.

  Distance: unweighted Jaccard distance (1 - |A∩B|/|A∪B|) between each
  pair of MRs' direct target-gene sets (see _jaccard_distance_matrix).
  Linkage: average linkage (UPGMA, see _average_linkage_tree) -- a
  moderate choice between single linkage (prone to chaining through weak
  intermediate overlaps) and complete linkage (overly sensitive to a
  single worst-case pair); Ward linkage is not used since Jaccard
  distance is not a Euclidean feature-space metric, which Ward's
  variance-minimization criterion assumes.

  Edge lengths: edge_length(child) = height(parent) - height(child),
  where `height` is each node's UPGMA merge distance (0.0 for leaves) --
  i.e. branch lengths are exactly the increments accumulated along the
  dendrogram from leaves to root, so the length of the path segment any
  two leaves *share* (root down to their lowest common ancestor) equals
  (root height - their LCA's height): MRs merging early (low LCA height,
  high overlap) share a *longer* portion of the path from the root, and
  MRs only merging near the root (low overlap) share almost none of it
  -- see sample_mr_state_from_tree's docstring for how this drives each
  MR pair's sampled correlation.

  Args:
    input_file_targets: path to the generated SERGIO-format targets CSV
                  (e.g. generate_sergio_grn_from_reference's output_path)
                  whose *final, DAG-ified* topology should be clustered.
    mr_ids:       master-regulator gene ids to cluster (e.g. generate_
                  sergio_grn_from_reference's own returned mr_ids) --
                  become the tree's leaves, in this order (the returned
                  tree.mr_ids == list(mr_ids); duplicates are rejected).
    target_sets:  optional precomputed build_mr_target_sets(...) result
                  (must have an entry for every id in mr_ids) -- avoids
                  re-parsing input_file_targets if the caller already
                  has it (e.g. run_mr_tree_experiment). Computed
                  internally via build_mr_target_sets if not given.

  Returns:
    MRTree with mr_ids in the same order as the `mr_ids` argument (leaf
    node i's gene id is mr_ids[i], regardless of merge order).
  """
  assert len(mr_ids) == len(set(mr_ids)), f"duplicate mr_ids: {mr_ids}"
  mr_ids = list(mr_ids)
  n = len(mr_ids)
  assert n >= 1, "build_mr_overlap_tree requires at least one MR"

  if target_sets is None:
    target_sets = build_mr_target_sets(input_file_targets, mr_ids)
  dist = _jaccard_distance_matrix([target_sets[mr_id] for mr_id in mr_ids])
  children, height, root = _average_linkage_tree(dist)

  edge_length = {}
  for node, (left, right) in children.items():
    edge_length[left]  = height[node] - height[left]
    edge_length[right] = height[node] - height[right]

  return MRTree(mr_ids=mr_ids, children=children, edge_length=edge_length, root=root)


def mr_tree_pairwise_covariance(tree: MRTree, root_variance: float = 0.0) -> np.ndarray:
  """
  (n_mrs, n_mrs) covariance matrix K implied by `tree`'s branch lengths
  under the tree-Brownian-motion model sample_mr_state_from_tree is based
  on: each edge e contributes an independent latent increment of variance
  edge_length[e], and (if root_variance > 0) the root itself contributes
  one more shared increment of variance root_variance common to every
  leaf. Two leaves' covariance is then exactly the total length of the
  path segment they share (root down to their lowest common ancestor),
  plus root_variance:

      K[i, j] = root_variance + sum(edge_length[e] for e on the shared
                                     root-to-LCA(i, j) path)

  with K[i, i] = root_variance + (total root-to-leaf-i path length) as
  the i == j case (LCA(i, i) = i). This is a valid (symmetric positive
  semidefinite) covariance matrix by construction -- it is the Gram
  matrix of the actual random increments in the model above -- provided
  here purely for diagnostics/testing (e.g. converting to a correlation
  matrix to check that MRs with more target-gene overlap end up more
  correlated, or verifying PSD-ness). sample_mr_state_from_tree never
  materializes this matrix itself; see its own docstring for the
  equivalent (and exactly distributionally equivalent) direct-simulation
  approach it actually uses.

  Ordering matches tree.mr_ids (row/column i corresponds to tree.mr_ids[i]).
  """
  assert root_variance >= 0.0, "root_variance must be >= 0"
  n = tree.n_mrs

  # cumulative root-to-node path length, computed top-down from the root.
  cum_length = {tree.root: root_variance}
  stack = [tree.root]
  while stack:
    node = stack.pop()
    children = tree.children.get(node)
    if not children:
      continue
    for child in children:
      cum_length[child] = cum_length[node] + tree.edge_length[child]
      stack.append(child)

  parent = {}
  for node, (left, right) in tree.children.items():
    parent[left]  = node
    parent[right] = node

  def _ancestors(leaf: int) -> list:
    chain = [leaf]
    while chain[-1] != tree.root:
      chain.append(parent[chain[-1]])
    return chain

  ancestor_chains = [_ancestors(i) for i in range(n)]

  K = np.empty((n, n), dtype=np.float64)
  for i in range(n):
    depth_i = {node: k for k, node in enumerate(ancestor_chains[i])}
    for j in range(i, n):
      if i == j:
        lca = i
      else:
        lca = next((node for node in ancestor_chains[j] if node in depth_i), None)
        assert lca is not None, "no common ancestor found -- malformed tree"
      K[i, j] = K[j, i] = cum_length[lca]
  return K


def mr_tree_correlation_matrix(tree: MRTree, root_variance: float = 0.0) -> np.ndarray:
  """
  Correlation-matrix normalization of mr_tree_pairwise_covariance's K
  (R[i, j] = K[i, j] / sqrt(K[i, i] * K[j, j])) -- this is the R_tree
  sample_mr_state_from_tree's docstring refers to. Provided as a
  standalone diagnostic (e.g. to confirm empirically that MRs with more
  target-gene overlap end up with a higher implied correlation) --
  sample_mr_state_from_tree does not use this function internally (see
  its own docstring for why). Entries involving a leaf with K[i, i] == 0
  (a degenerate zero-length root-to-leaf path, only possible if
  root_variance == 0.0 and that leaf's entire path to the root has zero
  total edge length) are NaN.
  """
  K = mr_tree_pairwise_covariance(tree, root_variance=root_variance)
  d = np.sqrt(np.diag(K))
  with np.errstate(invalid='ignore', divide='ignore'):
    R = K / np.outer(d, d)
  return R


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
  """Standard normal CDF, vectorized via math.erf -- avoids adding a hard
  scipy.stats/scipy.special dependency for this one elementwise
  transform (Phi(z) = 0.5 * (1 + erf(z / sqrt(2))))."""
  erf_vec = np.vectorize(math.erf, otypes=[np.float64])
  return 0.5 * (1.0 + erf_vec(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


def sample_mr_state_from_tree(
    tree:          MRTree,
    n_states:      int,
    low:           float,
    high:          float,
    seed:          int | None = None,
    tree_strength: float      = 1.0,
    root_variance: float      = 0.0,
) -> np.ndarray:
  """
  Sample an (n_states, n_mrs) matrix of master-regulator basal production
  rates whose *marginal* distribution is (via a Gaussian copula, see
  below) similar to _sample_mr_state's plain i.i.d. Uniform(low, high),
  but whose *cross-MR correlation* is driven by `tree`: MRs that merge
  low in the tree (e.g. high target-gene overlap, if `tree` came from
  build_mr_overlap_tree) get correlated rates across states; MRs that
  only share the root get ~independent rates. Column i of the output
  corresponds to tree.mr_ids[i] -- pass tree.mr_ids as make_synthetic_
  data6's mr_gene_ids to keep them aligned.

  Model: each edge e of `tree` contributes one independent latent
  Gaussian increment per state, N(0, edge_length[e]); each leaf's raw
  tree-latent value is the sum of the increments along its root-to-leaf
  path (plus one N(0, root_variance) term shared by every leaf, if
  root_variance > 0) -- i.e. Brownian motion along the tree's branches,
  the standard phylogenetic-comparative-methods model for trait
  covariance induced by shared ancestry. This makes Cov(leaf_i, leaf_j)
  exactly mr_tree_pairwise_covariance's K[i, j] (root_variance plus the
  length of the root-to-LCA path segment leaf_i and leaf_j share). This
  function never builds or factorizes that (n_mrs, n_mrs) covariance
  matrix: it draws the underlying per-edge increments directly and sums
  them along each leaf's path, which is exactly distributionally
  equivalent, cheaper (O(n_mrs) edges instead of an (n_mrs, n_mrs) matrix
  + Cholesky/eigendecomposition), and has no possible positive-
  semidefiniteness numerical edge cases.

  Each leaf's raw tree-latent value is rescaled to unit variance (using
  its theoretical variance, root_variance + its total root-to-leaf edge
  length -- not a finite-n_states sample estimate) and blended with an
  independent standard-normal draw:

      z_i = sqrt(tree_strength) * (tree_latent_i / std_i)
          + sqrt(1 - tree_strength) * independent_i

  so z_i always has unit variance and Corr(z_i, z_j) = tree_strength *
  R_tree[i, j] for i != j (R_tree = mr_tree_correlation_matrix(tree,
  root_variance)), 1.0 for i == j -- i.e. `tree_strength` interpolates
  linearly, in *correlation*, between fully tree-structured
  (tree_strength=1.0) and fully independent (tree_strength=0.0) rates,
  without ever changing each individual MR's own marginal distribution.
  A leaf whose total root-to-leaf path length is exactly 0.0 (only
  possible if root_variance == 0.0 too -- i.e. that leaf shares no branch
  length with anything, a degenerate tree essentially never produced by
  build_mr_overlap_tree for >1 MR with any genuine target-set
  differences) has no tree signal to contribute; such a leaf's column
  always uses a plain independent standard normal regardless of
  tree_strength, so its marginal distribution is never distorted.

  Each z_i is then mapped through the standard normal CDF (a Gaussian
  copula) and rescaled into [low, high]:

      mr_state[:, i] = low + (high - low) * Phi(z_i)

  At tree_strength=0.0, every z_i is a plain independent standard normal
  (the tree-latent term contributes nothing), so mr_state is a matrix of
  independent Uniform(low, high) columns via the same Gaussian-copula
  transform used at any other tree_strength -- similar in distribution
  to (but not a byte-identical reproduction of) _sample_mr_state's direct
  rng.uniform(...) draw, by design: this lets a tree_strength=0.0 vs.
  >0.0 comparison isolate the tree's covariance structure as the only
  variable, without also changing the underlying univariate sampling
  method itself.

  Args:
    tree:          an MRTree (e.g. from build_mr_overlap_tree()).
    n_states:      number of rows (e.g. make_synthetic_data6's n_clusters)
                   to sample.
    low, high:     output range (matches _sample_mr_state's low/high --
                   typically config["mr_rate_low"]/["mr_rate_high"]).
    seed:          seeds a local np.random.default_rng (same convention
                   as _sample_mr_state) -- deterministic given the same
                   tree/n_states/seed/tree_strength/root_variance.
    tree_strength: in [0, 1]. 0.0 = independent columns (see above); 1.0
                   = purely tree-structured (no independent component).
    root_variance: variance of a single latent factor shared by every
                   leaf regardless of tree structure (e.g. representing
                   unmodeled shared regulatory context). 0.0 (default)
                   means every leaf's tree-latent value is driven purely
                   by edges on its own root-to-leaf path, with no
                   universal shared component.

  Returns:
    (n_states, tree.n_mrs) numpy array, values in [low, high].
  """
  assert 0.0 <= tree_strength <= 1.0, f"tree_strength must be in [0, 1], got {tree_strength}"
  assert root_variance >= 0.0, "root_variance must be >= 0"
  assert high > low, f"high ({high}) must be > low ({low})"
  n = tree.n_mrs
  rng = np.random.default_rng(seed)

  # --- top-down simulation of the tree-latent Brownian values, plus the
  # theoretical (not sampled) cumulative edge-length-to-root of each node,
  # used below to rescale each leaf to unit variance. ---
  cum_length = {tree.root: root_variance}
  node_value = {tree.root: rng.normal(0.0, math.sqrt(root_variance), size=n_states)}
  stack = [tree.root]
  while stack:
    node = stack.pop()
    children = tree.children.get(node)
    if not children:
      continue
    for child in children:
      length = tree.edge_length[child]
      cum_length[child] = cum_length[node] + length
      std = math.sqrt(length) if length > 0.0 else 0.0
      increment = rng.normal(0.0, std, size=n_states) if std > 0.0 else np.zeros(n_states)
      node_value[child] = node_value[node] + increment
      stack.append(child)

  tree_latent = np.stack([node_value[i] for i in range(n)], axis=1)          # (n_states, n)
  leaf_var    = np.array([cum_length[i] for i in range(n)], dtype=np.float64)  # (n,)

  independent = rng.normal(0.0, 1.0, size=(n_states, n))
  degenerate  = leaf_var <= 0.0

  z = np.empty((n_states, n), dtype=np.float64)
  if (~degenerate).any():
    tree_latent_unit = tree_latent[:, ~degenerate] / np.sqrt(leaf_var[~degenerate])
    z[:, ~degenerate] = (
        math.sqrt(tree_strength) * tree_latent_unit
        + math.sqrt(1.0 - tree_strength) * independent[:, ~degenerate]
    )
  if degenerate.any():
    z[:, degenerate] = independent[:, degenerate]

  u = _standard_normal_cdf(z)
  return low + (high - low) * u


# --------------------------------------------------------------------------
# Real-data reference loading & summary statistics, for calibrating
# make_synthetic_data6's SERGIO-derived parameters against a real scRNA-seq
# reference matrix.
# --------------------------------------------------------------------------
#
# Typical usage:
#   X_ref, _     = load_reference_h5ad("ref.h5ad")
#   target_stats = compute_summary_stats(X_ref)
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

# Thresholds shared by _check_value_scale's two directions ("raw_counts" in
# load_reference_h5ad, "log1p_counts" in compute_summary_stats) -- see its
# docstring for why these two particular signals (and their conjunction)
# were chosen.
_SCALE_CHECK_MAX_INTEGER_VALUE = 20.0
_SCALE_CHECK_MIN_INTEGER_FRAC  = 0.9
_SCALE_CHECK_SAMPLE_BUDGET     = 20_000


def _check_value_scale(sample: np.ndarray, name: str, expect: str) -> None:
  """
  Smoke-test guarding against silently mixing up raw-count-scale and
  log1p-scale data at either boundary of the make_synthetic_data6 <->
  load_reference_h5ad <-> compute_summary_stats pipeline: either treating
  already-normalized/log-transformed data as if it were raw counts (about
  to be log1p'd a second time), or treating not-yet-log1p'd raw counts as
  if they were already on the log1p scale compute_summary_stats expects.
  Both would silently corrupt every downstream summary statistic without
  erroring anywhere else. `sample` should be a 1D array of *nonzero*
  values drawn from the candidate matrix, on whichever scale `expect`
  names (i.e. pre-log1p for "raw_counts", post-log1p for "log1p_counts").

  Args:
    expect: "raw_counts" (used by load_reference_h5ad, checking the
            pre-log1p source about to be transformed) or "log1p_counts"
            (used by compute_summary_stats, checking its own input).

  Two signals, checked jointly rather than independently, using the same
  two thresholds in opposite directions:
    1. any negative values -> never valid for either scale (both raw
       counts and log1p(raw counts) are always non-negative) -- rules out
       z-scored/scaled/PCA-space data.
    2. "raw_counts": mostly non-integer-valued AND a small dynamic range
       (max < 20) -> the classic log1p(raw-counts) signature: log1p
       compresses realistic scRNA-seq counts into roughly [0, ~10], and
       the result is essentially never exactly integer-valued. Flagged as
       an error since expect="raw_counts" data should NOT look like this.
       "log1p_counts": the mirror image -- mostly integer-valued AND a
       *large* dynamic range (max >= 20) -> looks like raw counts that
       were never log1p'd. Flagged as an error since expect="log1p_counts"
       data should NOT look like this.
  Signal 2 is deliberately a *conjunction*, not "non-integer"/"integer"
  alone: the CELLxGENE schema explicitly permits non-UMI (e.g. Smart-
  seq2/RSEM) "raw" matrices to contain fractional *estimated* read/
  fragment counts, which are non-integer but still span a wide dynamic
  range (housekeeping genes reach into the hundreds/thousands) -- so they
  correctly pass the "raw_counts" check (non-integer alone isn't enough to
  fail it), while genuinely log-transformed data (non-integer AND
  compressed) fails it.
  """
  assert expect in ("raw_counts", "log1p_counts"), f"unknown expect {expect!r}"
  if sample.size == 0:
    return
  frac_negative = float(np.mean(sample < 0))
  if frac_negative > 0:
    raise ValueError(
        f"{name}: {frac_negative:.1%} of sampled values are negative -- "
        "this does not look like raw counts or log1p(raw counts) (both "
        "are always non-negative). It looks like already-scaled/centered "
        "data (e.g. post scanpy.pp.scale or a PCA embedding)."
    )
  frac_integer = float(np.mean(np.abs(sample - np.round(sample)) <= 1e-4))
  max_value    = float(np.max(sample))
  if expect == "raw_counts":
    looks_wrong = (frac_integer < _SCALE_CHECK_MIN_INTEGER_FRAC
                   and max_value < _SCALE_CHECK_MAX_INTEGER_VALUE)
    detail = (
        f"only {frac_integer:.1%} of sampled nonzero values are "
        f"integer-valued and the sampled max is {max_value:.3g} (< "
        f"{_SCALE_CHECK_MAX_INTEGER_VALUE:g}) -- this looks like "
        "already-normalized/log-transformed data, not raw UMI/read counts "
        "(log1p-transformed scRNA-seq data is typically non-integer and "
        "compressed into roughly this range). Pass a different `source` "
        "(e.g. source='raw' if you used source='X', or vice versa)."
    )
  else:
    looks_wrong = (frac_integer >= _SCALE_CHECK_MIN_INTEGER_FRAC
                   and max_value >= _SCALE_CHECK_MAX_INTEGER_VALUE)
    detail = (
        f"{frac_integer:.1%} of sampled nonzero values are integer-valued "
        f"and the sampled max is {max_value:.3g} (>= "
        f"{_SCALE_CHECK_MAX_INTEGER_VALUE:g}) -- this looks like raw "
        "counts that were never log1p-transformed, not log1p-scale data. "
        "compute_summary_stats expects log1p(raw counts), matching "
        "make_synthetic_data6's and load_reference_h5ad's output "
        "convention. Pass validate_scale=False if you're confident this "
        "is actually log1p-scale data and are hitting a false positive."
    )
  if looks_wrong:
    raise ValueError(f"{name}: {detail}")


def load_reference_h5ad(
    path:                str,
    source:              str      = "raw",
    layer:               str|None = None,
    n_top_genes:         int|None = None,
    select_gene_symbols: list[str]|None = None,
    feature_name_col:    str|None = "feature_name",
    n_cells_subsample:   int|None = 50_000,
    chunk_size:          int      = 5_000,
    seed:                int|None = 0,
    validate_raw_counts: bool     = True,
    device:              str|None = None,
) -> tuple[torch.Tensor, list[str]|None]:
  """
  Load a real scRNA-seq reference matrix from an .h5ad file and bring it to
  the same (n_cells, n_genes) float32, log1p-transformed tensor format
  produced by make_synthetic_data6 (minus its MCAR NaN mask -- a real
  reference has no such artificial missingness; see the module-level note
  above on how to compare fairly against make_synthetic_data6 output).

  Per the CZ CELLxGENE Discover data schema (and common scanpy/AnnData
  practice generally), a curated .h5ad frequently stores *normalized*,
  log-transformed values in `adata.X` (e.g. because that's what was used
  to compute a stored embedding) and the *raw* UMI/read counts separately
  in `adata.raw.X` -- raw counts are only expected directly in `adata.X`
  when no normalized layer was provided at all. Since this function needs
  raw counts (it applies its own log1p, matching make_synthetic_data6's
  convention), it defaults to `adata.raw.X` (see `source` below) rather
  than `adata.X`, and independently sanity-checks the selected source's
  scale (see `validate_raw_counts` below) to catch this class of mismatch
  even for non-CxG files that don't follow the same convention.

  Reads the file in AnnData's disk-backed mode and streams over row-chunks
  rather than densifying the full (n_cells, n_genes) matrix, since a
  real reference's sparse on-disk footprint can be many times smaller than
  its dense in-memory footprint (e.g. a modest multi-GB sparse .h5ad can
  balloon to 100+ GB dense). Two sequential passes are made over the data,
  each reading every chunk exactly once (no random-access seeks):
    1. an exact per-gene total-count pass, over *all* cells, using
       sparse-native `.sum(axis=0)` per chunk (never densifies) -- used to
       drop all-zero genes and pick `n_top_genes`, both computed exactly
       over the whole dataset; also collects a bounded sample of nonzero
       values (from a handful of chunks spread across the dataset) for the
       validate_raw_counts scale check, at no extra I/O cost;
    2. a row-subsampling pass that keeps only a random `n_cells_subsample`
       cells (exact count, chosen upfront), densifying only the selected
       rows x selected genes as each chunk streams past.
  The returned matrix is therefore an exact-genes / subsampled-cells view
  of the reference -- appropriate for compute_summary_stats, which
  describes aggregate distributions (mean/variance/library-size/dropout/
  PCA/correlation structure) that are stable under cell subsampling at
  this scale, rather than exact per-cell values.

  Args:
    path:         path to an .h5ad file (AnnData format).
    source:       which slot holds the raw counts to load -- one of:
                    "raw":   adata.raw.X (default). Raises ValueError if
                             adata.raw is None -- deliberately does NOT
                             silently fall back to adata.X, since silently
                             guessing the wrong slot is exactly the failure
                             mode this function guards against. Also
                             asserts adata.raw.n_obs == adata.n_obs (the
                             schema-required cell alignment between the
                             two).
                    "X":     adata.X directly.
                    "layer": adata.layers[layer] (layer must be given).
                  Note: only adata.X is guaranteed to stay disk-backed
                  (streamed) in every anndata version -- some versions
                  eagerly load adata.raw / non-default layers into memory
                  even under backed=True. The chunked extraction below is
                  correct either way, just without the memory savings in
                  that case.
    layer:        the layer name to read from when source="layer".
    n_top_genes:  if given, keep only the `n_top_genes` genes with the
                  highest mean raw-count expression (a simple,
                  dependency-free stand-in for highly-variable-gene
                  selection), computed exactly over all cells in pass 1
                  above (not subsampled). Useful for keeping the reference
                  at a comparable gene-count scale to a small SERGIO GRN,
                  and for bounding the memory of pass 2's densified
                  blocks. Gene *identity*/order is not otherwise aligned
                  with any synthetic GRN -- compute_summary_stats compares
                  aggregate distributions, not per-gene identity. Mutually
                  exclusive with `select_gene_symbols` (which aligns by
                  identity instead).
    select_gene_symbols: if given, keep only genes whose `feature_name_col`
                  value is in this list (e.g. the exact gene symbols used
                  by a `generate_sergio_grn_from_reference`-built synthetic
                  GRN, so the reference and synthetic sides describe the
                  *same* genes rather than two differently-selected panels
                  -- see that function's `gene_id_to_symbol` return value).
                  Still subject to the same "drop all-zero-total-count
                  genes" filter as the default path, and to
                  `n_cells_subsample`'s gene-count-independent row
                  subsampling. Symbols not found in `feature_name_col` (or
                  found but all-zero) are silently dropped and reported via
                  a warnings.warn() (not an error) -- check the returned
                  `kept_gene_symbols` against your requested list if you
                  need an exact accounting. If a requested symbol matches
                  more than one row in `feature_name_col` (e.g. distinct
                  Ensembl IDs sharing a symbol), all matching rows are kept
                  as separate columns and a second warnings.warn() reports
                  which symbol(s) are duplicated -- `kept_gene_symbols` will
                  then contain repeated entries and be longer than the
                  number of requested symbols; deduplicate the result
                  yourself if you need exactly one column per symbol.
                  Mutually exclusive with `n_top_genes`; requires
                  `feature_name_col` to be set.
    feature_name_col: name of the `adata.var` (or, when source="raw",
                  `adata.raw.var`) column holding gene symbols -- "feature_name"
                  matches the CZ CELLxGENE Discover schema (var.index is
                  the Ensembl gene ID; var["feature_name"] is the human-
                  readable HGNC-style symbol). Only used to populate this
                  function's `kept_gene_symbols` return value and (when
                  given) to resolve `select_gene_symbols`. Set to None to
                  skip reading gene symbols entirely (e.g. for a reference
                  file that doesn't follow the CxG schema and has no
                  comparable column) -- `select_gene_symbols` then cannot
                  be used, and `kept_gene_symbols` is returned as None.
    n_cells_subsample: number of cells to randomly keep (without
                  replacement, uniform over all cells) via pass 2 above.
                  None (or a value >= the dataset's cell count) disables
                  subsampling and keeps every cell -- only recommended for
                  datasets that are known to fit in memory.
    chunk_size:   number of cells read per chunk in each streaming pass.
                  Larger values reduce per-chunk overhead at the cost of a
                  larger transient per-chunk buffer; the default is
                  conservative and does not need tuning for typical
                  reference sizes.
    seed:         seeds the random cell subsample (pass 2) and the scale-
                  check chunk sampling. Does not affect the exact,
                  deterministic gene selection (pass 1).
    validate_raw_counts: if True (default), sanity-check that the selected
                  source's values actually look like raw counts (see
                  _check_value_scale) and raise ValueError if not, before
                  doing any further (more expensive) work. Only disable
                  this if you've independently confirmed the selected
                  source's scale and are hitting a false positive.
    device:       torch device for the returned tensor.

  Returns:
    (X, kept_gene_symbols):
      X: (n_cells_kept, n_genes_kept) float32 tensor, log1p-transformed, no
         NaNs, where n_cells_kept = min(n_cells_subsample, total cells) and
         n_genes_kept = min(n_top_genes, genes with nonzero total count)
         (or, with select_gene_symbols, the number of requested symbols
         actually found with nonzero total count).
      kept_gene_symbols: list[str] of length n_genes_kept, in the same
         column order as X -- the `feature_name_col` value for each kept
         gene -- or None if feature_name_col is None or that column isn't
         present on this file's var/raw.var.

  Note: requires the `anndata` package (`pip install anndata`) -- not
  installed in this dev environment, so this can only be syntax-checked
  here, not executed (same situation as make_synthetic_data6's SERGIO
  dependency). The streaming/chunking/source-selection/scale-check logic
  was separately validated against a pure numpy/scipy mock of anndata's
  backed API (see dev notes) since anndata/h5py aren't available to test
  against a real file here.
  """
  assert source in ("raw", "X", "layer"), f"unknown source {source!r}"
  if select_gene_symbols is not None and n_top_genes is not None:
    raise ValueError(
        "load_reference_h5ad: select_gene_symbols and n_top_genes are "
        "mutually exclusive (two different gene-selection strategies)."
    )
  if select_gene_symbols is not None and feature_name_col is None:
    raise ValueError(
        "load_reference_h5ad: select_gene_symbols requires feature_name_col "
        "to be set (it's used to resolve the requested symbols to columns)."
    )
  try:
    import anndata
  except ImportError as e:
    raise ImportError(
        "load_reference_h5ad requires the 'anndata' package: "
        "pip install anndata"
    ) from e
  import scipy.sparse

  adata = anndata.read_h5ad(path, backed='r')

  if source == "raw":
    if adata.raw is None:
      raise ValueError(
          "load_reference_h5ad(source='raw') requires adata.raw, but this "
          "file has none. Pass source='X' if raw counts are stored "
          "directly in adata.X (e.g. no separate normalized layer was "
          "provided), or source='layer' with an explicit `layer=...` if "
          "raw counts live in a named layer instead."
      )
    assert adata.raw.n_obs == adata.n_obs, (
        f"adata.raw.n_obs ({adata.raw.n_obs}) != adata.n_obs "
        f"({adata.n_obs}) -- expected raw and normalized matrices to stay "
        "cell-aligned."
    )
    matrix = adata.raw.X
    var_df = adata.raw.var
  elif source == "X":
    matrix = adata.X
    var_df = adata.var
  else:
    assert layer is not None, "source='layer' requires an explicit `layer` name"
    matrix = adata.layers[layer]
    var_df = adata.var  # layers share adata.var's gene set/order

  assert len(matrix.shape) == 2, f"expected a 2D matrix, got shape {matrix.shape}"
  n_cells, n_genes = matrix.shape

  symbol_array = None
  if feature_name_col is not None:
    if feature_name_col in var_df.columns:
      symbol_array = var_df[feature_name_col].to_numpy()
    elif select_gene_symbols is not None:
      raise ValueError(
          f"load_reference_h5ad: feature_name_col={feature_name_col!r} not "
          f"found in this file's var columns ({list(var_df.columns)}) -- "
          "cannot resolve select_gene_symbols."
      )

  def _chunks():
    for start in range(0, n_cells, chunk_size):
      end = min(start + chunk_size, n_cells)
      yield start, end, matrix[start:end]

  n_total_chunks = max(1, math.ceil(n_cells / chunk_size))
  sample_chunk_positions = set(
      np.linspace(0, n_total_chunks - 1, min(5, n_total_chunks), dtype=int).tolist()
  )
  sample_rng     = np.random.default_rng(seed)
  sample_budget  = _SCALE_CHECK_SAMPLE_BUDGET // max(1, len(sample_chunk_positions))
  scale_samples  = []

  # --- Pass 1: exact per-gene totals over ALL cells, streamed. Sparse
  # chunks are reduced with sparse-native axis=0 sums -- never densified.
  # Also opportunistically collects a bounded sample of nonzero values
  # (from a handful of chunks spread across the dataset) for the
  # validate_raw_counts scale check below, at no extra I/O cost.
  gene_totals = np.zeros(n_genes, dtype=np.float64)
  for chunk_i, (_, _, chunk) in enumerate(_chunks()):
    if scipy.sparse.issparse(chunk):
      gene_totals += np.asarray(chunk.sum(axis=0)).ravel()
      chunk_values = chunk.data
    else:
      chunk = np.asarray(chunk)
      gene_totals += chunk.sum(axis=0)
      chunk_values = chunk[chunk != 0]
    if validate_raw_counts and chunk_i in sample_chunk_positions and chunk_values.size:
      take = min(chunk_values.size, sample_budget)
      idx  = sample_rng.choice(chunk_values.size, size=take, replace=False)
      scale_samples.append(np.asarray(chunk_values).ravel()[idx])

  if validate_raw_counts:
    _check_value_scale(
        np.concatenate(scale_samples) if scale_samples else np.array([]),
        name=f"{source} matrix in {path!r}", expect="raw_counts",
    )

  # Drop genes with zero total count across all cells: undefined
  # variance/log-mean, and uninformative for the summary statistics below.
  nonzero_genes = np.flatnonzero(gene_totals > 0)
  if select_gene_symbols is not None:
    requested    = set(select_gene_symbols)
    matches      = np.flatnonzero(np.isin(symbol_array, list(requested)))
    keep_genes   = np.intersect1d(nonzero_genes, matches)
    kept_symbols = symbol_array[keep_genes]
    n_found      = len(set(kept_symbols.tolist()))
    n_missing    = len(requested) - n_found
    if n_missing:
      import warnings
      warnings.warn(
          f"load_reference_h5ad(select_gene_symbols=...): {n_missing} of "
          f"{len(requested)} requested gene symbols were not found in "
          f"{feature_name_col!r} (or had zero total count across all "
          f"cells) and were dropped -- kept {n_found} genes.",
          stacklevel=2,
      )
    dup_symbols, dup_counts = np.unique(kept_symbols, return_counts=True)
    dup_symbols = dup_symbols[dup_counts > 1]
    if dup_symbols.size:
      import warnings
      warnings.warn(
          f"load_reference_h5ad(select_gene_symbols=...): {dup_symbols.size} "
          f"requested gene symbol(s) matched more than one row in "
          f"{feature_name_col!r} (e.g. distinct Ensembl IDs that share a "
          "symbol) -- all matching rows were kept as separate columns, so "
          "kept_gene_symbols contains repeated entries for: "
          f"{sorted(dup_symbols.tolist())}. This breaks the usual 1:1 "
          "symbol<->column assumption; if you need exactly one column per "
          "requested symbol, deduplicate the returned (X, kept_gene_symbols) "
          "yourself (e.g. keep only the highest-total-count row per "
          "duplicated symbol).",
          stacklevel=2,
      )
  else:
    keep_genes = nonzero_genes
    if n_top_genes is not None and keep_genes.size > n_top_genes:
      gene_means_kept = gene_totals[keep_genes] / n_cells
      top             = np.argsort(gene_means_kept)[::-1][:n_top_genes]
      keep_genes      = np.sort(keep_genes[top])

  # --- Choose which cells to keep (exact count, uniform without replacement). ---
  rng = np.random.default_rng(seed)
  if n_cells_subsample is not None and n_cells_subsample < n_cells:
    keep_cells = np.sort(rng.choice(n_cells, size=n_cells_subsample, replace=False))
  else:
    keep_cells = np.arange(n_cells)

  # --- Pass 2: sequential streamed extraction of only the selected rows x
  # genes -- small relative to the full matrix, densified incrementally.
  blocks = []
  for start, end, chunk in _chunks():
    lo, hi = np.searchsorted(keep_cells, [start, end])
    if lo == hi:
      continue
    local_idx = keep_cells[lo:hi] - start
    block     = chunk[:, keep_genes][local_idx]
    if scipy.sparse.issparse(block):
      block = block.toarray()
    blocks.append(np.asarray(block, dtype=np.float32))

  X_np = (np.concatenate(blocks, axis=0) if blocks
          else np.zeros((0, keep_genes.size), dtype=np.float32))

  X = torch.tensor(X_np, dtype=torch.float32, device=device)
  X = torch.log1p(X.clamp(min=0))
  kept_gene_symbols = symbol_array[keep_genes].tolist() if symbol_array is not None else None
  return X, kept_gene_symbols


def pca_participation_ratio(ratio: np.ndarray, skip_leading: int = 1) -> float:
  """Participation ratio ("effective number of components") of a PCA
  explained-variance-ratio vector, excluding the first `skip_leading`
  entries (default 1, i.e. PC1/index 0).

  Defined as (sum(tail))**2 / sum(tail**2), where tail = ratio[skip_leading:]
  -- scale-invariant (unaffected by how much of the total variance PC1 or
  any other excluded leading component captured), so it purely measures
  how evenly variance is spread across the remaining components: 1.0 if
  all of the tail's variance sits in a single component, up to
  len(tail) if it's spread perfectly evenly across all of them.

  Returns NaN if fewer than 2 finite, positive tail entries remain (no
  meaningful "spread" to measure -- e.g. n_pca_components < skip_leading + 2).
  """
  tail = np.asarray(ratio, dtype=np.float64)[skip_leading:]
  tail = tail[np.isfinite(tail)]
  if tail.size < 2 or not np.any(tail > 0):
    return float('nan')
  return float((tail.sum() ** 2) / np.sum(tail ** 2))


def compute_summary_stats(
    X:                  torch.Tensor,
    n_pca_components:   int      = 10,
    n_structure_genes:  int|None = 500,
    percentiles:        tuple    = (5, 25, 50, 75, 95),
    seed:               int|None = 0,
    validate_scale:     bool     = True,
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
                       min(n_cells, n_structure_genes) - 1 internally).
    n_structure_genes: if n_genes exceeds this, a random subsample of this
                       many genes is used for *both* the PCA and the
                       pairwise gene-gene correlation statistics below (the
                       same subsample is shared by both, for tractability
                       -- a full SVD or correlation matrix over a whole
                       transcriptome's worth of genes is expensive/large).
                       None disables subsampling (uses all genes for both).
                       Does not affect gene_mean/gene_var/mean-variance/
                       library-size/zero-frac/dropout-curve stats, which
                       are always computed over every gene (cheap, O(n_genes)).
    percentiles:       percentiles reported for each *_p<pct> distribution
                       summary entry.
    seed:              seeds the shared gene subsampling used for the PCA
                       and correlation statistics, and the scale-check
                       sampling below (does not affect any other statistic).
    validate_scale:    if True (default), sanity-check that X's values
                       actually look like log1p(raw counts) -- as opposed
                       to e.g. accidentally-unlogged raw counts, already-
                       scaled/z-scored data, or double-log1p'd data -- and
                       raise ValueError if not (see _check_value_scale).
                       This mirrors load_reference_h5ad's own
                       validate_raw_counts guard, but checks the opposite
                       end of the pipeline: this function's own input,
                       regardless of which loader produced it. Only
                       disable this if you've independently confirmed X's
                       scale and are hitting a false positive.

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
      lib_size_zero_frac: fraction of cells whose total raw count is ~0
        (i.e. every gene dropped out for that cell) -- a direct signal of
        a degenerate/collapsed draw that log_lib_size's percentiles only
        catch incidentally.
      zero_frac: overall fraction of observed entries that are exactly 0.
      dropout_curve_bin_edges: bin edges (over per-gene mean log1p
        expression) used for the binned dropout curve below.
      dropout_curve_zero_frac: mean per-gene zero-fraction within each bin
        of dropout_curve_bin_edges (NaN for empty bins) -- a compact,
        gene-count-independent summary of the zero-fraction/dropout curve,
        comparable to SERGIO's dropout_shape/dropout_percentile knobs.
      pca_explained_variance_ratio: (n_pca_components,) array, computed on
        up to n_structure_genes genes. PCA on the raw (mean-centered only,
        not per-gene standardized) covariance -- so, unlike
        pca_standardized_explained_variance_ratio below, this is affected
        by each gene's absolute variance/scale, not just cross-gene
        correlation structure.
      pca_tail_participation_ratio: scalar "effective number of components"
        among PC2..PC<n_pca_components> (PC1/index 0 excluded -- see
        pca_participation_ratio), i.e. how evenly variance is spread
        across the non-dominant leading PCs rather than how much PC1
        itself dominates. NaN if fewer than 2 non-PC1 components are
        available (n_pca_components < 3). Computed from
        pca_explained_variance_ratio, so inherits its scale-sensitivity.
      pca_standardized_explained_variance_ratio: same shape/semantics as
        pca_explained_variance_ratio, but computed on each gene
        standardized to unit variance first (correlation-matrix PCA
        instead of covariance-matrix PCA; per-gene variance floored at
        1e-6 before dividing). Scale-invariant: rescaling any one gene's
        expression (in isolation) does not change this ratio, unlike
        pca_explained_variance_ratio -- use this one when what you care
        about is cross-gene *correlation* structure independent of which
        genes happen to have the largest absolute variance (see this
        module's AGENTS.md 20260816 entry for why the two can otherwise be
        in tension as tuning objectives).
      pca_standardized_tail_participation_ratio: pca_tail_participation_ratio's
        counterpart computed from pca_standardized_explained_variance_ratio.
      gene_corr_abs_{mean,std,p50,p90}: distribution summary of |pairwise
        gene-gene Pearson correlation| (upper triangle, off-diagonal),
        computed on the same up-to-n_structure_genes genes as PCA above.
      gene_corr_abs_normalized_{mean,std,p50,p90}: same, but computed after
        renormalizing each cell's raw counts to a common total (median
        library size across cells) before log1p -- standard scRNA-seq
        size-factor/CPM-style normalization. Decouples this statistic from
        per-cell library-size (sequencing-depth) variation, which -- via
        e.g. SERGIO's lib_size_effect, a per-cell multiplicative rescale
        applied identically across every gene -- otherwise inflates every
        pairwise raw correlation roughly uniformly (hitting
        gene_corr_abs_mean/p50, dominated by the bulk of gene pairs with
        ~0 true correlation, far harder than gene_corr_abs_std/p90,
        dominated by genuinely co-regulated tail pairs); see this module's
        AGENTS.md 20260818 entry. Exactly analogous to
        pca_standardized_explained_variance_ratio's relationship to
        pca_explained_variance_ratio -- computed in addition to, not
        replacing, the raw family above.
      pca_size_normalized_standardized_explained_variance_ratio: same
        shape/semantics as pca_standardized_explained_variance_ratio, but
        computed on the library-size-renormalized matrix underlying
        gene_corr_abs_normalized_* above (each gene additionally
        standardized to unit variance first, same convention/1e-6 floor as
        pca_standardized_explained_variance_ratio). Combines BOTH
        confound-removal steps: pca_standardized_explained_variance_ratio's
        per-gene standardization removes dependence on which genes happen
        to have the largest absolute variance, but does NOT remove a
        per-CELL multiplicative confound -- SERGIO's lib_size_effect
        rescales a cell's *entire* gene vector by one shared lognormal
        factor, which remains a real, shared source of cross-cell
        correlated variation even after standardizing each column (gene)
        to unit variance, so it can still inflate
        pca_standardized_explained_variance_ratio's PC2+ components as
        library-size variance grows -- confirmed empirically (see this
        module's AGENTS.md 20260820 entry): across the 20260819 tuning
        trials, pca_standardized_pc2_9_explained_variance_ratio's ratio-to-
        target rose monotonically from ~0.7 to ~1.6 as lib_size_scale rose
        across its explored range, with this relationship essentially
        undiluted (not explained by any other knob) in a full rank-
        regression against all 18 search-space parameters. Use this family
        instead of pca_standardized_explained_variance_ratio when you want
        PCA structure fidelity independent of *both* confounds at once.
      pca_size_normalized_standardized_tail_participation_ratio:
        pca_tail_participation_ratio's counterpart computed from
        pca_size_normalized_standardized_explained_variance_ratio.
  """
  assert X.dim() == 2, f"expected a 2D (n_cells, n_genes) tensor, got shape {tuple(X.shape)}"
  n_cells, n_genes = X.shape
  X_np = X.detach().to(torch.float64).cpu().numpy()
  obs_mask = ~np.isnan(X_np)

  if validate_scale:
    nonzero = X_np[obs_mask & (X_np != 0)]
    if nonzero.size > _SCALE_CHECK_SAMPLE_BUDGET:
      idx = np.random.default_rng(seed).choice(
          nonzero.size, size=_SCALE_CHECK_SAMPLE_BUDGET, replace=False)
      nonzero = nonzero[idx]
    _check_value_scale(nonzero, name="compute_summary_stats input X", expect="log1p_counts")

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
  # Fraction of cells with ~0 total counts -- log_lib_size's percentiles
  # only catch this incidentally (e.g. if p50 happens to land on 0); this
  # is a direct, robust signal of a degenerate/collapsed simulation draw
  # (most of the matrix wiped out by dropout for a chunk of cells) that a
  # tuning harness can guard against explicitly (see tune_synthetic_data.py's
  # max_lib_size_zero_frac).
  stats["lib_size_zero_frac"] = float(np.mean(lib_size <= 0.5))

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

  # Shared gene subsample for PCA + gene-gene correlation (both O(n_genes^2)-
  # ish or worse, unlike the per-gene stats above) -- capped at
  # n_structure_genes for tractability on a full transcriptome, and shared
  # between the two so they describe the same gene subset.
  rng = np.random.default_rng(seed)
  if n_structure_genes is not None and n_genes > n_structure_genes:
    gene_idx = np.sort(rng.choice(n_genes, size=n_structure_genes, replace=False))
  else:
    gene_idx = np.arange(n_genes)
  X_struct = X_filled[:, gene_idx]
  n_struct_genes = len(gene_idx)

  k = max(0, min(n_pca_components, n_cells - 1, n_struct_genes - 1))
  if k >= 1:
    Xc = X_struct - X_struct.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    explained_var = s ** 2
    total = explained_var.sum()
    ratio = explained_var / total if total > 0 else np.zeros_like(explained_var)
    stats["pca_explained_variance_ratio"] = ratio[:k]

    # Standardized (per-gene z-scored) companion: PCA on the correlation
    # matrix rather than the raw (non-standardized) covariance matrix above.
    # The raw version's explained-variance-ratio is directly driven by
    # which genes happen to have the largest absolute variance -- so it is
    # NOT independent of each gene's expression *scale*, only of the
    # dataset's overall scale (mean-centering only, no per-gene rescaling,
    # see Xc above). This makes "match the reference's PCA structure" and
    # "match the reference's per-gene variance magnitude" two different
    # views of the *same* underlying per-gene variances rather than
    # orthogonal concerns (see AGENTS.md's 20260816 entry for the full
    # analysis) -- concentrating variance in a few genes to build PC2+
    # structure mechanically pulls down the population's average variance.
    # Standardizing each gene to unit variance before PCA removes this:
    # the resulting explained-variance-ratio only reflects *relative*
    # cross-gene correlation structure, unaffected by any gene's absolute
    # variance/scale. Gene variances are floored at 1e-6 (same convention
    # as this module's MCAR-imputation gap-fill, see _random_fill) before
    # dividing, so a near-constant gene in this subsample doesn't blow up
    # into a spurious dominant component.
    gene_std_struct = np.clip(X_struct.std(axis=0, keepdims=True), 1e-6, None)
    Xz = (X_struct - X_struct.mean(axis=0, keepdims=True)) / gene_std_struct
    sz = np.linalg.svd(Xz, full_matrices=False, compute_uv=False)
    explained_var_z = sz ** 2
    total_z = explained_var_z.sum()
    ratio_z = explained_var_z / total_z if total_z > 0 else np.zeros_like(explained_var_z)
    stats["pca_standardized_explained_variance_ratio"] = ratio_z[:k]
  else:
    stats["pca_explained_variance_ratio"] = np.array([])
    stats["pca_standardized_explained_variance_ratio"] = np.array([])
  stats["pca_tail_participation_ratio"] = pca_participation_ratio(stats["pca_explained_variance_ratio"])
  stats["pca_standardized_tail_participation_ratio"] = pca_participation_ratio(
      stats["pca_standardized_explained_variance_ratio"])

  if n_struct_genes >= 2:
    with np.errstate(invalid='ignore'):
      corr_matrix = np.corrcoef(X_struct, rowvar=False)
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

  # --- library-size-normalized companion: decouples gene-gene correlation
  # from per-cell library-size (sequencing-depth) variation -- the raw
  # version above is computed on X_struct, which is mean-centered/filled
  # log1p(raw counts) with each cell's *actual* total count baked in. A
  # per-cell multiplicative library-size effect (e.g. SERGIO's own
  # lib_size_effect, which rescales a cell's entire gene vector by one
  # shared lognormal factor before log1p) is by construction perfectly
  # correlated across every gene, so it inflates *every* pairwise raw
  # correlation roughly uniformly as library-size variance grows --
  # hitting gene_corr_abs_mean/p50 (dominated by the bulk of gene pairs,
  # which have ~0 true correlation in real reference data) far harder
  # than gene_corr_abs_std/p90 (dominated by genuinely co-regulated tail
  # pairs, which already have real covariance to compete with the
  # confound). Confirmed empirically (see AGENTS.md's 20260818 entry):
  # across ~1900 real tuning trials, gene_corr_abs_mean/p50 explode to
  # 6-17x the target's value well before gene_var_mean/gene_mean_mean/
  # pca_standardized_pc2_9 (which all *improve* with the same knob) reach
  # their own best match, actively blocking the region where those other
  # stats would otherwise land close to target. Renormalizing each cell to
  # a common total count before log1p (standard scRNA-seq size-factor/CPM-
  # style normalization) removes this shared multiplicative confound while
  # preserving genuine cross-gene covariance, exactly analogous to
  # pca_standardized_explained_variance_ratio's per-gene standardization
  # removing PC1's dependence on absolute gene variance. Computed *in
  # addition to*, not replacing, the raw family above.
  lib_size_safe  = np.maximum(lib_size, 1e-8)
  size_factor    = np.median(lib_size_safe) / lib_size_safe
  raw_for_norm   = np.where(obs_mask, np.expm1(X_np), np.nan)
  X_normalized   = np.log1p(raw_for_norm * size_factor[:, np.newaxis])
  with np.errstate(invalid='ignore'):
    gene_mean_norm = np.nanmean(X_normalized, axis=0)
  fill_norm            = np.where(np.isfinite(gene_mean_norm), gene_mean_norm, 0.0)
  X_normalized_filled  = np.where(obs_mask, X_normalized, fill_norm[np.newaxis, :])
  X_struct_normalized  = X_normalized_filled[:, gene_idx]

  if n_struct_genes >= 2:
    with np.errstate(invalid='ignore'):
      corr_matrix_norm = np.corrcoef(X_struct_normalized, rowvar=False)
    abs_corr_norm = np.abs(corr_matrix_norm[iu])
    abs_corr_norm = abs_corr_norm[np.isfinite(abs_corr_norm)]
  else:
    abs_corr_norm = np.array([])
  if abs_corr_norm.size:
    stats["gene_corr_abs_normalized_mean"] = float(np.mean(abs_corr_norm))
    stats["gene_corr_abs_normalized_std"]  = float(np.std(abs_corr_norm))
    stats["gene_corr_abs_normalized_p50"]  = float(np.percentile(abs_corr_norm, 50))
    stats["gene_corr_abs_normalized_p90"]  = float(np.percentile(abs_corr_norm, 90))
  else:
    stats["gene_corr_abs_normalized_mean"] = float('nan')
    stats["gene_corr_abs_normalized_std"]  = float('nan')
    stats["gene_corr_abs_normalized_p50"]  = float('nan')
    stats["gene_corr_abs_normalized_p90"]  = float('nan')

  # --- library-size-normalized + per-gene-standardized PCA companion: see
  # pca_size_normalized_standardized_explained_variance_ratio's docstring
  # above for why this combines the two confound-removal steps above
  # (pca_standardized_explained_variance_ratio's per-gene standardization,
  # and gene_corr_abs_normalized_*'s per-cell library-size renormalization)
  # rather than just reusing one or the other. Computed on
  # X_struct_normalized (already built above for gene_corr_abs_normalized_*),
  # standardized exactly like pca_standardized_explained_variance_ratio's Xz.
  if k >= 1:
    gene_std_struct_norm = np.clip(X_struct_normalized.std(axis=0, keepdims=True), 1e-6, None)
    Xzn = (X_struct_normalized - X_struct_normalized.mean(axis=0, keepdims=True)) / gene_std_struct_norm
    szn = np.linalg.svd(Xzn, full_matrices=False, compute_uv=False)
    explained_var_zn = szn ** 2
    total_zn = explained_var_zn.sum()
    ratio_zn = explained_var_zn / total_zn if total_zn > 0 else np.zeros_like(explained_var_zn)
    stats["pca_size_normalized_standardized_explained_variance_ratio"] = ratio_zn[:k]
  else:
    stats["pca_size_normalized_standardized_explained_variance_ratio"] = np.array([])
  stats["pca_size_normalized_standardized_tail_participation_ratio"] = pca_participation_ratio(
      stats["pca_size_normalized_standardized_explained_variance_ratio"])

  return stats


def _percentile_profile(stats: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
  """Extract the `{prefix}_p<N>` distribution-summary entries written by
  compute_summary_stats's `_dist_summary` helper (e.g. prefix="gene_mean"
  picks up gene_mean_p5, gene_mean_p25, ...), sorted by percentile. Works
  for whatever `percentiles` tuple the caller used when computing `stats`
  -- not hardcoded to any particular set.

  Returns (percentiles, values) as parallel 1D arrays (both empty if no
  matching keys are present).
  """
  pat = re.compile(rf"^{re.escape(prefix)}_p(\d+(?:\.\d+)?)$")
  pts = [(float(m.group(1)), stats[k]) for k in stats if (m := pat.match(k))]
  pts.sort(key=lambda t: t[0])
  if not pts:
    return np.array([]), np.array([])
  xs, ys = zip(*pts)
  return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _grouped_bars(ax, items: list[tuple[str, dict]], keys: list[str],
                   key_labels: list[str] | None = None) -> bool:
  """Draw one group of bars per (label, stats_dict) in `items`, with one
  bar per entry in `keys` within each group -- e.g. keys=["zero_frac"]
  gives one plain bar per dict, keys=["gene_corr_abs_mean",
  "gene_corr_abs_p50", "gene_corr_abs_p90"] gives 3 bars per dict.

  Returns False (and leaves `ax` untouched beyond an explanatory message)
  if none of `keys` is present with a finite value in any of `items`.
  """
  key_labels = key_labels or keys
  values = np.array(
      [[float(d.get(k, float('nan'))) for k in keys] for _, d in items],
      dtype=np.float64,
  )
  if not np.isfinite(values).any():
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    return False

  n_items, n_keys = values.shape
  width = 0.8 / max(n_keys, 1)
  x = np.arange(n_items)
  for j in range(n_keys):
    ax.bar(x + j * width - 0.4 + width / 2, values[:, j], width=width, label=key_labels[j])
  ax.set_xticks(x)
  ax.set_xticklabels([label for label, _ in items], rotation=20, ha="right")
  if n_keys > 1:
    ax.legend(fontsize=7)
  return True


def plot_summary_stats(
    stats:   dict | list[dict] | dict[str, dict],
    labels:  list[str] | None = None,
    path:    str | None = None,
    figsize: tuple = (16, 8),
):
  """
  Build an 8-panel figure summarizing/comparing one or more
  compute_summary_stats() dicts -- e.g. tune_synthetic_data.py's
  `target_stats` (real reference) overlaid with one or more candidate
  `candidate_stats` (synthetic), or several synthetic configs against
  each other.

  Args:
    stats:  a single compute_summary_stats() dict, a list of such dicts,
            or a {label: dict} mapping (mapping keys are used directly as
            legend/x-axis labels, taking precedence over `labels`).
    labels: legend/x-axis label per dict, used when `stats` is a single
            dict or a list of dicts. Defaults to "stats" (single dict) or
            "stats 0", "stats 1", ... (list). Ignored when `stats` is
            itself a {label: dict} mapping.
    path:   if given, the figure is saved here (dpi=120) and closed (like
            sample_efficiency.py's `_plot`); otherwise the open Figure is
            returned for the caller to show/save/close itself.
    figsize: passed to plt.subplots.

  Panels (2x4 grid), one line/bar-group per input dict in each:
    1. gene_mean percentile profile
    2. gene_var  percentile profile (log-y)
    3. log_lib_size percentile profile
    4. dropout curve (dropout_curve_zero_frac vs. binned per-gene mean
       expression, i.e. the midpoints of dropout_curve_bin_edges)
    5. pca_explained_variance_ratio vs. component index (log-y)
    6. gene_corr_abs_{mean,p50,p90} grouped bars
    7. zero_frac bar
    8. mean_var_log_{slope,corr} grouped bars

  Any panel whose required key(s)/arrays are absent or empty in *every*
  input dict is left blank with a "no data" note rather than raising --
  e.g. calling this on stats produced with n_pca_components=0.

  Returns the Figure (already closed if `path` was given).
  """
  import matplotlib.pyplot as plt

  if isinstance(stats, dict) and stats and all(isinstance(v, dict) for v in stats.values()):
    items = list(stats.items())
  elif isinstance(stats, dict):
    items = [(labels[0] if labels else "stats", stats)]
  else:
    stats = list(stats)
    labels = labels or [f"stats {i}" for i in range(len(stats))]
    assert len(labels) == len(stats), (
        f"got {len(labels)} labels for {len(stats)} stats dicts"
    )
    items = list(zip(labels, stats))
  assert items, "plot_summary_stats: need at least one stats dict"

  fig, axes = plt.subplots(2, 4, figsize=figsize)

  # --- 1-3: percentile profiles ---
  for ax, prefix, ylabel, logy in (
      (axes[0, 0], "gene_mean",    "log1p expression", False),
      (axes[0, 1], "gene_var",     "variance",          True),
      (axes[0, 2], "log_lib_size", "log1p(lib size)",   False),
  ):
    any_data = False
    for label, d in items:
      xs, ys = _percentile_profile(d, prefix)
      if xs.size:
        ax.plot(xs, ys, marker="o", ms=4, label=label)
        any_data = True
    if any_data:
      ax.set_xlabel("percentile")
      ax.set_ylabel(ylabel)
      if logy:
        ax.set_yscale("log")
      ax.legend(fontsize=7)
    else:
      ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{prefix} percentile profile")

  # --- 4: dropout curve ---
  ax = axes[0, 3]
  any_data = False
  for label, d in items:
    edges = np.asarray(d.get("dropout_curve_bin_edges", []), dtype=np.float64)
    curve = np.asarray(d.get("dropout_curve_zero_frac", []), dtype=np.float64)
    if edges.size >= 2 and curve.size:
      mids = 0.5 * (edges[:-1] + edges[1:])
      n = min(mids.size, curve.size)
      ax.plot(mids[:n], curve[:n], marker="o", ms=4, label=label)
      any_data = True
  if any_data:
    ax.set_xlabel("per-gene mean expression (binned)")
    ax.set_ylabel("zero fraction")
    ax.legend(fontsize=7)
  else:
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
  ax.set_title("dropout curve")

  # --- 5: PCA explained-variance ratio ---
  ax = axes[1, 0]
  any_data = False
  for label, d in items:
    ratio = np.asarray(d.get("pca_explained_variance_ratio", []), dtype=np.float64)
    if ratio.size:
      ax.plot(np.arange(1, ratio.size + 1), ratio, marker="o", ms=4, label=label)
      any_data = True
  if any_data:
    ax.set_xlabel("PCA component")
    ax.set_ylabel("explained variance ratio")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
  else:
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
  ax.set_title("pca_explained_variance_ratio")

  # --- 6-8: grouped bars ---
  _grouped_bars(axes[1, 1], items,
                ["gene_corr_abs_mean", "gene_corr_abs_p50", "gene_corr_abs_p90"],
                ["mean", "p50", "p90"])
  axes[1, 1].set_title("gene_corr_abs")

  _grouped_bars(axes[1, 2], items, ["zero_frac"])
  axes[1, 2].set_title("zero_frac")

  _grouped_bars(axes[1, 3], items,
                ["mean_var_log_slope", "mean_var_log_corr"],
                ["slope", "corr"])
  axes[1, 3].set_title("mean_var_log_{slope,corr}")

  fig.tight_layout()
  if path is not None:
    fig.savefig(path, dpi=120)
    plt.close(fig)
  return fig


# --------------------------------------------------------------------------
# MR-tree paired experiment: tree-correlated vs. i.i.d. mr_state sampling.
# --------------------------------------------------------------------------
# Compares sample_mr_state_from_tree's tree-correlated MR states (built from
# build_mr_overlap_tree) against its own tree_strength=0.0 control, holding
# everything else (GRN topology, simulation parameters, seeds) fixed, via
# compute_summary_stats -- see this module's/AGENTS.md's "MR overlap tree"
# design discussion for the full rationale. Reuses tune_synthetic_data.py's
# config-resolution/distance-scoring helpers (imported lazily -- see
# _import_tune_synthetic_data -- to avoid a circular top-level import, since
# tune_synthetic_data.py itself imports several names from this module) so a
# tune_synthetic_data.py-style JSON config is handled identically to how an
# actual tuning trial would.
#
# Typical usage (Colab or command line):
#   import synthetic_data
#   result = synthetic_data.run_mr_tree_experiment(
#       "synthetic_tuning_20260820.00/tuned_synthetic_config.20260820.00.json",
#       n_replicates=20, tree_strength=1.0)
#   print(result["summary"])
# or, from the command line:
#   python synthetic_data.py --config tuned_synthetic_config.20260820.00.json \
#       --n-replicates 20 --tree-strength 1.0 --output result.pickle

def _import_tune_synthetic_data():
  """Lazily import tune_synthetic_data.py (deferred to function-call time,
  not module import time, to avoid a circular top-level import -- that
  module imports several names from this one at ITS OWN module level).
  Used by _load_mr_tree_experiment_config/run_mr_tree_experiment to reuse
  its config-resolution/distance-scoring helpers rather than duplicating
  them here."""
  try:
    import tune_synthetic_data as _tsd
  except ImportError as e:
    raise ImportError(
        "This function relies on tune_synthetic_data.py's own config-"
        "resolution/distance-scoring helpers (_CONFIG_DEFAULTS, "
        "_SIM_KWARG_DEFAULTS, _GRN_KWARG_DEFAULTS, _split_resolved, "
        "compute_stats_distance), which in turn requires that module's "
        "own dependencies (optuna, tqdm) to be importable -- even though "
        "nothing here runs an actual Optuna study."
    ) from e
  return _tsd


def _load_mr_tree_experiment_config(config) -> dict:
  """
  Normalize `config` (a path to a tune_synthetic_data.py-style JSON
  config, or an already-loaded dict of the same shape) into a flat dict
  of every keyword argument run_mr_tree_experiment needs to call
  generate_sergio_grn_from_reference / make_synthetic_data6 /
  compute_summary_stats identically to how tune_synthetic_data.run_trial
  would for that same config.

  Accepts either shape of JSON tune_synthetic_data.py produces/consumes:
    - a *tuning* config (e.g. synthetic_tuning_config.20260820.00.json):
      flat top-level keys (reference_grn_path, n_clusters, n_genes, ...)
      plus "search_space" (ignored here -- this experiment does not tune
      anything, it needs one concrete parameter set, not a search space).
      Every tunable key (see tune_synthetic_data.TUNABLE_KEYS) MUST be
      given a fixed value directly at this config's top level (not merely
      a search_space entry) for it to be anything other than tune_
      synthetic_data's plain default here -- a bare tuning config carries
      no *resolved* values for its search_space keys at all.
    - a *tuned-result* config (e.g. tuned_synthetic_config.20260820.00.json,
      i.e. run()'s own `output` -- what this project actually uses in
      practice day to day): structural/reference settings live under
      "_meta" and resolved tunable values live under
      "best_resolved_params" (falling back to "best_params" if that key
      is absent). Note that a tuned-result config's "_meta" is a much
      smaller key set than a tuning config's top level (only
      reference_h5ad_path/target_stats_path/reference_grn_path/
      n_clusters/n_genes/n_cells/sampling_state/shared_coop_state/
      grn_seed/sim_seed -- see tune_synthetic_data.run()'s own `summary`
      dict) -- so mr_rate_low/mr_rate_high/dt/noise_type/
      min_cells_per_cluster/add_*/grn_delimiter & friends/stats_*/
      weights/max_lib_size_zero_frac/max_pca_pc1_ratio_factor are NOT
      captured by a tuned-result config and fall back to tune_
      synthetic_data's plain defaults here. Pass the *original* tuning
      config instead (or run_mr_tree_experiment's own stats_*/
      mr_rate_low/mr_rate_high override kwargs) if you need those to
      match a specific tuning run exactly.

  A plain flat dict (already containing every needed key directly, no
  "_meta"/"best_resolved_params" nesting) is also accepted unchanged.

  Returns the merged dict: {**_CONFIG_DEFAULTS, **_SIM_KWARG_DEFAULTS,
  **_GRN_KWARG_DEFAULTS, **flat}. Raises ValueError if reference_grn_path/
  n_genes/n_clusters/n_cells are still missing after that merge (i.e. not
  a plain _CONFIG_DEFAULTS key and not present in `config` either).
  """
  _tsd = _import_tune_synthetic_data()

  if isinstance(config, str):
    with open(config) as f:
      raw = json.load(f)
  else:
    raw = dict(config)

  if "_meta" in raw:
    flat = {**raw["_meta"], **(raw.get("best_resolved_params") or raw.get("best_params") or {})}
  else:
    if "search_space" in raw and not (raw.get("best_resolved_params") or raw.get("best_params")):
      print(
          "*** WARNING: this config looks like a bare *tuning* config "
          "(has 'search_space' but no 'best_resolved_params'/'best_params') "
          "-- every tunable parameter will fall back to tune_synthetic_"
          "data's plain defaults, NOT any tuned value. Pass the tuning "
          "*output* config (e.g. tuned_synthetic_config....json) if you "
          "want the actual best-found parameters instead. ***"
      )
    flat = dict(raw)
    flat.pop("search_space", None)

  merged = {**_tsd._CONFIG_DEFAULTS, **_tsd._SIM_KWARG_DEFAULTS, **_tsd._GRN_KWARG_DEFAULTS, **flat}

  required = ["reference_grn_path", "n_genes", "n_clusters", "n_cells"]
  missing = [k for k in required if merged.get(k) is None]
  if missing:
    raise ValueError(f"MR-tree experiment config is missing required key(s): {missing}")

  return merged


def _augment_candidate_stats(stats: dict, _tsd) -> None:
  """Applies the same derived-key augmentations tune_synthetic_data.py's
  run()/run_trial() apply to every target/candidate stats dict (the
  pca_*_pc2_9_explained_variance_ratio slices and nonzero_frac) -- see
  those functions' own calls to _add_pca_pc2_9_key/_add_nonzero_frac_key.
  Mutates `stats` in place; a no-op for any key whose source array/value
  is absent (see those helpers' own docstrings). Only matters for this
  module's optional compute_stats_distance comparison below, since a
  config's `weights` dict may reference these derived keys."""
  _tsd._add_pca_pc2_9_key(stats)
  _tsd._add_pca_pc2_9_key(
      stats,
      src_key="pca_standardized_explained_variance_ratio",
      dst_key="pca_standardized_pc2_9_explained_variance_ratio",
  )
  _tsd._add_pca_pc2_9_key(
      stats,
      src_key="pca_size_normalized_standardized_explained_variance_ratio",
      dst_key="pca_size_normalized_standardized_pc2_9_explained_variance_ratio",
  )
  _tsd._add_nonzero_frac_key(stats)


def _mr_state_participation_ratio(mr_state: np.ndarray) -> float:
  """Participation ratio of the singular-value spectrum of an
  (n_states, n_mrs) mr_state matrix: (sum(sv**2))**2 / sum(sv**4), the
  "effective number of independent directions" among the n_states MR-state
  rows. Returns NaN when fewer than 2 rows or all singular values are zero.
  This is a natural companion to pca_participation_ratio: a purely i.i.d.
  sampler will tend toward n_mrs (all MRs move independently), while a
  tree-correlated sampler with one dominant module will tend toward 1."""
  if mr_state.shape[0] < 2 or mr_state.shape[1] < 1:
    return float("nan")
  sv = np.linalg.svd(mr_state - mr_state.mean(axis=0, keepdims=True),
                     full_matrices=False, compute_uv=False)
  sv2 = sv ** 2
  total = sv2.sum()
  if total <= 0.0:
    return float("nan")
  return float(total ** 2 / np.sum(sv2 ** 2))


def _gene_module_correlations(
    X_sim,
    cluster_labels: np.ndarray,
    winner_mr:      list | None,
    mr_ids:         list,
    gene_id_to_symbol: dict,
    n_structure_genes: int | None = 500,
    seed:              int | None = 0,
) -> dict:
  """
  Compute within-module vs. across-module gene-pair Pearson correlation
  using the winner_mr assignments from generate_sergio_grn_from_reference's
  diagnostics to define which gene belongs to which regulatory module.

  winner_mr[k] is the dominant master regulator for the k-th target gene
  (in diagnostics["tgt_ids"] order, parallel to diagnostics["winner_mr"]).
  Genes whose winner_mr is the same are considered in the same module.

  Returns a dict with:
    "within_module_mean_abs_corr":  mean |corr| among gene pairs in the
                                    same module (NaN if no within-module
                                    pairs exist, e.g. all modules are
                                    singletons).
    "across_module_mean_abs_corr":  mean |corr| among gene pairs in
                                    different modules (NaN if fewer than
                                    2 modules).
    "within_minus_across":          difference (NaN if either is NaN).
    "n_modules":                    number of distinct winner_mr values.
    "n_within_pairs":               number of within-module pairs scored.
    "n_across_pairs":               number of across-module pairs scored.

  All three correlation values use the same size-normalized standardized
  data path as compute_summary_stats (raw counts renormalized to median
  library size -> log1p -> per-gene standardize), so they are directly
  comparable to gene_corr_abs_normalized_* from compute_summary_stats.

  Returns a dict with all values NaN / 0 if winner_mr is None/empty, or
  if X_sim has fewer than 2 genes, or if there are no valid pairs.
  """
  nan_result = {
      "within_module_mean_abs_corr": float("nan"),
      "across_module_mean_abs_corr": float("nan"),
      "within_minus_across":         float("nan"),
      "n_modules":                   0,
      "n_within_pairs":              0,
      "n_across_pairs":              0,
  }
  if winner_mr is None or len(winner_mr) == 0:
    return nan_result

  X_np = X_sim.detach().to(torch.float64).cpu().numpy() if hasattr(X_sim, "detach") else np.asarray(X_sim, dtype=np.float64)
  n_cells, n_genes = X_np.shape
  if n_genes < 2:
    return nan_result

  # Library-size-normalize then per-gene standardize, same as compute_
  # summary_stats's pca_size_normalized_standardized_* / gene_corr_abs_
  # normalized_* path.
  obs_mask = ~np.isnan(X_np)
  raw = np.expm1(np.where(obs_mask, X_np, 0.0))
  raw[~obs_mask] = 0.0
  lib_size      = raw.sum(axis=1)
  lib_size_safe = np.maximum(lib_size, 1e-8)
  size_factor   = np.median(lib_size_safe) / lib_size_safe
  raw_for_norm  = np.where(obs_mask, np.expm1(X_np), np.nan)
  X_norm        = np.log1p(raw_for_norm * size_factor[:, np.newaxis])
  with np.errstate(invalid="ignore"):
    gene_mean_norm = np.nanmean(X_norm, axis=0)
  fill_norm    = np.where(np.isfinite(gene_mean_norm), gene_mean_norm, 0.0)
  X_norm_filled = np.where(obs_mask, X_norm, fill_norm[np.newaxis, :])
  gene_std = np.clip(X_norm_filled.std(axis=0, keepdims=True), 1e-6, None)
  X_std = (X_norm_filled - X_norm_filled.mean(axis=0, keepdims=True)) / gene_std

  # Subsample genes for tractability (same budget as compute_summary_stats).
  rng = np.random.default_rng(seed)
  if n_structure_genes is not None and n_genes > n_structure_genes:
    gene_idx = np.sort(rng.choice(n_genes, size=n_structure_genes, replace=False))
  else:
    gene_idx = np.arange(n_genes)
  X_sub = X_std[:, gene_idx]

  # Map each sampled gene index back to its winner_mr (via gene_id_to_symbol
  # / tgt_ids alignment in diagnostics). winner_mr is parallel to tgt_ids.
  # gene_id_to_symbol maps gene_id (int) -> symbol; here we need gene_id ->
  # winner_mr. Build that mapping from diagnostics lists.
  # (MR genes themselves have no winner_mr entry -- they are excluded from
  # the within/across comparison, since only target genes have a winner_mr.)
  gene_id_to_winner: dict = {}
  for gene_id_val, wm in zip(winner_mr[0], winner_mr[1]):
    gene_id_to_winner[int(gene_id_val)] = int(wm)

  gene_winners = np.array(
      [gene_id_to_winner.get(int(gene_idx[i]), -1) for i in range(len(gene_idx))],
      dtype=np.int64,
  )
  # -1 means "no winner_mr" (MR gene or not in diagnostics) -- exclude from pairs.
  valid_mask = gene_winners >= 0
  if valid_mask.sum() < 2:
    return nan_result

  X_valid   = X_sub[:, valid_mask]
  gw_valid  = gene_winners[valid_mask]
  n_valid   = X_valid.shape[1]

  with np.errstate(invalid="ignore"):
    corr = np.corrcoef(X_valid, rowvar=False)
  np.fill_diagonal(corr, np.nan)

  iu_rows, iu_cols = np.triu_indices(n_valid, k=1)
  same_module   = gw_valid[iu_rows] == gw_valid[iu_cols]
  abs_corr_vals = np.abs(corr[iu_rows, iu_cols])
  finite        = np.isfinite(abs_corr_vals)

  within_vals = abs_corr_vals[same_module  & finite]
  across_vals = abs_corr_vals[~same_module & finite]

  within_mean = float(np.mean(within_vals)) if within_vals.size else float("nan")
  across_mean = float(np.mean(across_vals)) if across_vals.size else float("nan")

  return {
      "within_module_mean_abs_corr": within_mean,
      "across_module_mean_abs_corr": across_mean,
      "within_minus_across":         (within_mean - across_mean)
                                     if math.isfinite(within_mean) and math.isfinite(across_mean)
                                     else float("nan"),
      "n_modules":      int(len(set(gw_valid.tolist()))),
      "n_within_pairs": int(within_vals.size),
      "n_across_pairs": int(across_vals.size),
  }


def _summarize_mr_tree_experiment(replicates: list) -> dict:
  """
  Aggregate paired tree-vs-control differences across run_mr_tree_
  experiment's replicates.

  Scalar metrics (each -> {"tree_mean", "control_mean", "diff_mean",
  "diff_std", "n"}): differences are computed paired (tree - control per
  replicate), so diff_mean/diff_std reflect within-replicate differences
  rather than between-arm marginals.  n = number of replicates where both
  arms' value was finite.

  Array metrics (each -> {"tree_mean": 1D array, "control_mean": 1D array,
  "diff_mean": 1D array, "diff_std": 1D array, "n": int}): elementwise
  paired differences across replicates where both arms' array had the same
  finite length; n is the number of such replicates.

  Scalar metrics returned:
    pca_size_normalized_standardized_pc1_ratio
    pca_size_normalized_standardized_pc2_9_sum
    pca_size_normalized_standardized_tail_participation_ratio
    gene_corr_abs_normalized_mean
    gene_corr_abs_normalized_p90
    gene_mean_mean, gene_var_mean, log_lib_size_std
    zero_frac
    mr_state_participation_ratio
    within_module_mean_abs_corr
    across_module_mean_abs_corr
    within_minus_across
    distance (NaN entries when target_stats unavailable)

  Array metrics returned:
    pca_size_normalized_standardized_explained_variance_ratio
      (per-PC mean and std, for directly inspecting the full spectrum)
    top_distance_terms
      (mean weighted contribution per key across replicates, tree and
      control separately -- helps identify which stats actually drove any
      distance change, not just the scalar total)
  """
  def _scalar(arm, *keys):
    """Dig out a scalar stat from arm['stats'] or arm directly, trying
    each key in order; returns NaN if none found or not finite."""
    for k in keys:
      v = arm["stats"].get(k) if k not in arm else arm.get(k)
      if v is None:
        v = arm["stats"].get(k)
      if v is not None:
        try:
          f = float(v)
          if math.isfinite(f):
            return f
        except (TypeError, ValueError):
          pass
    return float("nan")

  scalar_metrics = {
      "pca_size_normalized_standardized_pc1_ratio": lambda arm: (
          float(arm["stats"]["pca_size_normalized_standardized_explained_variance_ratio"][0])
          if len(arm["stats"].get("pca_size_normalized_standardized_explained_variance_ratio", [])) >= 1
          else float("nan")),
      "pca_size_normalized_standardized_pc2_9_sum": lambda arm: (
          float(np.sum(arm["stats"]["pca_size_normalized_standardized_explained_variance_ratio"][1:9]))
          if len(arm["stats"].get("pca_size_normalized_standardized_explained_variance_ratio", [])) >= 9
          else float("nan")),
      "pca_size_normalized_standardized_tail_participation_ratio": lambda arm:
          _scalar(arm, "pca_size_normalized_standardized_tail_participation_ratio"),
      "gene_corr_abs_normalized_mean": lambda arm: _scalar(arm, "gene_corr_abs_normalized_mean"),
      "gene_corr_abs_normalized_p90":  lambda arm: _scalar(arm, "gene_corr_abs_normalized_p90"),
      "gene_mean_mean":     lambda arm: _scalar(arm, "gene_mean_mean"),
      "gene_var_mean":      lambda arm: _scalar(arm, "gene_var_mean"),
      "log_lib_size_std":   lambda arm: _scalar(arm, "log_lib_size_std"),
      "zero_frac":          lambda arm: _scalar(arm, "zero_frac"),
      "mr_state_participation_ratio": lambda arm: _scalar(arm, "mr_state_participation_ratio"),
      "within_module_mean_abs_corr":  lambda arm: _scalar(arm, "within_module_mean_abs_corr"),
      "across_module_mean_abs_corr":  lambda arm: _scalar(arm, "across_module_mean_abs_corr"),
      "within_minus_across":          lambda arm: _scalar(arm, "within_minus_across"),
      "distance": lambda arm: float(arm["distance"]) if arm.get("distance") is not None else float("nan"),
  }

  summary: dict = {}
  for name, fn in scalar_metrics.items():
    tree_vals = np.array([fn(rep["tree"])    for rep in replicates], dtype=np.float64)
    ctrl_vals = np.array([fn(rep["control"]) for rep in replicates], dtype=np.float64)
    valid = np.isfinite(tree_vals) & np.isfinite(ctrl_vals)
    diff  = tree_vals[valid] - ctrl_vals[valid]
    summary[name] = {
        "tree_mean":    float(np.mean(tree_vals[valid])) if valid.any() else float("nan"),
        "control_mean": float(np.mean(ctrl_vals[valid])) if valid.any() else float("nan"),
        "diff_mean":    float(np.mean(diff))              if valid.any() else float("nan"),
        "diff_std":     float(np.std(diff))               if valid.any() else float("nan"),
        "n":            int(valid.sum()),
    }

  # --- per-PC PCA spectrum: elementwise mean/std across replicates ---
  for arm_name in ("tree", "control"):
    key = "pca_size_normalized_standardized_explained_variance_ratio"
    arrays = []
    for rep in replicates:
      v = rep[arm_name]["stats"].get(key)
      if v is not None and len(v):
        arrays.append(np.asarray(v, dtype=np.float64))
    if arrays:
      # align to shortest (different n_pca_components would be a config
      # mistake, but handle gracefully rather than crashing)
      min_len = min(len(a) for a in arrays)
      mat = np.stack([a[:min_len] for a in arrays], axis=0)  # (n_reps, n_pcs)
      summary.setdefault("pca_spectrum_per_arm", {})[arm_name] = {
          "mean": mat.mean(axis=0).tolist(),
          "std":  mat.std(axis=0).tolist(),
          "n":    len(arrays),
      }

  # --- top distance terms: mean weighted contribution per key per arm ---
  for arm_name in ("tree", "control"):
    term_accum: dict = {}
    n_reps_with_terms = 0
    for rep in replicates:
      breakdown = rep[arm_name].get("distance_breakdown") or {}
      terms  = breakdown.get("terms",   {})
      weights = breakdown.get("weights", {})
      if not terms:
        continue
      n_reps_with_terms += 1
      for k, raw_term in terms.items():
        w = weights.get(k, 1.0)
        weighted = w * raw_term
        term_accum.setdefault(k, []).append(weighted)
    if term_accum:
      mean_terms = {k: float(np.mean(v)) for k, v in term_accum.items()}
      sorted_terms = dict(sorted(mean_terms.items(), key=lambda kv: kv[1], reverse=True))
      summary.setdefault("top_distance_terms_per_arm", {})[arm_name] = {
          "mean_weighted_contribution": sorted_terms,
          "n_replicates": n_reps_with_terms,
      }

  return summary


def run_mr_tree_experiment(
    config:                  str | dict,
    n_replicates:            int         = 20,
    n_states:                int | None  = None,
    tree_strength:           float       = 1.0,
    root_variance:           float       = 0.0,
    seed_base:               int         = 0,
    mr_rate_low:             float | None = None,
    mr_rate_high:            float | None = None,
    stats_n_pca_components:  int | None  = None,
    stats_n_structure_genes: int | None  = None,
    stats_percentiles:       tuple | None = None,
    stats_seed:              int | None  = None,
    grn_tmp_dir:             str | None  = None,
    grn_output_path:         str | None  = None,
    verbose:                 bool        = True,
) -> dict:
  """
  Paired experiment comparing sample_mr_state_from_tree's tree-correlated
  MR states (test arm, tree_strength=`tree_strength`) against its own
  tree_strength=0.0 control arm, holding every other simulation input
  fixed.

  For a *single*, fixed GRN (built once from `config` -- exactly one call
  to generate_sergio_grn_from_reference / build_mr_overlap_tree in this
  whole function, shared by every replicate and both arms): runs
  `n_replicates` paired (tree, control) make_synthetic_data6 simulations,
  each pair sharing the same per-replicate state/simulation seed (so the
  MR-state draw and every other simulation randomness source -- cluster
  sizing, SERGIO's own stochastic simulation, technical noise -- are as
  comparable as possible between the pair; the *only* difference between
  a pair's two arms is tree_strength itself, via sample_mr_state_from_
  tree's own derivation), computes compute_summary_stats() on each of the
  2 * n_replicates resulting matrices (missing_rate=0.0, matching how
  target_stats_path's own reference statistics are generated -- see
  compute_summary_stats' module-level usage note), and returns everything
  needed for a paired comparison (see _summarize_mr_tree_experiment).

  Args:
    config:          path to a tune_synthetic_data.py-style JSON config
                      (either a *tuning* config or a *tuned-result*
                      config, e.g. tuned_synthetic_config.20260820.00.json
                      -- see _load_mr_tree_experiment_config), or an
                      already-loaded/flat dict of the same shape. Supplies
                      every make_synthetic_data6/generate_sergio_grn_
                      from_reference/compute_summary_stats argument this
                      experiment needs; a tuned-result config does not
                      capture every such argument (see
                      _load_mr_tree_experiment_config) -- use the
                      mr_rate_low/mr_rate_high/stats_* keyword arguments
                      below, or pass the original tuning config instead,
                      to override its gaps explicitly.
    n_replicates:    number of paired (tree, control) datasets to
                      generate from the one fixed GRN/tree.
    n_states:        number of MR-state rows per make_synthetic_data6
                      call, i.e. its `n_clusters` -- defaults to
                      config["n_clusters"] (None means "use the config's
                      own value unchanged").
    tree_strength:   sample_mr_state_from_tree's tree_strength for the
                      test arm (the control arm always uses 0.0).
    root_variance:   sample_mr_state_from_tree's root_variance, shared by
                      both arms.
    seed_base:       replicate r's paired state/sim seed is
                      seed_base + r (r in range(n_replicates)) -- the
                      GRN's own grn_seed (from `config`) is untouched
                      (the GRN/tree stay fixed across every replicate).
    mr_rate_low, mr_rate_high: override config["mr_rate_low"]/
                      ["mr_rate_high"] (not captured by a tuned-result
                      config's "_meta" -- see _load_mr_tree_experiment_
                      config). None (default) uses the resolved config
                      value (tune_synthetic_data's plain defaults, 1.0/
                      5.0, if `config` doesn't set them either).
    stats_n_pca_components, stats_n_structure_genes, stats_percentiles,
    stats_seed:      override the corresponding config["stats_*"] entry
                      (also not captured by a tuned-result config's
                      "_meta") -- IMPORTANT: if target_stats_path is set,
                      these should match however that pickle's own
                      compute_summary_stats() call was configured (this
                      function warns, but does not raise, on a
                      stats_n_pca_components mismatch against the loaded
                      target's own PCA depth -- see the loud warning
                      below, mirroring tune_synthetic_data.run()'s
                      identical check).
    grn_tmp_dir:     directory for the one generated GRN CSV -- defaults
                      to config.get("grn_tmp_dir") or the system temp dir
                      (same convention as tune_synthetic_data.run_trial).
    grn_output_path: if given, the generated GRN CSV is written here and
                      NOT deleted afterward (e.g. to inspect/reuse it);
                      otherwise a temp file is used and removed at the
                      end of this function.
    verbose:         print a one-line progress message per replicate.

  Returns a dict:
    "config":            the normalized flat config actually used (after
                         applying every override argument above).
    "tree_strength", "root_variance", "n_states", "n_replicates": as given/resolved.
    "mr_ids":            list[int], the GRN's master-regulator gene ids
                         (== tree.mr_ids).
    "gene_id_to_symbol": dict[int, str], from generate_sergio_grn_from_reference.
    "tree":              the single MRTree built and reused for every
                         replicate.
    "mr_target_sets":    dict[mr_id, frozenset[int]] used to build the
                         tree (see build_mr_target_sets).
    "grn_diagnostics":   dict populated by generate_sergio_grn_from_
                         reference's own `diagnostics` parameter --
                         contains "tgt_ids", "winner_mr", "vote_margin",
                         "n_regs", "n_ambiguous", "n_ambiguous_flipped",
                         "n_aligned_edges", "propagated_strength",
                         "mr_load", "n_distinct_winners",
                         "top1_winner_share" (see that function's
                         docstring for full per-field semantics). Use
                         this to check whether canalization/balancing
                         produced sensible winner assignments, and
                         whether those assignments align with the
                         Jaccard-overlap-based tree structure.
    "tree_correlation_matrix": (n_mrs, n_mrs) ndarray, the theoretical
                         MR-MR correlation matrix implied by the tree
                         under the Brownian-motion model with the given
                         root_variance (see mr_tree_correlation_matrix).
                         Compare to each arm's per-replicate empirical
                         "mr_mr_correlation" to verify the sampler's
                         fidelity, and inspect directly to confirm that
                         MRs with more target-gene overlap have higher
                         implied correlation.
    "jaccard_similarity_offdiag": 1D ndarray of all upper-triangle
                         pairwise Jaccard similarities (= 1 - Jaccard
                         distance) among the n_mrs MR target sets --
                         the raw overlap distribution the tree is built
                         from. If this distribution is concentrated near
                         0 (all MRs have nearly disjoint target sets),
                         the tree will be nearly flat and tree_strength
                         will have little effect.
    "target_set_sizes":  list[int], direct target-gene count per MR
                         (parallel to mr_ids) -- quick check for whether
                         MRs actually regulate enough genes to produce
                         meaningful pairwise overlap.
    "target_stats":      compute_summary_stats()-style dict loaded from
                         config["target_stats_path"], if set and
                         loadable, else None.
    "replicates": list (len n_replicates) of per-replicate dicts, each:
        "seed": this replicate's paired state/sim seed.
        "tree" and "control": per-arm dicts, each containing:
            "mr_state":       (n_states, n_mrs) ndarray of sampled
                              basal production rates.
            "cluster_labels": (n_cells,) int64 ndarray from
                              make_synthetic_data6.
            "stats":          compute_summary_stats() dict (all families,
                              plus augmented derived keys from
                              _augment_candidate_stats).
            "distance":       float or None -- compute_stats_distance
                              vs. target_stats (None if target_stats
                              unavailable).
            "distance_breakdown": dict with "terms" (per-key raw
                              relative-squared-error), "skipped", and
                              "weights" -- the full breakdown discarded
                              by tune_synthetic_data.run_trial (kept
                              here for post-hoc diagnosis of which stats
                              actually drive the distance, independently
                              of the scalar total). None if target_stats
                              unavailable.
            "mr_mr_correlation": (n_mrs, n_mrs) ndarray, empirical MR-
                              MR correlation matrix from the sampled
                              mr_state rows. Compare to
                              "tree_correlation_matrix" (theoretical) to
                              verify fidelity; compare tree vs. control
                              arms to confirm tree_strength has the
                              intended effect at this n_states.
            "mr_state_participation_ratio": float, effective rank of
                              the mr_state matrix (participation ratio
                              of its mean-centered singular-value
                              spectrum). Near n_mrs = fully independent;
                              near 1 = one dominant shared direction.
            "within_module_mean_abs_corr": float, mean |Pearson corr|
                              among gene pairs in the same winner_mr
                              module (see _gene_module_correlations).
                              NaN if winner_mr unavailable.
            "across_module_mean_abs_corr": float, same for gene pairs
                              in different modules.
            "within_minus_across": float, difference of the two above.
    "summary": aggregate paired-difference statistics across replicates,
      see _summarize_mr_tree_experiment -- includes scalar metric
      mean/std/diff tables, per-PC PCA spectrum (mean/std per arm), and
      mean top distance terms per arm.

  Note: this function does not persist anything to disk itself (unlike
  tune_synthetic_data.run_trial's grn_archive_dir) -- the caller (e.g.
  this module's own CLI, see main()) is responsible for pickling/saving
  the returned dict if it should outlive the calling process. mr_state/
  cluster_labels/stats/correlation matrices are numpy arrays throughout,
  not JSON-ified (see tune_synthetic_data._jsonify_stats for a ready-
  made converter if a JSON-based archive is preferred).
  """
  _tsd = _import_tune_synthetic_data()
  cfg = _load_mr_tree_experiment_config(config)

  if mr_rate_low             is not None: cfg["mr_rate_low"]             = mr_rate_low
  if mr_rate_high            is not None: cfg["mr_rate_high"]            = mr_rate_high
  if stats_n_pca_components  is not None: cfg["stats_n_pca_components"]  = stats_n_pca_components
  if stats_n_structure_genes is not None: cfg["stats_n_structure_genes"] = stats_n_structure_genes
  if stats_percentiles       is not None: cfg["stats_percentiles"]       = stats_percentiles
  if stats_seed              is not None: cfg["stats_seed"]              = stats_seed

  n_states = cfg["n_clusters"] if n_states is None else n_states
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  target_stats = None
  if cfg.get("target_stats_path"):
    with open(cfg["target_stats_path"], "rb") as f:
      target_stats = pickle.load(f)
    _augment_candidate_stats(target_stats, _tsd)
    target_pca = target_stats.get("pca_explained_variance_ratio")
    if target_pca is not None and len(target_pca) and len(target_pca) != cfg["stats_n_pca_components"]:
      print(
          f"*** WARNING: target_stats_path's pca_explained_variance_ratio "
          f"has {len(target_pca)} components, but stats_n_pca_components="
          f"{cfg['stats_n_pca_components']} -- candidates will not be "
          f"apples-to-apples with the target's PCA depth. Pass an explicit "
          f"stats_n_pca_components={len(target_pca)} to run_mr_tree_"
          f"experiment (matching how target_stats_path was generated), or "
          f"supply the *original* tuning config (not just its tuned-result "
          f"output) if it sets stats_n_pca_components itself. ***"
      )

  sim_kwargs, grn_k_dist, hill_coeff_dist, unknown_mode_repressor_prob, grn_coherency_kwargs = \
      _tsd._split_resolved(cfg)

  grn_tmp_dir = grn_tmp_dir or cfg.get("grn_tmp_dir") or tempfile.gettempdir()
  Path(grn_tmp_dir).mkdir(parents=True, exist_ok=True)
  grn_path = grn_output_path or os.path.join(
      grn_tmp_dir, f"sergio_grn_mr_tree_experiment_{os.getpid()}.csv")

  if verbose:
    print(f"[run_mr_tree_experiment] generating GRN ({cfg['n_genes']} genes, "
          f"grn_seed={cfg['grn_seed']}) -> {grn_path!r} ...")
  grn_diagnostics: dict = {}
  _, mr_ids, gene_id_to_symbol = generate_sergio_grn_from_reference(
      reference_grn_path=cfg["reference_grn_path"],
      n_genes=cfg["n_genes"],
      output_path=grn_path,
      delimiter=cfg["grn_delimiter"],
      regulator_col=cfg["grn_regulator_col"],
      target_col=cfg["grn_target_col"],
      mode_col=cfg["grn_mode_col"],
      activation_labels=cfg["grn_activation_labels"],
      repression_labels=cfg["grn_repression_labels"],
      unknown_mode_repressor_prob=unknown_mode_repressor_prob,
      k_dist=grn_k_dist,
      hill_coeff_dist=hill_coeff_dist,
      max_seed_attempts=cfg["grn_max_seed_attempts"],
      seed=cfg["grn_seed"],
      diagnostics=grn_diagnostics,
      **grn_coherency_kwargs,
  )

  # winner_mr_by_gene: pair of parallel lists (tgt_ids, winner_mr) taken
  # directly from grn_diagnostics -- used by _gene_module_correlations.
  winner_mr_by_gene = (
      (grn_diagnostics.get("tgt_ids", []), grn_diagnostics.get("winner_mr", []))
      if grn_diagnostics.get("winner_mr") else None
  )

  try:
    mr_target_sets = build_mr_target_sets(grn_path, mr_ids)
    tree = build_mr_overlap_tree(grn_path, mr_ids, target_sets=mr_target_sets)

    # GRN-level instrumentation (computed once, shared across all replicates).
    tree_correlation_matrix = mr_tree_correlation_matrix(tree)
    R_offdiag = tree_correlation_matrix[np.triu_indices(tree.n_mrs, k=1)]
    target_set_sizes = [len(mr_target_sets[mr]) for mr in mr_ids]
    dist_matrix = _jaccard_distance_matrix([mr_target_sets[mr] for mr in mr_ids])
    jaccard_sim_offdiag = (1.0 - dist_matrix)[np.triu_indices(tree.n_mrs, k=1)]

    if verbose:
      print(f"[run_mr_tree_experiment] built MR overlap tree over "
            f"{tree.n_mrs} MRs | target-set sizes min/median/max = "
            f"{min(target_set_sizes)}/{int(np.median(target_set_sizes))}/{max(target_set_sizes)} | "
            f"Jaccard similarity off-diag mean={float(np.mean(jaccard_sim_offdiag)):.3f} | "
            f"R_tree off-diag mean={float(np.mean(R_offdiag)):.3f} ...")

    replicates = []
    for r in range(n_replicates):
      replicate_seed = seed_base + r
      if verbose:
        print(f"[run_mr_tree_experiment] replicate {r + 1}/{n_replicates} "
              f"(seed={replicate_seed}) ...")

      arms = {}
      for arm_name, strength in (("tree", tree_strength), ("control", 0.0)):
        mr_state_np = sample_mr_state_from_tree(
            tree, n_states=n_states,
            low=cfg["mr_rate_low"], high=cfg["mr_rate_high"],
            seed=replicate_seed, tree_strength=strength, root_variance=root_variance,
        )
        mr_state = torch.tensor(mr_state_np, dtype=torch.float32, device=device)

        X_sim, cluster_labels = make_synthetic_data6(
            mr_state=mr_state,
            input_file_targets=grn_path,
            n_cells=cfg["n_cells"],
            mr_gene_ids=mr_ids,
            shared_coop_state=cfg["shared_coop_state"],
            noise_type=cfg["noise_type"],
            sampling_state=cfg["sampling_state"],
            dt=cfg["dt"],
            min_cells_per_cluster=cfg["min_cells_per_cluster"],
            add_outlier_genes=cfg["add_outlier_genes"],
            add_lib_size_effect=cfg["add_lib_size_effect"],
            add_dropout=cfg["add_dropout"],
            convert_to_umi_counts=cfg["convert_to_umi_counts"],
            missing_rate=0.0,
            seed=replicate_seed,
            device=device,
            **sim_kwargs,
        )
        stats = compute_summary_stats(
            X_sim,
            n_pca_components=cfg["stats_n_pca_components"],
            n_structure_genes=cfg["stats_n_structure_genes"],
            percentiles=tuple(cfg["stats_percentiles"]),
            seed=cfg["stats_seed"],
        )
        _augment_candidate_stats(stats, _tsd)

        distance = None
        distance_breakdown = None
        if target_stats is not None:
          distance, breakdown = _tsd.compute_stats_distance(
              target_stats, stats, weights=cfg.get("weights"),
              eps=cfg.get("distance_eps", 1e-6),
              eps_frac=cfg.get("distance_eps_frac", 0.05),
              eps_abs_floor=cfg.get("distance_eps_abs_floor", 0.02),
          )
          # Store terms + the weights used so _summarize can re-rank them.
          distance_breakdown = {
              "terms":   breakdown["terms"],
              "skipped": breakdown["skipped"],
              "weights": dict(cfg.get("weights") or {}),
          }

        # Per-arm instrumentation.
        mr_state_corr = (
            np.corrcoef(mr_state_np, rowvar=True)  # (n_states, n_states) -- rows as obs
            if mr_state_np.shape[0] >= 2 else np.full((1, 1), float("nan"))
        )
        # MR-MR empirical correlation (columns as variables).
        mr_mr_corr = (
            np.corrcoef(mr_state_np, rowvar=False)  # (n_mrs, n_mrs)
            if mr_state_np.shape[0] >= 2 else np.full((mr_state_np.shape[1],) * 2, float("nan"))
        )
        mr_state_pr = _mr_state_participation_ratio(mr_state_np)

        gene_mod_corr = _gene_module_correlations(
            X_sim, cluster_labels, winner_mr_by_gene, mr_ids, gene_id_to_symbol,
            n_structure_genes=cfg["stats_n_structure_genes"],
            seed=cfg["stats_seed"],
        )

        arms[arm_name] = {
            "mr_state":                    mr_state_np,
            "cluster_labels":              cluster_labels,
            "stats":                       stats,
            "distance":                    distance,
            "distance_breakdown":          distance_breakdown,
            "mr_mr_correlation":           mr_mr_corr,
            "mr_state_participation_ratio": mr_state_pr,
            "within_module_mean_abs_corr": gene_mod_corr["within_module_mean_abs_corr"],
            "across_module_mean_abs_corr": gene_mod_corr["across_module_mean_abs_corr"],
            "within_minus_across":         gene_mod_corr["within_minus_across"],
        }

      replicates.append({"seed": replicate_seed, "tree": arms["tree"], "control": arms["control"]})
  finally:
    if grn_output_path is None and os.path.exists(grn_path):
      os.remove(grn_path)

  summary = _summarize_mr_tree_experiment(replicates)

  return {
      "config":            cfg,
      "tree_strength":     tree_strength,
      "root_variance":     root_variance,
      "n_states":          n_states,
      "n_replicates":      n_replicates,
      "mr_ids":            mr_ids,
      "gene_id_to_symbol": gene_id_to_symbol,
      "tree":              tree,
      "mr_target_sets":    mr_target_sets,
      "grn_diagnostics":   grn_diagnostics,
      "tree_correlation_matrix":        tree_correlation_matrix,
      "jaccard_similarity_offdiag":     jaccard_sim_offdiag,
      "target_set_sizes":               target_set_sizes,
      "target_stats":      target_stats,
      "replicates":        replicates,
      "summary":           summary,
  }


def main(argv=None) -> dict:
  parser = argparse.ArgumentParser(
      description=(
          "Compare tree-correlated vs. i.i.d. master-regulator mr_state "
          "sampling (build_mr_overlap_tree / sample_mr_state_from_tree) "
          "via run_mr_tree_experiment."
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--config", required=True,
      help="Path to a tune_synthetic_data.py-style JSON config (tuning or "
           "tuned-result shape -- see _load_mr_tree_experiment_config).",
  )
  parser.add_argument("--n-replicates", type=int, default=20)
  parser.add_argument("--n-states", type=int, default=None)
  parser.add_argument("--tree-strength", type=float, default=1.0)
  parser.add_argument("--root-variance", type=float, default=0.0)
  parser.add_argument("--seed-base", type=int, default=0)
  parser.add_argument("--mr-rate-low", type=float, default=None)
  parser.add_argument("--mr-rate-high", type=float, default=None)
  parser.add_argument("--stats-n-pca-components", type=int, default=None)
  parser.add_argument("--grn-output-path", default=None)
  parser.add_argument(
      "--output", default="mr_tree_experiment_result.pickle",
      help="Where to pickle the full result dict (see run_mr_tree_"
           "experiment's return-value docstring).",
  )
  args = parser.parse_args(argv)

  result = run_mr_tree_experiment(
      args.config,
      n_replicates=args.n_replicates,
      n_states=args.n_states,
      tree_strength=args.tree_strength,
      root_variance=args.root_variance,
      seed_base=args.seed_base,
      mr_rate_low=args.mr_rate_low,
      mr_rate_high=args.mr_rate_high,
      stats_n_pca_components=args.stats_n_pca_components,
      grn_output_path=args.grn_output_path,
  )

  with open(args.output, "wb") as f:
    pickle.dump(result, f)
  print(f"\nSaved full result to {args.output!r}")

  print(f"\nPaired tree-vs-control summary (tree_strength={args.tree_strength}, "
        f"n_replicates={args.n_replicates}):")
  for name, s in result["summary"].items():
    print(f"  {name}: tree={s['tree_mean']:.4f}  control={s['control_mean']:.4f}  "
          f"diff={s['diff_mean']:.4f} +/- {s['diff_std']:.4f}  (n={s['n']})")

  return result


if __name__ == "__main__":
  main()

