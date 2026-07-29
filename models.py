from abc import ABC, abstractmethod  # re-evaluate this
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from synthetic_data import masked_loss, get_random_mask, _random_fill

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _block(in_dim:int, out_dim:int, dropout:float) -> nn.Sequential:
  return nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.BatchNorm1d(out_dim),
      nn.ReLU(),
      nn.Dropout(dropout)
  )

def build_encoder(n_genes:int, encoder_dims:list[int], dropout:float) -> tuple[nn.Sequential, int]:
    """Returns encoder backbone and output dimension."""
    layers = []
    prev = n_genes * 2
    for h in encoder_dims:
        layers.append(_block(prev, h, dropout))
        prev = h
    return nn.Sequential(*layers)


def build_decoder(n_genes:int, latent_dim:int, decoder_dims:list[int],  dropout:float) -> nn.Sequential:
    """
    Returns decoder network.
    """
    # TODO use _block ?
    layers = []
    prev = latent_dim
    for h in decoder_dims:
        layers.extend([
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        prev = h
    layers.append(nn.Linear(prev, n_genes))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class VAEBase(nn.Module, ABC): #nn.Module
  """Shared encoder backbone for both single and mixture VAEs."""
  def __init__(self,
                n_genes:int,
                latent_dim:int,
                hidden_dims:  list[int]|None = None,
                encoder_dims: list[int]|None = None,
                decoder_dims: list[int]|None = None,
                dropout:float = 0.05):

    super().__init__()

    # hidden_dims | encoder_dims | decoder_dims | Note
    #-------------+--------------+--------------+-------------------------------
    #      +      |      -       |      -       | valid (hidden used for both)
    #      -      |      +       |      +       | valid
    #      -      |      -       |      -       | valid (default used)
    if encoder_dims is not None: assert hidden_dims is None
    if decoder_dims is not None: assert hidden_dims is None
    if hidden_dims is not None:
      assert encoder_dims is None and decoder_dims is None
      encoder_dims = hidden_dims
      decoder_dims = list(reversed(hidden_dims))
    else:
      if encoder_dims is None and decoder_dims is None:
        encoder_dims = [124, 64]
        decoder_dims = [64, 124]

    self.latent_dim   = latent_dim
    self.encoder_dims = encoder_dims
    self.decoder_dims = decoder_dims
    self.enc_out_dim  = encoder_dims[-1]  # encoder output dim (!= latent dim)
    self.n_genes      = n_genes           # input dimension (input features)
    self.dropout      = dropout

    # descendants will initialize model elements


  def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
    if self.training:
      std = torch.exp(0.5 * logvar)
      return mu + torch.randn_like(std) * std
    else:
      return mu

  #def get_encoder_features(self, x:torch.Tensor) -> torch.Tensor:
  #    """Get encoder hidden representation before mu/logvar."""
  #    return self.encoder(x)

class GeneExpressionVAE(VAEBase):
  """
  Standard VAE for gene expression imputation.
  """
  def __init__(self,
                n_genes:int,
                latent_dim:int = 8,
                hidden_dims:  list[int]|None = None,
                encoder_dims: list[int]|None = None,
                decoder_dims: list[int]|None = None,
                dropout:float = 0.05):

    super().__init__(n_genes,
                      latent_dim,
                      hidden_dims,
                      encoder_dims,
                      decoder_dims,
                      dropout)

    # Build encoder
    self.encoder = build_encoder(n_genes, encoder_dims, dropout)

    # Build latent
    self.fc_mu     = nn.Linear(self.enc_out_dim, latent_dim)
    self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

    # Build decoder
    self.decoder = build_decoder(n_genes, latent_dim, decoder_dims, dropout)


  def encode(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = self.encoder(x)
    return h, self.fc_mu(h), self.fc_logvar(h)

  def decode(self, z:torch.Tensor) -> torch.Tensor:
    return self.decoder(z)

  def forward(self, x:torch.Tensor):
    _, mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z)
    return recon, mu, logvar, None, None, None


def _fixed_point_iterate(f, z0:torch.Tensor, max_iter:int=30, tol:float=1e-4):
  """Repeated substitution: iterate z <- f(z) until convergence.

  Minimal local copy of the same helper used by DEQBlock in
  normalizing_flow_demo.py (not imported from there because that module
  runs a full training demo as a side effect of import). Valid whenever f
  is a contraction (Lipschitz constant < 1), which DEQCell guarantees by
  construction (spectral-normalized weight * coeff < 1, composed with a
  1-Lipschitz tanh).

  Returns (z, n_iter, final_rel_change, final_rel_change_per_sample).

  Instrumentation note: the stopping decision is made from a single
  *whole-batch* scalar (`rel_change`, computed from the aggregate norm over
  all rows) -- this matches the original behaviour exactly, so training
  dynamics/results are unaffected. The extra `rel_change_per_sample` return
  value (shape (B,), one scalar per row of z0) is a diagnostic computed
  alongside it at no extra cost (same z/z_next already in hand), added so
  callers can tell whether "converged" (by the aggregate check) actually
  means every row converged, or just that most rows did while a few
  particular rows (e.g. ones from an under-represented class) were still
  changing a lot when the loop stopped. A single aggregate scalar cannot
  distinguish those two cases; rel_change_per_sample can.
  """
  z = z0
  n_iter = max_iter
  rel_change = float("nan")
  rel_change_per_sample = torch.full(
      (z0.shape[0],), float("nan"), device=z0.device, dtype=z0.dtype)
  for i in range(max_iter):
    z_next = f(z)
    diff = z_next - z
    rel_change = (diff.norm() / (z.norm() + 1e-6)).item()
    rel_change_per_sample = diff.flatten(1).norm(dim=1) / (z.flatten(1).norm(dim=1) + 1e-6)
    z = z_next
    if rel_change < tol:
      n_iter = i + 1
      break
  return z, n_iter, rel_change, rel_change_per_sample


class DEQCell(nn.Module):
  """
  Weight-tied implicit-depth recurrence: z* = coeff * tanh(W_zz @ z* + c),
  solved to its fixed point for a given conditioning vector c.

  This is the "DEQ as encoder" building block: instead of a fixed stack of
  N distinct MLP layers, a single shared transition is applied until it
  converges to self-consistency with the conditioning input c (derived
  from the observed/masked gene expression). This gives:
    - parameter efficiency: one shared (dim x dim) weight instead of N
      separate hidden-layer weights, relevant for the p >> N regime.
    - variable test-time compute: harder/more-incomplete rows can be run
      for more iterations without retraining or changing model capacity.

  Unlike DEQBlock in normalizing_flow_demo.py, this cell does not need to
  be invertible and never computes a Jacobian log-determinant (it isn't
  part of a bijective flow) -- it is only ever used as an encoder, which
  makes it considerably cheaper than DEQFlow's DEQBlock.

  Forward/backward follow the standard DEQ pattern (Bai, Kolter & Koltun,
  2019): solve for z* under no_grad, re-apply the transition once more
  with grad enabled to attach a shallow graph, then replace the
  backward-pass gradient with the solution of the adjoint fixed-point
  equation via a backward hook -- avoiding backprop through the unrolled
  solver entirely.

  Diagnostics (populated on every forward() call, read-only, for inspection):
    last_forward_iters / last_forward_residual   -- whole-batch (aggregate)
        iteration count and residual for the forward solve, as before.
    last_forward_residual_per_sample : (B,) tensor -- per-row residual at
        the same final iteration used for the aggregate check above (see
        _fixed_point_iterate's docstring). Lets a caller check whether
        particular rows (e.g. cells from a specific class) were still far
        from converged even when the aggregate check passed.
    last_z_star : (B, dim) tensor, detached -- the converged fixed point
        itself, *before* any downstream Linear projection (e.g. fc_mu).
        Since z* = coeff*tanh(...), every component is confined to
        (-coeff, coeff); inspecting last_z_star's histogram/std is how to
        check whether the recurrence is saturating (most mass near
        +/-coeff) for some inputs, which would compress the gradient signal
        available to whatever consumes z* downstream.
    last_backward_iters / last_backward_residual -- same as above, but for
        the adjoint (backward-pass) fixed-point solve.
  """

  def __init__(self, dim:int, coeff:float=0.9, max_iter:int=30, tol:float=1e-4,
               n_power_iterations:int=5):
    super().__init__()
    self.coeff = coeff
    self.W_zz = nn.utils.spectral_norm(
        nn.Linear(dim, dim, bias=False), n_power_iterations=n_power_iterations)
    self.max_iter = max_iter
    self.tol = tol

    # Diagnostics populated on every forward() call, for inspection only.
    self.last_forward_iters:                int|None = None
    self.last_forward_residual:            float|None = None
    self.last_forward_residual_per_sample: torch.Tensor|None = None
    self.last_z_star:                      torch.Tensor|None = None
    self.last_backward_iters:               int|None = None
    self.last_backward_residual:          float|None = None

  def _f(self, z:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
    """
    Note: Lipshitz <= self.coeff < 1
    This guarantees z*=f(z*,c) has a unique solution (Banach fixed-point theorem)
    """
    return self.coeff * torch.tanh(self.W_zz(z) + c)

  def forward(self, c:torch.Tensor) -> torch.Tensor:
    """c: (B, dim) conditioning vector -> z*: (B, dim) fixed point."""
    z0 = torch.zeros_like(c)
    with torch.no_grad():
      z_star, n_iter, residual, residual_per_sample = _fixed_point_iterate(
          lambda z: self._f(z, c), z0, self.max_iter, self.tol)
    self.last_forward_iters = n_iter
    self.last_forward_residual = residual
    self.last_forward_residual_per_sample = residual_per_sample.detach()
    self.last_z_star = z_star.detach()

    z = self._f(z_star, c)  # value ~= z_star, but now differentiable

    if z.requires_grad:
      z0d = z.detach().requires_grad_()
      f0 = self._f(z0d, c)

      def backward_hook(grad):
        v, n_iter_bwd, residual_bwd, _residual_bwd_per_sample = _fixed_point_iterate(
            lambda v: torch.autograd.grad(f0, z0d, grad_outputs=v, retain_graph=True)[0] + grad,
            grad, self.max_iter, self.tol,
        )
        self.last_backward_iters = n_iter_bwd
        self.last_backward_residual = residual_bwd
        return v

      z.register_hook(backward_hook)

    return z


class DEQEncoderVAE(VAEBase):
  """
  VAE with an implicit-depth (DEQ) encoder.

  Architecture:
    x_concat = [x_masked ; mask]            -- identical input every other
                                                model in this file receives
    h = build_encoder(x_concat)             -- shared MLP feature extractor
    c = Linear(h)                           -- conditioning vector (latent_dim)
    z* = DEQCell(z -> coeff*tanh(W_zz@z + c))  -- weight-tied fixed point
    mu, logvar = Linear(z*), Linear(z*)
    z  = reparameterize(mu, logvar)
    recon = build_decoder(z)

  The prior is the standard N(0, I) and the decoder is the standard MLP
  decoder -- only the encoder's recognition network is replaced by a
  weight-tied implicit-depth recurrence instead of a fixed stack of MLP
  layers. This isolates the "DEQ as amortized encoder" role for direct,
  single-variable comparison against GeneExpressionVAE at matched
  encoder/decoder capacity.

  Note: like every other model in this file, the encoder receives
  randomly-filled values for missing entries (per-gene N(mean, std) noise,
  see get_random_mask / epoch_vae) plus a mask channel -- this class does
  not by itself remove that preprocessing step. What it adds is a
  weight-tied, variable-depth refinement of the latent given that input,
  with convergence diagnostics available via self.deq_cell.last_forward_iters
  / last_forward_residual (whole-batch), or
  self.deq_cell.last_forward_residual_per_sample / last_z_star (per-cell --
  see DEQCell's docstring) for e.g. correlating convergence quality with
  per-class imputation error. Unlike DEQEncoderVampVAE, deq_cell here is
  only ever called once per forward pass, so these attributes always
  reflect the most recent real batch with no clobbering concern.
  """

  def __init__(self,
               n_genes:int,
               latent_dim:int = 8,
               hidden_dims:  list[int]|None = None,
               encoder_dims: list[int]|None = None,
               decoder_dims: list[int]|None = None,
               dropout:float = 0.05,
               coeff:float = 0.9,
               max_iter:int = 30,
               tol:float = 1e-4):

    super().__init__(n_genes,
                      latent_dim,
                      hidden_dims,
                      encoder_dims,
                      decoder_dims,
                      dropout)

    # Feature extractor (same shape convention as build_encoder elsewhere)
    self.encoder = build_encoder(n_genes, encoder_dims, dropout)
    self.fc_cond = nn.Linear(self.enc_out_dim, latent_dim)

    # Weight-tied implicit-depth recurrence
    self.deq_cell = DEQCell(latent_dim, coeff=coeff, max_iter=max_iter, tol=tol)

    # Latent heads (applied to the DEQ fixed point, not to h directly)
    self.fc_mu     = nn.Linear(latent_dim, latent_dim)
    self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    # Build decoder
    self.decoder = build_decoder(n_genes, latent_dim, decoder_dims, dropout)

  def encode(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = self.encoder(x)
    c = self.fc_cond(h)
    z_star = self.deq_cell(c)
    return h, self.fc_mu(z_star), self.fc_logvar(z_star)

  def decode(self, z:torch.Tensor) -> torch.Tensor:
    return self.decoder(z)

  def forward(self, x:torch.Tensor):
    _, mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z)
    return recon, mu, logvar, None, None, None


class MoVEVAE(VAEBase):
  """
  Mixture of Variational Encoders using GeneExpressionVAE decoders.

  Architecture:
    - Shared encoder backbone (from VAEBase)
    - Per-component μ/logvar heads (component-specific recognition)
    - Per-component decoders (component-specific generative model)
    - Gating network for component selection
  """
  def __init__(self,
               n_genes:int,
               latent_dim:int=8,
               n_components:int=3,
               gating_net_dim:int=32,
               hidden_dims:  list[int]|None = None,
               encoder_dims: list[int]|None = None,
               decoder_dims: list[int]|None = None,
               dropout:float = 0.05):

    super().__init__(n_genes, latent_dim, hidden_dims, encoder_dims, decoder_dims, dropout)

    self.n_components = n_components

    #self.component_encoders = nn.ModuleList([
    #    nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
    #])
    #self.component_encoders_logvar = nn.ModuleList([
    #    nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
    #])

    # Build encoder
    self.encoder = build_encoder(n_genes, encoder_dims, dropout)

    # Build latent
    self.fc_mu     = nn.ModuleList([
        nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
    ])
    self.fc_logvar = nn.ModuleList([
        nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
    ])

    # Build gating network
    self.gating = nn.Sequential(
        nn.Linear(self.enc_out_dim, gating_net_dim),
        nn.ReLU(),
        nn.Linear(gating_net_dim,    n_components)
    )

    # Build decoders
    self.decoders = nn.ModuleList([
        build_decoder(n_genes, latent_dim, decoder_dims, dropout)
        for _ in range(n_components)
    ])


  #def _encode_single_ecomponent(self, h:torch.Tensor, component:int) -> tuple[torch.Tensor, torch.Tensor]:
  #  """Encode for a specific component."""
  #  return self.component_encoders[component](h), self.component_encoders_logvar[component](h)

  def encode(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = self.encoder(x)
    #return h, self.fc_mu(h), self.fc_logvar(h)
    return (h,
      torch.stack( [mu(h)     for mu     in self.fc_mu    ] ),
      torch.stack( [logvar(h) for logvar in self.fc_logvar] ))

  def decode(self, z:torch.Tensor, component:int) -> torch.Tensor:
    """Decode using specific component."""
    return self.decoders[component](z)

  def forward(self, x:torch.Tensor):
    """
    Returns: (recons_all_components, mu, logvar, gate_probs, soft_weights, selected_component)
    """
    h, mu, logvar = self.encode(x)

    gate_logits = self.gating(h)
    gate_probs = F.softmax(gate_logits, dim=-1)

    if self.training:
        # Gumbel-Softmax: straight-through estimator
        gumbel = -torch.log(-torch.log(torch.rand_like(gate_probs) + 1e-20) + 1e-20)
        gate_logits_st = gate_logits + gumbel
        soft_weights = F.softmax(gate_logits_st, dim=-1)  # (B, K)
        k_indices = soft_weights.argmax(dim=-1)  # for logging
        hard = F.one_hot( k_indices, self.n_components).float()
        soft_weights = hard - soft_weights.detach() + soft_weights
    else:
        k_indices = gate_probs.argmax(dim=-1)
        soft_weights = gate_probs


    #z = self.reparameterize(mu, logvar)

    #recons_all = torch.stack([self.decoders[k](z) for k in range(self.n_components)])
    recons_all = torch.stack([
        self.decoders[k]( self.reparameterize(mu[k], logvar[k]))
        for k in range(self.n_components)
        ]).permute(1, 0, 2)  # (B,K,G)
    batch_indices = torch.arange(x.shape[0], device=x.device)
    #print(f'recons_all:{recons_all.shape} batch_indices:{batch_indices.shape}')
    #recons = recons_all[k_indices, batch_indices]
    recons = (recons_all * soft_weights.unsqueeze(-1)).sum(dim=1)  # marginalize over k decoders

    return recons, mu, logvar, gate_probs, soft_weights, k_indices


def gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Element-wise log N(z; mu, exp(logvar)).
    Supports broadcasting: z/mu/logvar can be (..., D).
    Returns a tensor of the same shape as inputs (before any .sum()).
    """
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp())

class VampPriorVAE(VAEBase):
  """
  VAE with Variational Mixture of Posteriors (VampPrior).

  The prior p(z) = (1/K) sum_k q_phi(z | u_k) is a mixture of K encoder
  posteriors evaluated at K learnable pseudo-inputs u_k (in gene-expression
  space).  The KL divergence KL(q(z|x) || p(z)) is estimated with a single
  Monte-Carlo sample z ~ q(z|x) using log-sum-exp over the K components.

  Architecture is identical to GeneExpressionVAE (single encoder head, single
  decoder); all expressiveness comes from the flexible prior.

  forward() returns the same 6-tuple as MoVEVAE so it is a drop-in replacement
  in epoch_vae.  The VampPrior KL scalar is stored in self._vamp_kl after each
  forward pass for use by the training loop.

  Anti-collapse regularizers (both optional, both consumed by epoch_vae if
  present -- see there for the loss-composition side):
    self._vamp_diversity_loss : population-level responsibility-entropy
        term (same sign/scale convention as MoVEVAE's gating_entropy_loss;
        weighted by the existing lambda_entropy hyperparameter). Encourages
        the K pseudo-inputs to *collectively* share responsibility for the
        batch, rather than one pseudo-input absorbing almost all of it.
        Blind spot: satisfiable by K identical/redundant components (a
        mixture of clones has perfectly uniform, but meaningless,
        responsibility) -- see _vamp_repulsion_loss below for why both
        regularizers are needed together.
    self._vamp_repulsion_loss : pairwise hinge repulsion on the pseudo-inputs'
        *encoded* posterior means (pu_mu), with an adaptive margin (see
        vampprior_kl) rather than a fixed guessed constant. Encourages the K
        components to be geometrically distinct in latent space, which
        _vamp_diversity_loss alone cannot guarantee.

        Caveat learned from a real training run: this alone was NOT
        sufficient to prevent effective_K collapse. Mutual repulsion only
        pushes pseudo-inputs apart from EACH OTHER -- it converged nicely
        (pu_mu pairwise distance comfortably exceeded the margin) while
        effective_K still collapsed to ~1, because nothing pulled the
        pseudo-inputs toward the region where real data actually lives. In
        a high-dimensional ambient latent space, mutually-repelled points
        can scatter away from a comparatively low-dimensional data manifold
        entirely, leaving only ~1 of K ever close enough to compete for
        responsibility. See _vamp_coverage_loss below, which is what
        actually addresses this.
    self._vamp_coverage_loss : Chamfer-style attraction term -- pulls each
        pseudo-input toward its *nearest* real-batch encoded mu, so
        pseudo-inputs individually land near the data manifold instead of
        merely being spread apart from each other in the ambient space.
        Combined with _vamp_repulsion_loss this gives the standard
        "attract to data, repel from siblings" pattern (as used in
        prototype/coverage learning): pulled onto the manifold
        individually, but can't pile onto the same point because repulsion
        still pushes them apart.
  None of these regularizers require this model's KL to have been computed
  via kl_override in any special way -- they are independent extra loss
  terms.

  ***effective_K vs effective_K_population -- READ THIS FIRST***
  self._effective_K (and self._effective_K_per_sample) is exp(mean_i[entropy
  of cell i's own responsibility distribution over the K components]).
  self._effective_K_population is exp(entropy(avg_resp)), i.e. entropy
  computed AFTER averaging responsibility over the batch. These are NOT
  interchangeable, and the difference is not a rounding nuance:

    Because entropy is concave, Jensen's inequality gives
    exp(mean_i[entropy_i]) <= exp(entropy(mean_i)) ALWAYS -- and in the
    THEORETICAL BEST CASE (every cell confidently/one-hot assigned to its
    own distinct pseudo-input, perfectly spread across all K -- i.e.
    maximally diverse population usage with zero redundancy) --
    self._effective_K reads EXACTLY 1.0 while self._effective_K_population
    reads K. This isn't a hypothetical edge case either: it's the expected
    *routine* regime for a well-separated Gaussian mixture in a
    high-dimensional latent space (latent_dim=32 here) -- individual
    per-cell assignment naturally sharpens toward one-hot long before the
    population's aggregate usage becomes non-diverse.

    Across this debugging session, self._effective_K stayed pinned near
    1.0-1.14 through three different regularizer designs that each targeted
    a different, confirmed-working geometric mechanism (population entropy,
    pairwise repulsion, data-manifold coverage) -- a strong signal in
    hindsight that the metric itself has a structural floor near 1, largely
    independent of whether the population-level mixture usage is actually
    healthy. self._effective_K_population is the metric that actually
    answers "does the population, in aggregate, spread its responsibility
    across many of the K components" -- i.e. the thing all three
    regularizers are actually trying to fix. Check *that* number before
    concluding collapse is still occurring.

  Diagnostics for tracking whether the regularizers are actually working
  (see collect_vamp_diagnostics, and vampprior_kl's inline comments for what
  each one distinguishes -- in particular, whether the adaptive margin is
  itself shrinking over training, a self-referential failure mode where the
  repulsion threshold retreats in step with collapse instead of resisting
  it; and separately, whether pseudo-inputs are actually landing near real
  data, which repulsion alone cannot guarantee):
    self._effective_K / self._effective_K_per_sample : responsibility-based
        mixture usage, per-sample-averaged -- see note above, near 1.0 is
        NORMAL, not necessarily collapse.
    self._effective_K_population : responsibility-based mixture usage,
        computed on the batch-averaged responsibility distribution -- see
        note above; THIS is the metric that indicates actual collapse.
    self._vamp_repulsion_typical_scale : real batch's own mu pairwise
        spread (the shared reference scale for both the repulsion margin
        and the coverage loss's normalization).
    self._vamp_repulsion_margin : the derived repulsion threshold.
    self._vamp_pu_mu_pairwise_median : what the pseudo-inputs actually
        achieve post-encoding -- compare against the margin above.
    self._vamp_pseudo_inputs_raw_pairwise_median : pre-encoding, raw
        gene-expression-space spread of the pseudo_inputs parameters
        themselves -- separates "did gradient descent spread the raw
        parameters apart" from "does the shared encoder collapse them
        anyway regardless of raw spread".
    self._vamp_coverage_mean_nearest_dist : mean (over K pseudo-inputs) of
        each one's distance to its nearest real-batch cell, in raw latent
        units -- should shrink toward 0 as coverage improves.
    self._vamp_resp : full (B, K) per-sample responsibility matrix,
        detached. On its own this is just the raw ingredient behind
        effective_K/effective_K_population above; its real purpose is to be
        joined against external per-cell class/cell-type labels (see
        vamp_usage_by_class) to test a distinct question from all of the
        above: not "is the mixture used diversely overall" but "is that
        diversity aligned with class identity at all", i.e. do different
        classes actually get routed to different pseudo-inputs, or does
        every class independently spread across ~the same broad set of
        components.

        Confirmed via a real run (8 classes): strongly class-aligned --
        per-class effective_K_population ~4.5-9.2 vs. a global value of
        ~40, and pairwise JS divergence between classes' avg_resp mostly at
        the theoretical max (ln 2). So different classes DO get routed to
        largely disjoint pseudo-inputs -- yet per-class reconstruction
        error variance was observed to be unchanged vs. a standard N(0,I)
        prior at matched latent_dim and non-negligible beta (beta*kl_loss
        ~7.9% of recon_loss at convergence). self._vamp_kl_per_sample below
        is the follow-up diagnostic this motivated.
    self._vamp_kl_per_sample : (B,) per-cell KL cost (log_q - log_p),
        detached -- see standard_gaussian_kl_per_sample for the equivalent
        quantity under a standard N(0,I) prior. Class-routing being
        class-aligned (above) is a RATE-ALLOCATION result: it says a class
        can occupy its own region of latent space cheaply. It does NOT by
        itself imply reconstruction FIDELITY becomes more class-fair too --
        fidelity is governed by posterior precision (logvar) and decoder
        capacity, which a flexible prior doesn't directly touch. Group this
        per-cell value with class_grouped_stats and compare its
        between_class_variance against the same computation on a
        standard-prior model's KL to test that distinction directly: if
        VampPrior's KL-cost variance across classes is lower than the
        standard prior's, but reconstruction-error variance
        (per_cell_masked_error) is the same for both, that confirms rate
        allocation and fidelity are simply different things the prior
        affects differently -- explaining the original null result without
        needing to invoke beta or capacity issues at all.
  """

  def __init__(self,
               n_genes:      int,
               latent_dim:   int = 8,
               n_pseudo:     int = 20,
               hidden_dims:  list[int]|None = None,
               encoder_dims: list[int]|None = None,
               decoder_dims: list[int]|None = None,
               dropout:      float = 0.05,
               pseudo_init_samples: torch.Tensor|None = None,
               repulsion_margin_frac: float = 0.5):

    super().__init__(n_genes, latent_dim, hidden_dims, encoder_dims, decoder_dims, dropout)

    self.n_pseudo = n_pseudo
    # Fraction of the *real batch's* typical pairwise mu-mu distance used as
    # the repulsion margin (see vampprior_kl) -- scale-free, so it doesn't
    # need re-tuning per latent_dim/encoder architecture/dataset the way a
    # fixed absolute margin would.
    self.repulsion_margin_frac = repulsion_margin_frac

    # Encoder (single head — same as GeneExpressionVAE)
    self.encoder   = build_encoder(n_genes, self.encoder_dims, dropout)
    self.fc_mu     = nn.Linear(self.enc_out_dim, latent_dim)
    self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

    # Decoder
    self.decoder = build_decoder(n_genes, latent_dim, self.decoder_dims, dropout)

    # Learnable pseudo-inputs in gene-expression space (K, n_genes).
    # A zero mask (all-observed) is appended before encoding so the
    # encoder sees a valid augmented input.
    if pseudo_init_samples is None:
      self.pseudo_inputs = nn.Parameter( torch.randn(n_pseudo, n_genes) * 0.01 )
    else:
      sample = torch.randint( 0, pseudo_init_samples.shape[0], size=(n_pseudo,) )
      self.pseudo_inputs = nn.Parameter( pseudo_init_samples[sample] )
    assert(self.pseudo_inputs.data.shape==(n_pseudo, n_genes))

    # Populated by forward()/vampprior_kl(); read by epoch_vae.
    self._vamp_kl:             torch.Tensor|None = None
    self._vamp_diversity_loss: torch.Tensor|None = None
    self._vamp_repulsion_loss: torch.Tensor|None = None
    self._vamp_coverage_loss:  torch.Tensor|None = None
    self._effective_K:              float|None = None
    self._effective_K_per_sample: torch.Tensor|None = None
    self._effective_K_population:   float|None = None
    self._vamp_resp:              torch.Tensor|None = None  # (B, K), see vamp_usage_by_class
    self._vamp_kl_per_sample:     torch.Tensor|None = None  # (B,), see standard_gaussian_kl_per_sample
    # Diagnostics only (see vampprior_kl / collect_vamp_diagnostics) -- track
    # whether the repulsion regularizer's adaptive margin is itself
    # collapsing, whether collapse (if any) originates in the raw
    # pseudo_inputs parameters or is introduced by the shared encoder, and
    # whether pseudo-inputs are actually landing near real data.
    self._vamp_repulsion_typical_scale:           torch.Tensor|None = None
    self._vamp_repulsion_margin:                  torch.Tensor|None = None
    self._vamp_pu_mu_pairwise_median:             torch.Tensor|None = None
    self._vamp_pseudo_inputs_raw_pairwise_median: torch.Tensor|None = None
    self._vamp_coverage_mean_nearest_dist:        torch.Tensor|None = None

  def _pseudo_encoded(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all K pseudo-inputs; return (pu_mu, pu_logvar) each (K, D)."""
    zero_mask  = torch.zeros(self.n_pseudo, self.n_genes, device=self.pseudo_inputs.device)
    pseudo_aug = torch.cat([self.pseudo_inputs, zero_mask], dim=1)  # (K, n_genes*2)
    h = self.encoder(pseudo_aug)
    return self.fc_mu(h), self.fc_logvar(h)  # (K, D), (K, D)

  def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = self.encoder(x)
    return h, self.fc_mu(h), self.fc_logvar(h)

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    return self.decoder(z)

  def vampprior_kl(self,
                   mu:     torch.Tensor,
                   logvar: torch.Tensor,
                   z:      torch.Tensor) -> torch.Tensor:
    """
    KL( q(z|x) || p_vamp(z) ) estimated with a single sample z ~ q(z|x).

    KL = E_q[log q(z|x)] - E_q[log p(z)]
       = E_q[log q(z|x)] - E_q[log (1/K) sum_k q(z|u_k)]

    mu, logvar, z : (B, D)
    returns scalar

    Side effects (read by epoch_vae / collect_deq_diagnostics):
      self._effective_K, self._effective_K_per_sample : diagnostics only,
          no_grad, unchanged from before.
      self._vamp_diversity_loss, self._vamp_repulsion_loss : anti-collapse
          regularizer terms, WITH gradient -- see class docstring and the
          inline comments below for what each targets.
    """
    # log q(z | x) : (B,)
    log_q = gaussian_log_prob(z, mu, logvar).sum(dim=-1)

    # Encode all pseudo-inputs once per forward pass
    pu_mu, pu_logvar = self._pseudo_encoded()  # (K, D)

    # Broadcast: z (B,1,D), pu_mu/pu_logvar (1,K,D)
    log_p_k = gaussian_log_prob(
        z.unsqueeze(1),           # (B, 1, D)
        pu_mu.unsqueeze(0),       # (1, K, D)
        pu_logvar.unsqueeze(0),   # (1, K, D)
    ).sum(dim=-1)                 # (B, K)

    # Responsibility distribution -- kept differentiable (NOT wrapped in
    # no_grad) because _vamp_diversity_loss below needs its gradient to
    # actually discourage collapse during training. The no_grad block
    # further down reuses this same `resp`/`ent` purely for read-only
    # diagnostics (_effective_K*), which is unaffected by resp being
    # differentiable here (no_grad there just stops those particular
    # diagnostic computations from being tracked further, it doesn't
    # detach resp itself for use elsewhere).
    log_resp = log_p_k - torch.logsumexp(log_p_k, dim=1, keepdim=True)  # (B, K)
    resp = log_resp.exp()

    # Full per-sample responsibility matrix, detached -- diagnostic only.
    # Lets external code (e.g. vamp_usage_by_class) join this against
    # per-cell class/cell-type labels to test whether the mixture's
    # diversity is actually class-aligned (different classes routed to
    # different pseudo-inputs), as opposed to diverse only in the
    # aggregate/population sense that effective_K_population confirms.
    with torch.no_grad():
      self._vamp_resp = resp.detach()  # (B, K)

    # effective K (diagnostic only)
    # range: 1.0 - n_pseudo
    with torch.no_grad():
      # effective K = exp(mean_batch[ entropy(resp)])
      ent = -(resp * (resp + 1e-20).log()).sum(dim=1)  # (B,)
      self._effective_K = torch.exp(ent.mean())
      # Per-sample version of the same diagnostic: exp(entropy) for each
      # individual cell's responsibility distribution over the K
      # pseudo-inputs, rather than averaged over the batch first. A
      # batch-level self._effective_K can look healthy (e.g. ~10 out of 50)
      # while actually being an average of "well-covered" cells (many
      # components effectively responsible, per-sample value close to K)
      # and "poorly-covered" cells (collapsed onto ~1 component each) --
      # the aggregate scalar can't distinguish uniform partial coverage from
      # a bimodal mix. Grouping this per-sample value by class/label
      # reveals whether specific classes are consistently under-covered by
      # the mixture prior (a partial mode-collapse signature) rather than
      # collapse being spread evenly across all cells.
      self._effective_K_per_sample = torch.exp(ent).detach()  # (B,)

    # --- Anti-collapse regularizer 1: population-level responsibility entropy ---
    # IMPORTANT (see class docstring "effective_K vs effective_K_population"):
    # self._effective_K above (exp of the MEAN of per-sample entropies) is
    # NOT the same quantity as "how many of the K pseudo-inputs does the
    # population, in aggregate, actually use" -- by Jensen's inequality
    # (entropy is concave), exp(mean_i[entropy_i]) <= exp(entropy(mean_i)).
    # In fact self._effective_K reads EXACTLY 1.0 even in the theoretical
    # best case (every cell confidently, one-hot, assigned to its OWN
    # distinct pseudo-input, perfectly spread across all K) -- confirmed by
    # direct calculation. This is expected/routine sharpening in a
    # high-dimensional (latent_dim=32) Gaussian mixture, not evidence of
    # collapse by itself. self._effective_K_population below (exp of the
    # entropy of avg_resp, i.e. entropy computed AFTER averaging over the
    # batch) is the metric that actually answers "is the population's usage
    # of the K components diverse", and is what the three regularizers
    # above/below are actually pulling toward. Track BOTH going forward:
    # self._effective_K near 1 is normal; self._effective_K_population near
    # 1 is the real collapse signal.
    # avg_resp: how much of the *whole batch's* total responsibility mass
    # falls on each pseudo-input, aggregated over cells. Same sign/scale
    # convention as MoVEVAE's gating_entropy_loss (synthetic_data.py):
    # (avg_resp * log(avg_resp)).sum() ranges from -log(K) (avg_resp
    # uniform -- every pseudo-input shares the load) to 0 (avg_resp
    # one-hot -- one pseudo-input absorbs virtually the whole batch).
    # epoch_vae adds lambda_entropy * this term to the loss, so minimizing
    # loss pushes toward uniform aggregate usage.
    # Blind spot (see class docstring): satisfiable by K identical/redundant
    # components, which is why regularizer 2 below also exists.
    avg_resp = resp.mean(dim=0)  # (K,), differentiable
    self._vamp_diversity_loss = (avg_resp * (avg_resp + 1e-20).log()).sum()
    with torch.no_grad():
      # exp(entropy(avg_resp)) -- see the note above self._effective_K_per_sample
      # for why this (population-level) quantity, not self._effective_K, is
      # the real collapse diagnostic. self._vamp_diversity_loss == sum(p*log p)
      # == -entropy(avg_resp), so entropy = -diversity_loss (reused directly,
      # no need to recompute).
      self._effective_K_population = torch.exp(-self._vamp_diversity_loss.detach())

    # Shared reference scale for regularizers 2 and 3 below: median pairwise
    # distance among the REAL batch's encoded posterior means (mu, not
    # pu_mu). Ties "what counts as near/far in latent space" to the actual
    # current scale/spread of the encoder's output, so neither regularizer
    # needs re-tuning per latent_dim / encoder architecture / dataset (and
    # automatically tracks the DEQ's tanh-bounded scale for
    # DEQEncoderVampVAE, vs. an unbounded scale for plain VampPriorVAE,
    # without needing to know which subclass this is). Detached
    # deliberately: this is a reference scale, not something we want
    # gradients flowing back through into the real batch's mu.
    K = pu_mu.shape[0]
    B = mu.shape[0]
    with torch.no_grad():
      if B >= 2:
        batch_pdist = torch.cdist(mu, mu, p=2)  # (B, B)
        off_diag_b = ~torch.eye(B, dtype=torch.bool, device=mu.device)
        typical_scale = batch_pdist[off_diag_b].median()
      else:
        typical_scale = mu.new_tensor(1.0)
      self._vamp_repulsion_typical_scale = typical_scale.detach()

    # --- Anti-collapse regularizer 2: adaptive-margin pairwise repulsion ---
    # Forces the pseudo-inputs' *encoded* posterior means (pu_mu) apart in
    # latent space, independent of how responsibility mass happens to be
    # distributed (closing regularizer 1's blind spot: K clones can't
    # satisfy this one, since their pairwise distance is ~0).
    #
    # Caveat learned from a real training run: this alone is NOT sufficient
    # to prevent effective_K collapse. Mutual repulsion only pushes
    # pseudo-inputs apart from EACH OTHER -- it has no notion of "near the
    # real data". In a high-dimensional ambient latent space, pseudo-inputs
    # can become nicely spread out from each other (confirmed: this
    # regularizer converges with pu_mu pairwise distance comfortably above
    # margin) while still mostly scattering away from the (comparatively
    # low-dimensional, per requirements.md) data manifold, leaving only ~1
    # of K ever close enough to compete for responsibility. Regularizer 3
    # below (coverage) is what actually targets that.
    if K >= 2:
      with torch.no_grad():
        margin = typical_scale * self.repulsion_margin_frac
        self._vamp_repulsion_margin = margin.detach()

      # Hinge repulsion: only pairs closer than `margin` incur any penalty
      # (bounded, and gives zero gradient once components are adequately
      # spread out -- doesn't fight reconstruction/KL once diversity is
      # achieved).
      pu_pdist = torch.cdist(pu_mu, pu_mu, p=2)  # (K, K), differentiable
      off_diag_k = ~torch.eye(K, dtype=torch.bool, device=pu_mu.device)
      self._vamp_repulsion_loss = F.relu(margin - pu_pdist[off_diag_k]).pow(2).mean()

      with torch.no_grad():
        self._vamp_pu_mu_pairwise_median = pu_pdist[off_diag_k].median().detach()
        raw_pdist = torch.cdist(self.pseudo_inputs, self.pseudo_inputs, p=2)  # (K, K), raw gene-expression space
        self._vamp_pseudo_inputs_raw_pairwise_median = raw_pdist[off_diag_k].median().detach()
    else:
      self._vamp_repulsion_loss                     = pu_mu.new_tensor(0.0)
      self._vamp_repulsion_margin                    = pu_mu.new_tensor(float('nan'))
      self._vamp_pu_mu_pairwise_median               = pu_mu.new_tensor(float('nan'))
      self._vamp_pseudo_inputs_raw_pairwise_median   = pu_mu.new_tensor(float('nan'))

    # --- Anti-collapse regularizer 3: attraction to the real data manifold ---
    # Pulls each pseudo-input toward its *nearest* real-batch encoded mu
    # (a Chamfer-style coverage loss), so pseudo-inputs individually land
    # near where the data actually is -- something regularizer 2's mutual
    # repulsion cannot provide on its own (it only discourages pseudo-inputs
    # from being close to EACH OTHER; nothing pulls them toward the data).
    # Combined with repulsion, this gives the standard "attract to data,
    # repel from siblings" pattern: pseudo-inputs get pulled onto the
    # manifold individually, but can't all pile onto the same nearby point
    # because repulsion still pushes them apart from each other.
    #
    # Distances are normalized by typical_scale (same reference as the
    # repulsion margin) so the loss magnitude/weighting is comparable
    # across different latent scales/architectures rather than tied to
    # whatever the raw absolute latent-space units happen to be.
    if K >= 1 and B >= 1:
      # mu.detach(): this pull should be one-directional (move pu_mu toward
      # the real data), NOT the reverse -- without detaching, gradients
      # would also flow into the real batch's mu, pulling actual data
      # points toward wherever the pseudo-inputs happen to be, which would
      # distort the encoder's fit to real data instead of just repositioning
      # the pseudo-inputs.
      dist_to_real = torch.cdist(pu_mu, mu.detach())       # (K, B); differentiable only w.r.t. pu_mu
      nearest_dist = dist_to_real.min(dim=1).values        # (K,) -- each pseudo-input's distance to its closest real cell
      normalized_nearest_dist = nearest_dist / (typical_scale + 1e-6)
      self._vamp_coverage_loss = normalized_nearest_dist.pow(2).mean()
      with torch.no_grad():
        self._vamp_coverage_mean_nearest_dist = nearest_dist.mean().detach()
    else:
      self._vamp_coverage_loss               = pu_mu.new_tensor(0.0)
      self._vamp_coverage_mean_nearest_dist  = pu_mu.new_tensor(float('nan'))

    log_p = torch.logsumexp(log_p_k, dim=1) - math.log(self.n_pseudo)  # (B,)

    kl_per_sample = log_q - log_p  # (B,)
    with torch.no_grad():
      # Per-cell KL cost, detached -- diagnostic only (the differentiable
      # kl_per_sample.mean() below is still what's actually optimized).
      # Purpose: test a DIFFERENT question from vamp_usage_by_class's
      # class-vs-pseudo-input routing. That showed classes route to
      # disjoint pseudo-inputs (a RATE-ALLOCATION effect: it's cheaper for
      # a class to sit in its own dedicated region than to be forced near a
      # single global N(0,I) mode) -- but rate allocation being more
      # class-fair doesn't imply reconstruction FIDELITY is more
      # class-fair, since fidelity is governed by posterior precision
      # (logvar) and decoder capacity, not by where the prior's mass sits.
      # Group this per-cell value with class_grouped_stats and compare its
      # between_class_variance against the equivalent quantity for a
      # standard-prior model (see standard_gaussian_kl_per_sample) to test
      # whether VampPrior actually makes per-class KL COST more uniform,
      # even if per-class reconstruction ERROR (per_cell_masked_error)
      # isn't.
      self._vamp_kl_per_sample = kl_per_sample.detach()  # (B,)

    return kl_per_sample.mean()  # scalar

  def forward(self, x: torch.Tensor):
    """
    Returns: (recon, mu, logvar, None, None, None)
    Side-effect: stores VampPrior KL in self._vamp_kl, plus the three
    anti-collapse regularizer terms self._vamp_diversity_loss /
    self._vamp_repulsion_loss / self._vamp_coverage_loss (see vampprior_kl),
    after each call. All are picked up automatically by epoch_vae via
    getattr/isinstance checks.
    """
    _, mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z)
    self._vamp_kl = self.vampprior_kl(mu, logvar, z)
    return recon, mu, logvar, None, None, None


class DEQEncoderVampVAE(VampPriorVAE):
  """
  Stage 2: VampPriorVAE with a DEQ (implicit-depth) encoder instead of a
  plain MLP head.

  Combines two previously-isolated changes:
    - DEQEncoderVAE's weight-tied fixed-point recognition network (encoder
      question / requirements b,c: missing-data handling, sample efficiency)
    - VampPriorVAE's mixture-of-posteriors prior (prior question /
      requirement a: multimodal, mode-collapse-resistant prior)

  Subtlety #1 -- prior and posterior share one DEQCell:
    Both the real batch and the K learnable pseudo-inputs are encoded
    through the *same* DEQCell instance. This means the prior's mixture
    components q(z|u_k) are fixed points of the identical implicit function
    that defines the posterior q(z|x), just evaluated at different
    conditioning vectors -- there is no separate "prior network" the way
    there would be for, e.g., a learned normalizing-flow prior. This is a
    deliberate design choice (keeps the "one weight-tied transition" DEQ
    argument intact) but it does mean the two roles are coupled: any change
    to W_zz that helps fit real data also reshapes every pseudo-input's
    fixed point, and vice versa.

  Subtlety #2 -- subclassing VampPriorVAE, not DEQEncoderVAE:
    epoch_vae (models.py) dispatches the VampPrior KL override via
    `isinstance(model, VampPriorVAE)`. Subclassing VampPriorVAE means that
    check fires with zero changes to the shared training loop. The
    DEQ-specific pieces (fc_cond, deq_cell) are grafted on top instead of
    inherited from DEQEncoderVAE, since VampPriorVAE.__init__ already builds
    everything else this class needs (encoder, decoder, pseudo_inputs,
    vampprior_kl).

  Subtlety #3 -- fc_mu/fc_logvar are re-declared, not reused:
    VampPriorVAE.__init__ (via super().__init__ below) allocates
    self.fc_mu / self.fc_logvar as Linear(enc_out_dim, latent_dim), sized to
    consume the *raw encoder output* h directly (its own encode() applies
    them to h). Here, mu/logvar must instead be read off the DEQ fixed point
    z* (dim == latent_dim), so those two Linear layers are immediately
    overwritten with Linear(latent_dim, latent_dim) versions after
    super().__init__() returns. The two original Linear(enc_out_dim,
    latent_dim) modules are allocated and then discarded (replaced before
    any forward/optimizer step touches them) -- wasted construction cost
    only, no effect on parameters actually trained or saved.

  Subtlety #4 -- DEQCell diagnostics get overwritten by the pseudo-input pass:
    DEQCell.forward() records convergence diagnostics (last_forward_iters /
    last_forward_residual / last_forward_residual_per_sample / last_z_star)
    as attributes on the DEQCell instance itself, updated on every call.
    Within one forward() call here, deq_cell runs twice: once for the real
    batch (inside encode()), and once more for the K pseudo-inputs (inside
    vampprior_kl() -> _pseudo_encoded(), triggered right after). The second
    call silently overwrites the first call's diagnostics, so reading
    self.deq_cell.last_forward_iters (or any of the other last_forward_*
    attributes) after forward() returns would reflect the pseudo-input
    solve, not the real-batch solve we actually want to track (e.g.
    iteration count / per-cell residual vs. mask fraction or class label --
    the diagnostics DEQEncoderVAE/DEQCell were built to expose). Fixed by
    snapshotting self.last_batch_* immediately after the real-batch
    encode() call, before vampprior_kl() runs and clobbers deq_cell's
    shared state. This is *not* an issue for self._effective_K /
    self._effective_K_per_sample (VampPriorVAE.vampprior_kl) -- those are
    computed exactly once per forward() call, from the already-encoded real
    batch and pseudo-inputs together, so there's no second overwriting call.
  """

  def __init__(self,
               n_genes:      int,
               latent_dim:   int = 32,
               n_pseudo:     int = 50,
               hidden_dims:  list[int]|None = None,
               encoder_dims: list[int]|None = None,
               decoder_dims: list[int]|None = None,
               dropout:      float = 0.02,
               coeff:        float = 0.9,
               max_iter:     int = 30,
               tol:          float = 1e-4,
               pseudo_init_samples: torch.Tensor|None = None,
               repulsion_margin_frac: float = 0.5):

    super().__init__(n_genes, latent_dim, n_pseudo,
                      hidden_dims, encoder_dims, decoder_dims, dropout,
                      pseudo_init_samples, repulsion_margin_frac)
    # VampPriorVAE.__init__ already built self.encoder, self.decoder,
    # self.pseudo_inputs, and (see Subtlety #3) fc_mu/fc_logvar sized for
    # the *wrong* input dim -- replace them here.
    self.fc_mu     = nn.Linear(latent_dim, latent_dim)
    self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    self.fc_cond  = nn.Linear(self.enc_out_dim, latent_dim)
    self.deq_cell = DEQCell(latent_dim, coeff=coeff, max_iter=max_iter, tol=tol)

    # Real-batch DEQ convergence diagnostics, snapshotted in forward() before
    # the pseudo-input pass overwrites deq_cell's shared attributes (see
    # Subtlety #4). None until the first forward() call.
    self.last_batch_iters:             int|None = None
    self.last_batch_residual:        float|None = None
    self.last_batch_residual_per_sample: torch.Tensor|None = None
    self.last_batch_z_star:              torch.Tensor|None = None

  def _encode_via_deq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared path for both real data and pseudo-inputs (see Subtlety #1)."""
    h = self.encoder(x)
    c = self.fc_cond(h)
    z_star = self.deq_cell(c)
    return h, self.fc_mu(z_star), self.fc_logvar(z_star)

  def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self._encode_via_deq(x)

  def _pseudo_encoded(self) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode all K pseudo-inputs through the same encoder+DEQ path as real
    data (overrides VampPriorVAE._pseudo_encoded, which used a plain MLP
    head). Called by the inherited vampprior_kl() -- this is also the call
    that overwrites deq_cell's shared diagnostics (Subtlety #4).
    """
    zero_mask  = torch.zeros(self.n_pseudo, self.n_genes, device=self.pseudo_inputs.device)
    pseudo_aug = torch.cat([self.pseudo_inputs, zero_mask], dim=1)
    _, pu_mu, pu_logvar = self._encode_via_deq(pseudo_aug)
    return pu_mu, pu_logvar

  def forward(self, x: torch.Tensor):
    """
    Returns: (recon, mu, logvar, None, None, None) -- same 6-tuple contract
    as every other model in this file.
    Side-effects:
      - self._vamp_kl set (read by epoch_vae's kl_override for this
        isinstance(model, VampPriorVAE) branch).
      - self._effective_K / self._effective_K_per_sample set by
        vampprior_kl() (no clobbering concern, see Subtlety #4).
      - self.last_batch_iters / last_batch_residual /
        last_batch_residual_per_sample / last_batch_z_star snapshotted for
        the *real batch* solve specifically (see Subtlety #4) -- read these
        instead of self.deq_cell.last_forward_* after this returns.
    """
    _, mu, logvar = self.encode(x)
    # Snapshot now: vampprior_kl() below re-runs deq_cell on pseudo-inputs
    # and would otherwise overwrite these.
    self.last_batch_iters               = self.deq_cell.last_forward_iters
    self.last_batch_residual            = self.deq_cell.last_forward_residual
    self.last_batch_residual_per_sample = self.deq_cell.last_forward_residual_per_sample
    self.last_batch_z_star              = self.deq_cell.last_z_star

    z = self.reparameterize(mu, logvar)
    recon = self.decode(z)
    self._vamp_kl = self.vampprior_kl(mu, logvar, z)
    return recon, mu, logvar, None, None, None


def epoch_vae(model, loader, opt=None, mask_fraction=0.1, beta=1.0, gamma=0.2, min_free_bits=0.05, lambda_entropy=0.1, lambda_vamp_repulsion=0.1, lambda_vamp_coverage=0.1):
  total_loss, total_err, grad_norm, recon_loss, kl_loss = 0.,0.,0.,0.,0.

  model.eval() if opt is None else model.train()

  loss_func = masked_loss # nn.MSELoss()

  if opt:
    opt.zero_grad()

  #curr_step = 0

  recon_loss_type2 = torch.tensor(0.0)

  for x in loader:

    x_masked,  mask, type2_mask = get_random_mask(x, mask_fraction)
    x_masked = torch.cat( (x_masked, mask.float()), dim=1 )
    x, x_masked ,mask, type2_mask = x.to(device), x_masked.to(device), mask.to(device), type2_mask.to(device)

    ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
    #return x_masked, mask
    ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##

    #assert( ~torch.isnan( x_masked ).any() )
    #assert( ~torch.isnan( mask ).any() )

    if opt:
      pred, mu, logvar, gate_probs, soft_weights, _ = model(x_masked)
    else:
      with torch.no_grad():
        pred, mu, logvar, gate_probs, soft_weights, _ = model(x_masked)

    #assert( ~torch.isnan( pred ).any() )
    #assert( ~torch.isnan( mu ).any() )
    #assert( ~torch.isnan( logvar ).any() )

    x.nan_to_num_(nan=0)

    kl_override = model._vamp_kl if isinstance(model, VampPriorVAE) else None

    loss, recon_loss, recon_loss_type2, kl_loss = loss_func(
        x,
        pred,
        mask,
        mu,
        logvar,
        gate_probs,
        soft_weights,
        beta=beta,
        gamma=gamma,
        mask_type2=type2_mask,
        free_bits=min_free_bits,
        lambda_entropy=lambda_entropy,
        kl_override=kl_override)

    # Anti-collapse regularizers for VampPriorVAE / DEQEncoderVampVAE (see
    # VampPriorVAE.vampprior_kl docstring). Additive, independent weights:
    # the entropy term reuses lambda_entropy (same convention as MoVEVAE's
    # gating_entropy_loss); repulsion and coverage each get their own weight
    # since they're geometric constraints rather than entropy targets, and
    # target different failure modes (repulsion alone was confirmed
    # insufficient -- see vampprior_kl -- coverage is what actually pulls
    # pseudo-inputs toward the data manifold).
    vamp_diversity_loss = getattr(model, '_vamp_diversity_loss', None)
    if vamp_diversity_loss is not None:
      loss = loss + lambda_entropy * vamp_diversity_loss

    vamp_repulsion_loss = getattr(model, '_vamp_repulsion_loss', None)
    if vamp_repulsion_loss is not None:
      loss = loss + lambda_vamp_repulsion * vamp_repulsion_loss

    vamp_coverage_loss = getattr(model, '_vamp_coverage_loss', None)
    if vamp_coverage_loss is not None:
      loss = loss + lambda_vamp_coverage * vamp_coverage_loss

    if opt:
      loss.backward()

      # record gradient histogram
      #if (curr_step+5)==len(loader):
      #  _debug_grads = collect_grads(model, (z.shape[1],z.shape[1]))

      # update weights
      grad_norm += torch.nn.utils.clip_grad_norm_( model.parameters(), max_norm=1 ) # clip in-place

      opt.step()
      opt.zero_grad()

    total_loss += loss.item() #* accumulate_steps # undo accumulation scaling  # / z_train.shape[0]

    #curr_step += 1


  total_loss /= len(loader)
  grad_norm  /= len(loader)

  return total_loss, grad_norm, recon_loss.detach(), recon_loss_type2.detach(), kl_loss.detach()*beta


# ---------------------------------------------------------------------------
# SSM Beta Controller, World, Controller, and REINFORCE training loop
# ---------------------------------------------------------------------------

class RunningNorm:
    """
    Per-episode exponential moving-average normalizer.

    Tracks mean and variance of a D-dimensional signal using EMA.
    Call .update(x) each step, then .normalize(x) to z-score.
    Reset with .reset() at the start of each episode.
    """
    def __init__(self, dim: int, alpha: float = 0.1, eps: float = 1e-6):
        self.dim   = dim
        self.alpha = alpha
        self.eps   = eps
        self.reset()

    def reset(self):
        self.mean = torch.zeros(self.dim)
        self.var  = torch.ones(self.dim)
        self._initialized = False

    def update(self, x: torch.Tensor):
        """x: (dim,) — one observation vector (detached, on CPU)."""
        x = x.detach().cpu().float()
        if not self._initialized:
            self.mean = x.clone()
            self.var  = torch.ones(self.dim)
            self._initialized = True
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * x
            self.var  = (1 - self.alpha) * self.var  + self.alpha * (x - self.mean).pow(2)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Returns z-scored x using current EMA statistics."""
        mean = self.mean.to(x.device)
        std  = self.var.clamp(min=self.eps).sqrt().to(x.device)
        return (x - mean) / std


class BetaSSMController(nn.Module):
    """
    Tiny diagonal linear SSM that outputs a beta distribution for VAE training.

    h_t = hidden state
    x_t = observations
    y_t = control output
    A   = hidden state evolution
    B   = observed variable effects
    C   = hidden state control contribution
    D   = observed variable control contribution

    State-space model (diagonal A for efficiency and stability):
        h_t = diag(A) * h_{t-1} + B_ctrl @ x_t
        y_t = C @ h_t + D @ x_t          → (mu_log_beta, log_sigma)

    A is parameterized via softplus to keep eigenvalues in (0, 1) (stable).

    Action: beta ~ LogNormal(mu_log_beta, exp(log_sigma)), clipped to [1e-5, 0.1].

    Two input paths:
      B_ctrl  (hidden_dim × obs_dim)     — used during control (act)
      B_world (hidden_dim × obs_dim+1)   — used during world-model pretraining;
                                           the extra column encodes log(beta_t),
                                           first obs_dim columns tied to B_ctrl
                                           at init (but trained independently).


    Args:
        obs_dim:    dimension of the observation vector (default 12)
        hidden_dim: SSM state dimension (default 32)
    """
    def __init__(self, obs_dim: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim

        # Diagonal A: A = exp(-softplus(a_raw)) ∈ (0, 1)
        self.a_raw   = nn.Parameter(torch.zeros(hidden_dim))

        # Control path input matrix (obs_dim → hidden_dim)
        self.B_ctrl  = nn.Parameter(torch.randn(hidden_dim, obs_dim) * 0.01)

        # World-model pretraining input matrix (obs_dim+1 → hidden_dim).
        # Initialised so first obs_dim columns match B_ctrl.
        B_world_init = torch.randn(hidden_dim, obs_dim + 1) * 0.01
        B_world_init[:, :obs_dim] = self.B_ctrl.data.clone()
        self.B_world = nn.Parameter(B_world_init)

        # Output projection (shared by both paths)
        self.C        = nn.Parameter(torch.randn(2, hidden_dim) * 0.01)
        self.D        = nn.Parameter(torch.zeros(2, obs_dim))

        # Output bias: mu_log_beta initialised to centre of log-range
        # log([1e-5, 0.1]) midpoint = (log(1e-5)+log(0.1))/2 ≈ -6.9
        # log_sigma initialised to 1.5 → sigma≈4.5, covering the full range
        self.out_bias = nn.Parameter(torch.tensor([-6.9, 1.5]))

    def _step(self, x_proj: torch.Tensor, x_obs: torch.Tensor,
              h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared SSM core given a pre-projected input x_proj (hidden_dim,)."""
        A     = torch.exp(-F.softplus(self.a_raw))          # (hidden_dim,)
        h_new = A * h + x_proj                               # (hidden_dim,)
        y     = self.C @ h_new + self.D @ x_obs + self.out_bias  # (2,)
        mu_log_beta = y[0]
        log_sigma   = y[1].clamp(-4.0, 2.0)
        return mu_log_beta, log_sigma, h_new


    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Control path: single SSM step using B_ctrl.

        Args:
            x: observation vector (obs_dim,)
            h: hidden state       (hidden_dim,)
        Returns:
            mu_log_beta, log_sigma, h_new
        """
        x = x.view(-1)
        h = h.view(-1)
        return self._step(self.B_ctrl @ x, x, h)

    def forward_world(self, x: torch.Tensor, log_beta: torch.Tensor,
                      h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        World-model pretraining path: single SSM step using B_world.
        Input is augmented with log(beta_t) so the model can learn how
        beta affects future observations.

        Args:
            x:        observation vector (obs_dim,)
            log_beta: scalar log of the beta applied this step
            h:        hidden state (hidden_dim,)
        Returns:
            mu_log_beta, log_sigma, h_new
        """
        x        = x.view(-1)
        h        = h.view(-1)
        log_beta = log_beta.view(1)
        x_aug    = torch.cat([x, log_beta], dim=0)   # (obs_dim+1,)
        return self._step(self.B_world @ x_aug, x, h)

    def initial_state(self, device=None) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, device=device)

# ---------------------------------------------------------------------------
# Beta source functions
# (t: int, obs: Tensor) -> float   — all share this signature
# ---------------------------------------------------------------------------

def beta_triangle(t: int, obs: torch.Tensor, *,
                  period: int = 31, lo: float = 1e-5, hi: float = 0.1) -> float:
    """Triangle wave in log-space between lo and hi."""
    phase = (t % period) / period          # [0, 1)
    frac  = 1.0 - 2.0 * abs(phase - 0.5)  # triangle: 0→1→0
    return lo * (hi / lo) ** frac


def beta_sine(t: int, obs: torch.Tensor, *,
              period: int = 12, lo: float = 1e-5, hi: float = 0.5) -> float:
    """Sine wave in log-space between lo and hi."""
    frac = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / period)
    return lo * (hi / lo) ** frac


def beta_impulse(t: int, obs: torch.Tensor, *,
                 impulse_epochs: tuple = (5, 12, 25, 32,50, 75),
                 lo: float = 1e-5, hi: float = 0.1) -> float:
    """Low beta everywhere except brief spikes at specified epochs."""
    return hi if t in impulse_epochs else lo

def beta_flip(t:int, obs:torch.Tensor, orig, lo: float = 1e-5, hi: float = 0.1) -> float:
  return orig(t, obs, hi=lo, lo=hi)

def beta_constant(t: int, obs: torch.Tensor, *,
                  value: float = 1e-4) -> float:
    """Fixed beta for every epoch."""
    return value


def beta_cosine_warmup(t: int, obs: torch.Tensor, *,
                       T: int = 30, lo: float = 1e-5, hi: float = 0.1) -> float:
    """Cosine annealing from lo to hi over T epochs (mirrors existing schedule)."""
    frac = 0.5 * (1.0 - math.cos(math.pi * min(t, T - 1) / T))
    return lo * (hi / lo) ** frac


def beta_from_controller(controller: 'Controller'):
    """
    Adapter: wraps a Controller so it can be used as a BetaSource.
    log_prob is discarded — only beta is returned.
    The caller is responsible for resetting controller state before the episode.
    """
    def _src(t: int, obs: torch.Tensor) -> float:
        beta, _ = controller.act(obs)
        return beta
    return _src


# Convenience list of all built-in waveform sources (excluding controller adapter)
BUILTIN_BETA_SOURCES = [beta_triangle, beta_sine, beta_impulse, lambda t,obs: beta_flip(t, obs, beta_impulse),
                         ]



# ---------------------------------------------------------------------------
# Helpers: data and VAE construction
# ---------------------------------------------------------------------------

def _build_obs_raw(metrics: dict, meta: dict) -> torch.Tensor:
    """
    Assemble the raw (un-normalized) 12-d observation vector.

    Dynamic features (9):
        recon_loss, kl_loss, recon_loss_type2, grad_norm, effective_K,
        delta_recon, delta_kl, delta_eff_K, log(epoch+1)

    Fixed meta-parameter features (3 — fixed normalization, not z-scored):
        log(n_genes)/log(1000), log(n_cells)/log(10000), missing_rate

    Returns a CPU float32 tensor of shape (12,).
    """
    recon      = float(metrics.get('recon_loss',       0.0))
    kl         = float(metrics.get('kl_loss',          0.0))
    recon2     = float(metrics.get('recon_loss_type2', 0.0))
    gnorm      = float(metrics.get('grad_norm',        0.0))
    eff_k      = float(metrics.get('effective_K',      1.0))
    prev_recon = float(metrics.get('prev_recon',       recon))
    prev_kl    = float(metrics.get('prev_kl',          kl))
    prev_eff_k = float(metrics.get('prev_eff_K',       eff_k))
    epoch      = float(metrics.get('epoch',            0))

    dyn = torch.tensor([
        recon, kl, recon2, gnorm, eff_k,
        recon - prev_recon,
        kl    - prev_kl,
        eff_k - prev_eff_k,
        math.log(epoch + 1),
    ], dtype=torch.float32)

    meta_feat = torch.tensor([
        math.log(meta['n_genes']) / math.log(1000.0),
        math.log(meta['n_cells']) / math.log(10000.0),
        meta['missing_rate'],
    ], dtype=torch.float32)

    return torch.cat([dyn, meta_feat], dim=0)   # (12,) on CPU



def _normalize_obs(raw_obs: torch.Tensor,
                   running_norm: RunningNorm,
                   device) -> torch.Tensor:
    """
    Update running stats with the 9 dynamic features of raw_obs,
    then return the fully normalized 12-d vector on `device`.
    Meta features (last 3) are already on a fixed scale and are passed through.
    """
    dyn = raw_obs[:9]
    running_norm.update(dyn)
    dyn_normed  = running_norm.normalize(dyn).to(device)
    meta_normed = raw_obs[9:].to(device)
    return torch.cat([dyn_normed, meta_normed], dim=0)



def _make_loaders(problem_config: dict, batch_size: int, device):
    """Build train / test / train-full DataLoaders for one episode."""
    n_cells  = problem_config['n_cells']
    n_blocks = problem_config['n_blocks']

    block_state = random_points_on_hypersphere(
        num_points=n_cells,
        K=n_blocks,
        N=min(24, n_blocks - 1),
        device=device,
    )
    scale = block_state.abs().max(dim=0).values.clamp(min=1e-6)
    block_state = block_state / scale * 4.0

    datasets = []
    for _ in range(2):
        data, _ = make_synthetic_data4(
            block_state=block_state,
            n_genes=problem_config['n_genes'],
            n_blocks=n_blocks,
            rho_within=problem_config['rho_within'],
            allow_negative=True,
            missing_rate=problem_config['missing_rate'],
            seed=None,
            device=device,
        )
        datasets.append(data)

    data_train, data_test = datasets
    return (
        DataLoader(data_train, batch_size=batch_size,  shuffle=True),
        DataLoader(data_test,  batch_size=n_cells,     shuffle=False),
        DataLoader(data_train, batch_size=n_cells,     shuffle=False),
    )


def _make_vae(problem_config: dict, loader_train_full,
              vae_training_config: dict, device):
    """Instantiate a fresh VampPriorVAE + AdamW optimizer."""
    for x in loader_train_full:
        x_init = x.nan_to_num(0).to(device)
        break
    vae = VampPriorVAE(
        n_genes=problem_config['n_genes'],
        n_pseudo=vae_training_config['n_pseudo'],
        dropout=0.02,
        encoder_dims=[512, 256, 128],
        decoder_dims=[128, 200, 200],
        latent_dim=64,
        pseudo_init_samples=x_init,
    ).to(device)
    opt = optim.AdamW(vae.parameters(), lr=vae_training_config['lr'])
    return vae, opt

# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
class World:
    """
    Encapsulates one VAE training problem and the controlled VAE model.

    Lifecycle per episode:
        obs = world.reset(problem_config)  # fresh data + VAE + ref epoch
        for t in range(max_epochs):
            beta = <some source>
            obs  = world.step(beta)        # train one epoch, return next obs
        reward, recon_term, eff_k_term = world.get_reward()

    world.obs_history: list of (raw_obs_tensor_CPU, beta_float) — one entry
                       per step, suitable for the replay buffer.
    """

    def __init__(self,
                 vae_training_config: dict,
                 vae_loss_config:     dict,
                 base_problem_config: dict,
                 beta_ref:  float = 1e-4,
                 lambda_K:  float = 0.1,
                 device            = None):
        """
        Args:
            vae_training_config: max_epochs, batch_size, mask_fraction, lr, n_pseudo
            vae_loss_config:     gamma_recon, min_free_bits, lambda_entropy
            base_problem_config: fixed keys shared across all episodes
                                 (e.g. n_blocks, rho_within); varied keys
                                 (n_genes, n_cells, missing_rate) are
                                 overridden per episode in reset().
            beta_ref:   beta used for the reference epoch (reward normalisation)
            lambda_K:   weight for effective_K term in reward
            device:     torch device
        """
        self.vtc              = vae_training_config
        self.vlc              = vae_loss_config
        self.base_problem_config = base_problem_config   # fixed across episodes
        self.beta_ref         = beta_ref
        self.lambda_K         = lambda_K
        self.device           = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs       = vae_training_config['max_epochs']

        # Set during reset()
        self.meta:         dict           = {}
        self.problem_config: dict         = {}
        self.vae                          = None
        self.vae_opt                      = None
        self.loader_train                 = None
        self.loader_test                  = None
        self.loader_train_full            = None
        self.ref_recon:    float          = 1.0
        self.running_norm: RunningNorm    = RunningNorm(dim=9)
        self.obs_history:  list           = []
        self._metrics:     dict           = {}
        self._last_eff_k:  float          = 1.0

    def reset(self, problem_config: dict) -> torch.Tensor:
        """
        Prepare a fresh episode:
          1. Build dataset from problem_config.
          2. Run reference epoch (1 epoch, beta_ref) → self.ref_recon.
          3. Instantiate a second fresh VAE for the controlled episode.
          4. Reset running normalizer and obs_history.

        Returns the initial (all-zeros) observation tensor.
        """
        self.problem_config = problem_config
        self.meta = dict(
            n_genes      = problem_config['n_genes'],
            n_cells      = problem_config['n_cells'],
            missing_rate = problem_config['missing_rate'],
        )

        # Build loaders
        self.loader_train, self.loader_test, self.loader_train_full = _make_loaders(
            problem_config, self.vtc['batch_size'], self.device)

        # Reference epoch
        vae_ref, opt_ref = _make_vae(problem_config, self.loader_train_full,
                                     self.vtc, self.device)
        epoch_vae(vae_ref, self.loader_train, opt_ref,
                  mask_fraction=self.vtc['mask_fraction'], beta=self.beta_ref,
                  gamma=self.vlc['gamma_recon'], min_free_bits=self.vlc['min_free_bits'],
                  lambda_entropy=self.vlc['lambda_entropy'])
        _, _, ref_test_recon, _, _ = epoch_vae(
            vae_ref, self.loader_test, opt=None,
            mask_fraction=self.vtc['mask_fraction'], beta=0.0,
            gamma=self.vlc['gamma_recon'], min_free_bits=self.vlc['min_free_bits'],
            lambda_entropy=self.vlc['lambda_entropy'])
        self.ref_recon = float(ref_test_recon)
        del vae_ref, opt_ref

        # Fresh controlled VAE
        self.vae, self.vae_opt = _make_vae(problem_config, self.loader_train_full,
                                           self.vtc, self.device)

        # Reset per-episode state
        self.running_norm.reset()
        self.obs_history = []
        self._metrics    = dict(recon_loss=0., kl_loss=0., recon_loss_type2=0.,
                                grad_norm=0., effective_K=1.,
                                prev_recon=0., prev_kl=0., prev_eff_K=1., epoch=0)
        self._last_eff_k = 1.0

        print(f'  ref_recon_test={self.ref_recon:.4f}  '
              f'n_genes={problem_config["n_genes"]}  '
              f'n_cells={problem_config["n_cells"]}  '
              f'missing_rate={problem_config["missing_rate"]:.3f}')

        # Return initial (zero) observation
        raw_obs = _build_obs_raw(self._metrics, self.meta)
        return _normalize_obs(raw_obs, self.running_norm, self.device)

    def step(self, beta: float) -> torch.Tensor:
        """
        Train the VAE for one epoch with the given beta.
        Appends (raw_obs_before_step, beta) to obs_history.
        Returns the normalized observation AFTER this epoch.
        """
        t = int(self._metrics['epoch'])

        # Build raw obs BEFORE the step (what the controller acted on)
        raw_obs_pre = _build_obs_raw(self._metrics, self.meta)
        self.obs_history.append((raw_obs_pre.clone(), beta))

        # Train one epoch
        _, grad_norm, recon_l, recon2_l, kl_l = epoch_vae(
            self.vae, self.loader_train, self.vae_opt,
            mask_fraction=self.vtc['mask_fraction'],
            beta=beta,
            gamma=self.vlc['gamma_recon'],
            min_free_bits=self.vlc['min_free_bits'],
            lambda_entropy=self.vlc['lambda_entropy'],
        )
        eff_k = float(getattr(self.vae, '_effective_K', torch.tensor(1.0)))
        self._last_eff_k = eff_k

        # Update metrics for the next observation
        self._metrics = dict(
            recon_loss       = float(recon_l),
            kl_loss          = float(kl_l),
            recon_loss_type2 = float(recon2_l),
            grad_norm        = float(grad_norm),
            effective_K      = eff_k,
            prev_recon       = self._metrics['recon_loss'],
            prev_kl          = self._metrics['kl_loss'],
            prev_eff_K       = self._metrics['effective_K'],
            epoch            = t + 1,
        )

        raw_obs_post = _build_obs_raw(self._metrics, self.meta)
        return _normalize_obs(raw_obs_post, self.running_norm, self.device)

    def get_reward(self) -> tuple[float, float, float]:
        """
        Evaluate test-set reconstruction (beta=0) on the current VAE.
        Returns (reward, recon_term, eff_k_term).
        """
        _, _, final_recon_t, _, _ = epoch_vae(
            self.vae, self.loader_test, opt=None,
            mask_fraction=self.vtc['mask_fraction'], beta=0.0,
            gamma=self.vlc['gamma_recon'], min_free_bits=self.vlc['min_free_bits'],
            lambda_entropy=self.vlc['lambda_entropy'])
        final_recon = float(final_recon_t)

        n_pseudo   = self.vtc['n_pseudo']
        recon_term = (self.ref_recon - final_recon) / (self.ref_recon + 1e-8)
        eff_k_term = math.log(max(self._last_eff_k, 1.0)) / math.log(n_pseudo)
        reward     = recon_term + self.lambda_K * eff_k_term
        return reward, recon_term, eff_k_term

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller:
    """
    Wraps BetaSSMController with:
      - REINFORCE policy update
      - World-model pretraining from a replay buffer
      - EMA reward baseline
    """

    def __init__(self,
                 ssm:            BetaSSMController,
                 lr:             float = 1e-3,
                 baseline_alpha: float = 0.1,
                 device                = None):
        self.ssm            = ssm
        self.device         = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baseline_alpha = baseline_alpha
        self.baseline       = 0.0
        self._opt           = optim.Adam(ssm.parameters(), lr=lr)
        self.h              = ssm.initial_state(device=self.device)

    def reset_state(self):
        """Reset SSM hidden state to zeros (call at the start of each episode)."""
        self.h = self.ssm.initial_state(device=self.device)

    def act(self, obs: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Step the SSM with obs, sample beta ~ LogNormal(mu, sigma).

        Args:
            obs: normalized observation (12,) on device

        Returns:
            beta_float: sampled beta, clipped to [1e-5, 0.1]
            log_prob:   log-probability of the sample (scalar tensor, has grad)
        """
        mu_log_beta, log_sigma, self.h = self.ssm(obs, self.h)
        dist           = torch.distributions.Normal(mu_log_beta, log_sigma.exp())
        log_beta_samp  = dist.rsample()
        log_prob       = dist.log_prob(log_beta_samp)
        beta           = log_beta_samp.exp().clamp(1e-5, 0.1).item()
        return beta, log_prob

    def update(self, log_probs: list, reward: float):
        """
        REINFORCE gradient step.

        Args:
            log_probs: list of scalar log-prob tensors (one per epoch)
            reward:    scalar episode reward
        """
        advantage   = reward - self.baseline
        policy_loss = -advantage * torch.stack(log_probs).sum()

        self._opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ssm.parameters(), max_norm=1.0)
        self._opt.step()

        self.baseline = ((1 - self.baseline_alpha) * self.baseline
                         + self.baseline_alpha * reward)

    def pretrain(self,
                 replay_buffer: list,
                 n_steps:       int   = 500,
                 lr_pretrain:   float = 1e-3,
                 seq_len:       int   = 50):
        """
        World-model pretraining via next-observation prediction.

        For each gradient step:
          1. Sample a random episode from the replay buffer.
          2. Sample a random start within it (subsequence of length seq_len).
          3. Re-run the SSM forward pass using forward_world(obs_t, log_beta_t).
          4. Predict obs_{t+1} with a linear head.
          5. Loss = MSE(prediction, actual_obs_{t+1}).

        A separate AdamW optimizer is used so pretraining LR is independent
        of the REINFORCE LR.  The prediction head (self._pred_head) is created
        lazily on first call.

        Args:
            replay_buffer: list of dicts with keys 'obs_sequence', 'meta'
            n_steps:       number of gradient steps
            lr_pretrain:   learning rate for the prediction head + SSM params
            seq_len:       BPTT truncation length
        """
        if len(replay_buffer) == 0:
            return

        obs_dim    = self.ssm.obs_dim
        hidden_dim = self.ssm.hidden_dim

        # Lazy-init prediction head (hidden_dim → obs_dim) and its optimizer
        if not hasattr(self, '_pred_head'):
            self._pred_head = nn.Linear(hidden_dim, obs_dim).to(self.device)
            self._pretrain_opt = optim.AdamW(
                list(self.ssm.parameters()) + list(self._pred_head.parameters()),
                lr=lr_pretrain)

        rng = np.random.default_rng()

        pretrain_log = []

        for _ in tqdm(range(n_steps)):
            # Sample episode
            ep   = replay_buffer[rng.integers(len(replay_buffer))]
            seq  = ep['obs_sequence']    # list of (raw_obs_CPU, beta_float)
            meta = ep['meta']
            T    = len(seq)
            if T < 2:
                continue

            # Sample start position; clip so we have at least 2 steps
            start = int(rng.integers(0, max(1, T - 1)))
            end   = min(start + seq_len, T - 1)   # we need obs[end] as target

            # Re-run RunningNorm from scratch up to `start` to get correct stats
            rn = RunningNorm(dim=9)
            for i in range(start):
                rn.update(seq[i][0][:9])

            # Forward pass over the subsequence
            h    = self.ssm.initial_state(device=self.device)
            loss = torch.tensor(0.0, device=self.device)
            n    = 0

            for i in range(start, end):
                raw_obs_t, beta_t = seq[i]
                raw_obs_next, _   = seq[i + 1]

                obs_t    = _normalize_obs(raw_obs_t,    rn, self.device)
                obs_next = _normalize_obs(raw_obs_next, rn, self.device)

                log_beta = torch.tensor(math.log(max(beta_t, 1e-10)),
                                        dtype=torch.float32, device=self.device)
                _, _, h = self.ssm.forward_world(obs_t, log_beta, h)
                pred    = self._pred_head(h)

                loss = loss + F.mse_loss(pred, obs_next.detach())
                n   += 1

            if n == 0:
                continue

            self._pretrain_opt.zero_grad()
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.ssm.parameters()) + list(self._pred_head.parameters()),
                max_norm=1.0)
            self._pretrain_opt.step()

            pretrain_log.append( float((loss/n).detach()) )
            #print(f'pretrain loss: {float((loss/n).detach()):.3g}')

        ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
        plt.subplots()
        plt.plot(pretrain_log)
        plt.gca().set_yscale('log')
        plt.show()
        ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##

        return pretrain_log


# ---------------------------------------------------------------------------
# run_pretrain_episode
# ---------------------------------------------------------------------------

def run_pretrain_episode(world: World, beta_source, problem_config: dict) -> dict:
    """
    Run one full episode using a fixed beta_source (not the controller).
    Populates world.obs_history and returns a replay-buffer entry.
    No gradient updates are performed.

    Args:
        world:          World instance
        beta_source:    callable (t, obs) -> float
        problem_config: dict passed to world.reset()

    Returns:
        dict with keys 'meta', 'obs_sequence', 'final_obs'
    """
    obs = world.reset(problem_config)
    for t in range(world.max_epochs):
        beta = beta_source(t, obs)
        obs  = world.step(beta)
    return {
        'meta':         world.meta.copy(),
        'obs_sequence': list(world.obs_history),   # list of (raw_obs_CPU, beta)
        'final_obs':    obs.detach().cpu(),
    }


# ---------------------------------------------------------------------------
# rl_training_loop  (thin orchestrator)
# ---------------------------------------------------------------------------

def rl_training_loop(
    world:               World,
    controller:          Controller,
    n_episodes:          int   = 30,
    # Problem meta-parameter ranges (sampled per episode)
    n_genes_choices:     list  = None,
    n_cells_choices:     list  = None,
    missing_rate_range:  tuple = (0.02, 0.15),
    # Optional pretraining
    pretrain_sources:    list  = None,   # list of beta_source callables
    n_pretrain_episodes: int   = 0,      # replay episodes to collect before RL
    pretrain_steps:      int   = 0,      # world-model gradient steps per RL episode
    pretrain_lr:         float = 1e-3,
    pretrain_seq_len:    int   = 20,
    replay_buffer:       list  = None,
    seed:                int   = 0,
) -> tuple:
    """
    Outer REINFORCE loop.

    Phase 0 (optional): collect `n_pretrain_episodes` episodes using random
        beta sources from `pretrain_sources`, then pretrain the controller's
        world model for `pretrain_steps` gradient steps.

    Phase 1: for each RL episode —
        1. Sample problem meta-parameters.
        2. world.reset(problem_config)
        3. controller.reset_state()
        4. Loop max_epochs: beta, log_prob = controller.act(obs); obs = world.step(beta)
        5. reward = world.get_reward()
        6. controller.update(log_probs, reward)
        7. Append episode to replay_buffer.
        8. Optional: controller.pretrain(replay_buffer, pretrain_steps)

    Returns:
        controller, replay_buffer, episode_log
    """
    if n_genes_choices  is None: n_genes_choices  = [100, 150, 200]
    if n_cells_choices  is None: n_cells_choices  = [1000, 1500, 2000]
    if pretrain_sources is None: pretrain_sources = BUILTIN_BETA_SOURCES
    if replay_buffer    is None: replay_buffer    = []

    rng = np.random.default_rng(seed)

    def _sample_problem():
        return {
            **world.base_problem_config,   # inherit fixed keys (n_blocks, rho_within, …)
            'n_genes':      int(rng.choice(n_genes_choices)),
            'n_cells':      int(rng.choice(n_cells_choices)),
            'missing_rate': float(rng.uniform(*missing_rate_range)),
        }

    # ------------------------------------------------------------------ #
    # Phase 0: pre-populate replay buffer with diverse beta sources
    # ------------------------------------------------------------------ #
    if n_pretrain_episodes > 0:
        print(f'\n--- Pretraining data collection: {n_pretrain_episodes} episodes ---')
        for i in range(n_pretrain_episodes):
            src    = pretrain_sources[i % len(pretrain_sources)]
            pcfg   = _sample_problem()
            entry  = run_pretrain_episode(world, src, pcfg)
            replay_buffer.append(entry)
            print(f'  pretrain ep {i+1}/{n_pretrain_episodes}  '
                  f'src={src.__name__}  n_genes={pcfg["n_genes"]}  '
                  f'missing_rate={pcfg["missing_rate"]:.3f}')

        if pretrain_steps > 0:
            print(f'  world-model pretraining: {pretrain_steps} steps ...')
            pretrain_log = controller.pretrain(replay_buffer, n_steps=pretrain_steps,
                                               lr_pretrain=pretrain_lr, seq_len=pretrain_seq_len)

    # ------------------------------------------------------------------ #
    # Phase 1: RL episodes
    # ------------------------------------------------------------------ #
    episode_log = []

    for ep in range(n_episodes):
        pcfg = _sample_problem()
        print(f'\n=== RL episode {ep+1}/{n_episodes}  '
              f'n_genes={pcfg["n_genes"]}  n_cells={pcfg["n_cells"]}  '
              f'missing_rate={pcfg["missing_rate"]:.3f} ===')

        obs = world.reset(pcfg)
        controller.reset_state()

        log_probs   = []
        epoch_betas = []

        for t in range(world.max_epochs):
            beta, log_prob = controller.act(obs)
            obs = world.step(beta)
            log_probs.append(log_prob)
            epoch_betas.append(beta)

            if (t + 1) % 20 == 0:
                m = world._metrics
                print(f'  epoch {t+1:3d}  beta={beta:.2e}  '
                      f'recon={m["recon_loss"]:.4f}  kl={m["kl_loss"]:.4f}  '
                      f'eff_K={m["effective_K"]:.1f}')

        reward, recon_term, eff_k_term = world.get_reward()
        print(f'  reward={reward:.4f}  recon_term={recon_term:.4f}  '
              f'eff_k_term={eff_k_term:.4f}  baseline={controller.baseline:.4f}')

        controller.update(log_probs, reward)

        # Add to replay buffer
        replay_buffer.append({
            'meta':         world.meta.copy(),
            'obs_sequence': list(world.obs_history),
            'final_obs':    obs.detach().cpu(),
        })

        # Optional interleaved world-model pretraining
        if pretrain_steps > 0:
            controller.pretrain(replay_buffer, n_steps=pretrain_steps,
                                lr_pretrain=pretrain_lr, seq_len=pretrain_seq_len)

        episode_log.append(dict(
            episode      = ep,
            n_genes      = pcfg['n_genes'],
            n_cells      = pcfg['n_cells'],
            missing_rate = pcfg['missing_rate'],
            ref_recon    = world.ref_recon,
            final_eff_k  = world._last_eff_k,
            recon_term   = recon_term,
            eff_k_term   = eff_k_term,
            reward       = reward,
            baseline     = controller.baseline,
            betas        = epoch_betas,
        ))

    return controller, replay_buffer, episode_log


def plot_rl_results(episode_log: list):
    """Plot reward components and beta trajectories from rl_training_loop."""
    episodes    = [e['episode']    for e in episode_log]
    rewards     = [e['reward']     for e in episode_log]
    baselines   = [e['baseline']   for e in episode_log]
    recon_terms = [e['recon_term'] for e in episode_log]
    eff_k_terms = [e['eff_k_term'] for e in episode_log]
    final_eff_ks= [e['final_eff_k'] for e in episode_log]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(episodes, rewards,     label='total reward', marker='o', ms=4)
    axes[0].plot(episodes, baselines,   label='baseline',     linestyle='--')
    axes[0].plot(episodes, recon_terms, label='recon term',   linestyle=':')
    axes[0].plot(episodes, eff_k_terms, label='eff_K term',   linestyle='-.')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('RL controller: reward components')
    axes[0].legend(fontsize=8)

    axes[1].plot(episodes, final_eff_ks, marker='o', ms=4, color='tab:orange')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('effective_K (final epoch)')
    axes[1].set_title('VampPrior latent utilization')

    for e in episode_log[-5:]:
        axes[2].plot(e['betas'], alpha=0.8,
                     label=f"ep{e['episode']} g{e['n_genes']} m{e['missing_rate']:.2f}")
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('beta (log scale)')
    axes[2].set_title('Beta trajectories (last 5 episodes)')
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('rl_controller_results.png', dpi=120)
    plt.show()

class MeansModel(nn.Module):
  """
  Model to impute genes based on the average value for each gene (baseline model for comparison)
  """
  def __init__(self,
               n_genes:int):
    super().__init__()
    self.means = nn.Parameter(torch.zeros(n_genes))

  def forward(self, x:torch.Tensor):
    if x.ndim==1:
      output = self.means  # (n_genes,)
    else:
      output = self.means.expand( x.shape[0], -1 )  # (batch_size, n_genes)

    return output, torch.ones(1), torch.zeros(1), None, None, None



# ===========================================================================
# ImputationTransformer: decoder-focused transformer for gene imputation
# ===========================================================================
#
# Architecture summary:
#   - Each gene is a token; embedding = concat(gene_id_emb, count_bin_emb, conf_bin_emb)
#   - Non-causal attention with a confidence-gated key mask: low-confidence
#     (unknown/imputed) tokens are excluded as keys so they cannot be attended to,
#     but they can still attend to all high-confidence (known) tokens.
#   - Two output heads per token: count bin logits + confidence bin logits.
#   - Confidence head is supervised by the magnitude of the count prediction error
#     (self-supervised, no external labels needed).
#   - Iterative inference: imputed values are re-entered with entropy-derived
#     confidence and the model is run again until convergence.
#
# Training (single-pass, v1):
#   - Type 1 mask (NaN values): set to unknown count token + conf bin 0.
#   - Type 2 mask (random subset of observed): same treatment; loss computed here.
#   - Count loss: cross-entropy on Type-2 masked positions only.
#   - Conf  loss: cross-entropy on Type-2 positions, target = binned |count_error|.
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. Discretization utilities
# ---------------------------------------------------------------------------

def make_log_bin_edges(n_bins: int, max_val: float = 8.5) -> torch.Tensor:
    """
    Produce n_bins+1 edges in log1p space covering [0, max_val].

    log1p(5000) ~ 8.52, so max_val=8.5 covers essentially all scRNA-seq counts.
    The edges are linearly spaced in log1p space, which gives log-spacing in
    raw count space (finer resolution at low counts).

    Returns: 1-D tensor of shape (n_bins+1,) on CPU.
    """
    return torch.linspace(0.0, max_val, n_bins + 1)


def discretize(x: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Map continuous log1p expression values to integer bin indices.

    NaN entries are assigned the special "unknown" index = len(bin_edges) - 1 = n_bins.
    Values below the lowest edge -> bin 0; above the highest -> bin n_bins-1.

    Args:
        x:         (B, G) float tensor, possibly containing NaN.
        bin_edges: (n_bins+1,) tensor of bin boundaries (from make_log_bin_edges).

    Returns:
        bins: (B, G) long tensor; values in {0, ..., n_bins-1} for observed,
              n_bins for NaN (unknown token index).
    """
    n_bins    = len(bin_edges) - 1
    nan_mask  = torch.isnan(x)

    # torch.bucketize assigns index i when bin_edges[i-1] <= x < bin_edges[i].
    # right=False: left-closed intervals; clamp to [0, n_bins-1].
    x_safe    = x.nan_to_num(0.0)
    edges_dev = bin_edges.to(x_safe.device)
    bins      = torch.bucketize(x_safe, edges_dev, right=False).clamp(0, n_bins - 1)
    bins[nan_mask] = n_bins  # unknown token
    return bins


def bins_to_midpoints(bins: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Dequantize bin indices to the midpoint of each bin (for correlation evaluation).

    Bins equal to n_bins (unknown token) are mapped to 0.0.

    Args:
        bins:      (B, G) long tensor of bin indices.
        bin_edges: (n_bins+1,) tensor of bin boundaries.

    Returns:
        (B, G) float tensor of dequantized values.
    """
    n_bins    = len(bin_edges) - 1
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # (n_bins,)
    # Clamp unknowns to 0 before indexing
    safe_bins = bins.clamp(0, n_bins - 1)
    vals      = midpoints.to(bins.device)[safe_bins]
    vals[bins == n_bins] = 0.0  # unknown positions -> 0
    return vals


def entropy_to_conf_bin(logits: torch.Tensor, n_conf_bins: int) -> torch.Tensor:
    """
    Convert count-bin logits to a confidence bin index via normalized entropy.

    Low entropy (peaked distribution)  -> high confidence bin.
    High entropy (flat distribution)   -> low confidence bin.

    Args:
        logits:      (..., n_bins) float tensor (raw, un-softmaxed).
        n_conf_bins: number of confidence bins.

    Returns:
        (...,) long tensor of confidence bin indices in {0, ..., n_conf_bins-1}.
    """
    n_bins   = logits.shape[-1]
    probs    = torch.softmax(logits, dim=-1).clamp(min=1e-9)
    entropy  = -(probs * probs.log()).sum(dim=-1)       # (...,)
    max_ent  = math.log(n_bins)                         # entropy of uniform
    norm_ent = (entropy / max_ent).clamp(0.0, 1.0)     # 0=peaked, 1=flat
    # High confidence = low entropy -> high bin index
    conf_bin = ((1.0 - norm_ent) * (n_conf_bins - 1)).long().clamp(0, n_conf_bins - 1)
    return conf_bin


def error_to_conf_bin(pred_bins: torch.Tensor,
                      true_bins: torch.Tensor,
                      n_conf_bins: int,
                      max_err_bins: int | None = None) -> torch.Tensor:
    """
    Map absolute count-bin prediction error to a confidence bin index.

    Small error -> high confidence bin; large error -> low confidence bin.
    Used to supervise the confidence output head.

    Args:
        pred_bins:    (...,) long tensor of predicted count bin indices.
        true_bins:    (...,) long tensor of true count bin indices.
        n_conf_bins:  number of confidence bins.
        max_err_bins: clip errors beyond this many bins (defaults to 4).

    Returns:
        (...,) long tensor of target confidence bin indices.
    """
    abs_err = (pred_bins.long() - true_bins.long()).abs().float()
    if max_err_bins is None:
        # Infer a reasonable cap: 4 bins is already a large error for 24 bins
        max_err_bins = 4
    norm_err = (abs_err / max_err_bins).clamp(0.0, 1.0)   # 0=perfect, 1=max_err
    conf_bin = ((1.0 - norm_err) * (n_conf_bins - 1)).long().clamp(0, n_conf_bins - 1)
    return conf_bin


# ---------------------------------------------------------------------------
# 2. GeneTokenEmbedding
# ---------------------------------------------------------------------------

class GeneTokenEmbedding(nn.Module):
    """
    Per-gene token embedding combining three information sources:
      - gene identity  (which gene)
      - count bin      (expression level, discretized)
      - confidence bin (measurement reliability)

    All three embeddings are concatenated and projected to d_model.
    """
    def __init__(self,
                 n_genes:     int,
                 n_bins:      int,
                 n_conf_bins: int,
                 d_gene:      int = 64,
                 d_count:     int = 64,
                 d_conf:      int = 32,
                 d_model:     int = 256):
        super().__init__()
        # n_bins+1 count tokens: indices 0..n_bins-1 are observed bins,
        # index n_bins is the special "unknown" token.
        self.emb_gene  = nn.Embedding(n_genes,      d_gene)
        self.emb_count = nn.Embedding(n_bins + 1,   d_count)
        self.emb_conf  = nn.Embedding(n_conf_bins,  d_conf)
        self.proj      = nn.Linear(d_gene + d_count + d_conf, d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self,
                gene_ids:   torch.Tensor,
                count_bins: torch.Tensor,
                conf_bins:  torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_ids:   (B, G) long -- gene index in {0, ..., n_genes-1}
            count_bins: (B, G) long -- count bin in {0, ..., n_bins}
            conf_bins:  (B, G) long -- confidence bin in {0, ..., n_conf_bins-1}
        Returns:
            (B, G, d_model) float
        """
        e = torch.cat([
            self.emb_gene(gene_ids),
            self.emb_count(count_bins),
            self.emb_conf(conf_bins),
        ], dim=-1)
        return self.norm(self.proj(e))


# ---------------------------------------------------------------------------
# 3. ImputationTransformer
# ---------------------------------------------------------------------------

class ImputationTransformer(nn.Module):
    """
    Decoder-focused transformer for gene expression imputation.

    Tokens are genes; the attention mask is confidence-gated:
      - Low-confidence tokens (conf_bin == 0) are masked out as keys,
        so they cannot be attended to by any other token.
      - All tokens (including low-confidence) can attend to high-confidence keys.

    This means imputed/unknown genes are invisible to the attention of other
    genes but can themselves gather information from all known genes.

    Output:
      - count_logits: (B, G, n_bins)      -- predicted count bin distribution
      - conf_logits:  (B, G, n_conf_bins) -- predicted confidence bin distribution
    """

    def __init__(self,
                 n_genes:     int,
                 n_bins:      int   = 24,
                 n_conf_bins: int   = 8,
                 d_gene:      int   = 64,
                 d_count:     int   = 64,
                 d_conf:      int   = 32,
                 d_model:     int   = 256,
                 n_heads:     int   = 8,
                 n_layers:    int   = 4,
                 dropout:     float = 0.1):
        super().__init__()

        self.n_genes     = n_genes
        self.n_bins      = n_bins
        self.n_conf_bins = n_conf_bins
        self.d_model     = d_model

        self.embedding = GeneTokenEmbedding(
            n_genes, n_bins, n_conf_bins,
            d_gene, d_count, d_conf, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,   # (B, G, d_model) convention
            norm_first      = True,   # pre-norm: more stable for deeper models
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,  # norm_first=True disables the fast path anyway
        )

        self.head_count = nn.Linear(d_model, n_bins)
        self.head_conf  = nn.Linear(d_model, n_conf_bins)

        # Fixed gene-id buffer (0..n_genes-1), registered so it moves with .to(device)
        self.register_buffer('_gene_ids',
                             torch.arange(n_genes).unsqueeze(0))  # (1, G)

    def forward(self,
                count_bins: torch.Tensor,
                conf_bins:  torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            count_bins: (B, G) long -- current count bin for each gene.
                        Unknown genes have count_bins == n_bins.
            conf_bins:  (B, G) long -- current confidence bin.
                        Unknown/low-confidence genes have conf_bins == 0.
        Returns:
            count_logits: (B, G, n_bins)
            conf_logits:  (B, G, n_conf_bins)
        """
        B, G = count_bins.shape

        gene_ids = self._gene_ids.expand(B, -1)   # (B, G)

        # Token embeddings
        tokens = self.embedding(gene_ids, count_bins, conf_bins)   # (B, G, d_model)

        # Confidence-gated key mask:
        # src_key_padding_mask: (B, G), True = this position is MASKED OUT as a key.
        # We mask out positions where conf_bin == 0 (unknown / lowest confidence).
        key_pad_mask = (conf_bins == 0)  # (B, G), bool

        # Guard against all-masked rows (would produce NaN in softmax).
        # If every gene in a sample is unknown (shouldn't happen in practice but
        # be safe), unmask everything for that sample.
        all_masked = key_pad_mask.all(dim=1)  # (B,)
        if all_masked.any():
            key_pad_mask[all_masked] = False

        out = self.transformer(tokens, src_key_padding_mask=key_pad_mask)  # (B, G, d_model)

        count_logits = self.head_count(out)   # (B, G, n_bins)
        conf_logits  = self.head_conf(out)    # (B, G, n_conf_bins)

        return count_logits, conf_logits


# ---------------------------------------------------------------------------
# 4. epoch_transformer -- training / evaluation loop
# ---------------------------------------------------------------------------

def epoch_transformer(
    model:         ImputationTransformer,
    loader:        DataLoader,
    bin_edges:     torch.Tensor,
    opt:           torch.optim.Optimizer | None = None,
    mask_fraction: float = 0.10,
    lambda_conf:   float = 0.10,
    device:        torch.device | str = 'cpu',
) -> tuple[float, float, float, float]:
    """
    One epoch of training (opt != None) or evaluation (opt == None).

    Masking strategy:
      - Type 1 (NaN in raw data):  always set to unknown token + conf bin 0.
      - Type 2 (random ~mask_fraction of *observed* positions): same treatment;
        cross-entropy loss is computed on these positions.

    Losses:
      - count_ce:  cross-entropy of predicted count bins vs. true bins,
                   averaged over Type-2 masked positions.
      - conf_ce:   cross-entropy of predicted confidence vs. error-derived target,
                   weighted by lambda_conf.

    Args:
        model:         ImputationTransformer instance.
        loader:        DataLoader yielding raw log1p tensors (B, G), possibly NaN.
        bin_edges:     (n_bins+1,) edges from make_log_bin_edges.
        opt:           optimizer (None for eval mode).
        mask_fraction: fraction of observed genes to mask for training.
        lambda_conf:   weight of the confidence auxiliary loss.
        device:        torch device.

    Returns:
        (total_loss, count_ce, conf_ce, grad_norm)  -- all floats, averaged over batches.
    """
    model.train() if opt is not None else model.eval()
    n_bins      = model.n_bins
    n_conf_bins = model.n_conf_bins

    total_loss = conf_ce_sum = count_ce_sum = grad_norm_sum = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if opt is not None else torch.no_grad()

    with ctx:
        for x_raw in loader:
            x_raw = x_raw.to(device)  # (B, G), float, may contain NaN

            # --- Discretize observed values ---
            true_count_bins = discretize(x_raw, bin_edges)  # (B, G) long
            # Confidence proxy: higher count -> higher confidence (simple heuristic).
            # Observed positions get conf bins 1..n_conf_bins-1 proportional to
            # their count level; NaN positions start at 0.
            nan_mask  = torch.isnan(x_raw)                  # (B, G) bool
            obs_count = true_count_bins.float().clamp(0, n_bins - 1)
            # Map [0, n_bins-1] -> [1, n_conf_bins-1] for observed genes
            true_conf_bins = (obs_count / (n_bins - 1) * (n_conf_bins - 2) + 1).long()
            true_conf_bins = true_conf_bins.clamp(1, n_conf_bins - 1)
            true_conf_bins[nan_mask] = 0   # unknown -> lowest confidence

            # --- Type 2 mask: random subset of observed positions ---
            type2_mask = (torch.rand_like(x_raw) < mask_fraction) & ~nan_mask  # (B, G) bool
            all_mask   = nan_mask | type2_mask                                  # (B, G) bool

            # Build model inputs: masked positions get unknown token + conf bin 0
            inp_count = true_count_bins.clone()
            inp_conf  = true_conf_bins.clone()
            inp_count[all_mask] = n_bins   # unknown token
            inp_conf[all_mask]  = 0        # lowest confidence

            # --- Forward pass ---
            count_logits, conf_logits = model(inp_count, inp_conf)
            # count_logits: (B, G, n_bins), conf_logits: (B, G, n_conf_bins)

            # --- Count cross-entropy loss (Type-2 positions only) ---
            if type2_mask.any():
                # True count bins for Type-2 positions (exclude the unknown token index)
                true_count_t2 = true_count_bins[type2_mask].clamp(0, n_bins - 1)
                pred_count_t2 = count_logits[type2_mask]          # (N2, n_bins)
                count_ce      = F.cross_entropy(pred_count_t2, true_count_t2)

                # --- Confidence auxiliary loss ---
                # Target: how accurate was the count prediction at Type-2 positions?
                with torch.no_grad():
                    pred_count_argmax = pred_count_t2.argmax(dim=-1)
                    conf_target = error_to_conf_bin(
                        pred_count_argmax, true_count_t2, n_conf_bins)
                pred_conf_t2 = conf_logits[type2_mask]            # (N2, n_conf_bins)
                conf_ce      = F.cross_entropy(pred_conf_t2, conf_target)
            else:
                count_ce = torch.tensor(0.0, device=device)
                conf_ce  = torch.tensor(0.0, device=device)

            loss = count_ce + lambda_conf * conf_ce

            if opt is not None:
                opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                grad_norm_sum += float(gn)

            total_loss   += float(loss.detach())
            count_ce_sum += float(count_ce.detach())
            conf_ce_sum  += float(conf_ce.detach())
            n_batches    += 1

    n_batches = max(n_batches, 1)
    return (total_loss   / n_batches,
            count_ce_sum / n_batches,
            conf_ce_sum  / n_batches,
            grad_norm_sum / n_batches)


# ---------------------------------------------------------------------------
# 5. impute_transformer -- iterative inference
# ---------------------------------------------------------------------------

def impute_transformer(
    model:               ImputationTransformer,
    x_raw:               torch.Tensor,
    bin_edges:           torch.Tensor,
    device:              torch.device | str = 'cpu',
    max_iters:           int   = 5,
    convergence_thresh:  float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iteratively impute missing values using the trained ImputationTransformer.

    At each iteration:
      1. Forward pass (only unknown positions are predicted).
      2. Update unknown positions with argmax of predicted count logits.
      3. Compute entropy of predicted distribution -> new confidence bin.
      4. Check convergence: fraction of unknown positions that changed bin.

    Args:
        model:              trained ImputationTransformer (eval mode).
        x_raw:              (B, G) float, NaN at missing positions.
        bin_edges:          (n_bins+1,) edges from make_log_bin_edges.
        device:             torch device.
        max_iters:          maximum number of refinement passes.
        convergence_thresh: stop when < this fraction of unknown bins change.

    Returns:
        imputed_continuous: (B, G) float -- dequantized bin midpoints for all genes.
        imputed_bins:       (B, G) long  -- final bin indices.
    """
    model.eval()
    n_bins      = model.n_bins
    n_conf_bins = model.n_conf_bins
    x_raw       = x_raw.to(device)

    nan_mask        = torch.isnan(x_raw)           # (B, G) -- positions to impute
    true_count_bins = discretize(x_raw, bin_edges)  # (B, G); NaN -> n_bins

    # Initialise: observed positions keep true bins + proportional conf; unknown = bin 0
    obs_count      = true_count_bins.float().clamp(0, n_bins - 1)
    true_conf_bins = (obs_count / (n_bins - 1) * (n_conf_bins - 2) + 1).long().clamp(1, n_conf_bins - 1)
    true_conf_bins[nan_mask] = 0

    cur_count = true_count_bins.clone()   # (B, G) long -- working count bins
    cur_conf  = true_conf_bins.clone()    # (B, G) long -- working conf bins
    # Unknown positions start with the unknown token
    cur_count[nan_mask] = n_bins
    cur_conf[nan_mask]  = 0

    with torch.no_grad():
        for _iter in range(max_iters):
            count_logits, _conf_logits = model(cur_count, cur_conf)
            # (B, G, n_bins), (B, G, n_conf_bins)

            # Predicted bin for each position
            pred_bins = count_logits.argmax(dim=-1)   # (B, G) long

            # New confidence from entropy of count predictions
            new_conf = entropy_to_conf_bin(count_logits, n_conf_bins)  # (B, G) long

            # Check convergence: fraction of unknown positions that changed
            prev_unknown_bins = cur_count[nan_mask]
            new_unknown_bins  = pred_bins[nan_mask]
            changed_frac      = (prev_unknown_bins != new_unknown_bins).float().mean().item()

            # Update only unknown positions
            cur_count[nan_mask] = pred_bins[nan_mask]
            cur_conf[nan_mask]  = new_conf[nan_mask]

            if changed_frac < convergence_thresh:
                break

    # Dequantize to continuous values for correlation evaluation
    imputed_continuous = bins_to_midpoints(cur_count, bin_edges)   # (B, G) float
    # Restore observed positions to their true continuous values
    imputed_continuous[~nan_mask] = x_raw[~nan_mask]

    return imputed_continuous, cur_count

# ---------------------------------------------------------------------------
# Unified imputation adapter
# ---------------------------------------------------------------------------

def impute(model, x_raw, frac, *, device=device):
    """Apply a type-2 mask at rate *frac* and return imputed values.

    Works for all model types without any caller-side preprocessing.

    Args:
        model   : any of GeneExpressionVAE / MoVEVAE / VampPriorVAE /
                  MeansModel / ImputationTransformer — must be in eval mode.
        x_raw   : (B, n_genes) float tensor with NaN for type-1 missing values.
        frac    : float, fraction of *observed* positions to mask for evaluation.
        device  : torch.device to run inference on.

    Returns:
        recon      : (B, n_genes) float tensor on CPU, imputed continuous values.
        type2_mask : (B, n_genes) bool tensor on CPU, positions used for scoring.
    """
    x_raw = x_raw.to(device)
    nan_mask = torch.isnan(x_raw)                          # type-1 positions

    # Draw type-2 mask: observed positions only
    type2_mask = (torch.rand_like(x_raw) < frac) & ~nan_mask   # (B, G), CPU-safe

    model.eval()
    with torch.no_grad():
        if isinstance(model, ImputationTransformer):
            # Mask type-2 positions by treating them as NaN for the transformer
            x_eval = x_raw.clone()
            x_eval[type2_mask] = float('nan')
            recon, _ = impute_transformer(
                model, x_eval, model.bin_edges,
                device=device, max_iters=5)
            recon = recon.cpu()

        elif isinstance(model, MeansModel):
            # MeansModel ignores its input; output is the per-gene mean broadcast
            # to batch size — no masking pre-processing needed.
            recon, *_ = model(x_raw.nan_to_num(0))
            recon = recon.detach().cpu()

        elif isinstance(model, VAEBase):
            # Apply get_random_mask logic manually so we control the type-2 mask
            combined_mask = type2_mask | nan_mask           # what the model sees as missing
            x_masked = _random_fill(x_raw, combined_mask)
            # same augmentation as training: noise on genuinely observed positions only
            x_masked = x_masked + torch.randn_like(x_masked) * 0.05 * (~combined_mask).float()
            x_in = torch.cat((x_masked, combined_mask.float()), dim=1)  # (B, n_genes*2)
            recon, *_ = model(x_in)
            recon = recon.detach().cpu()

        else:
            raise TypeError(f"impute(): unsupported model type {type(model).__name__}")

    return recon, type2_mask.cpu()


def evaluate_imputation(model, loader_test, mask_fraction: float, n_draws: int,
                         device=device) -> dict:
    """Unified imputation-quality metric, usable for *any* model type
    supported by impute() (VAEBase subclasses / MeansModel /
    ImputationTransformer).

    This is the single source of truth for "how good is this model's
    imputation" -- both tuning.py's Optuna objective and
    sample_efficiency.py's reported metrics call this, specifically so the
    two pipelines can't drift apart and so the resulting number is
    comparable *across model families* (unlike each model's own training
    loss -- see sample_efficiency.py's module docstring / benchmark_plan.md
    for why raw loss values are not comparable across architectures: they
    mix incommensurable units -- e.g. ImputationTransformer's categorical
    cross-entropy vs. the VAE family's Gaussian MSE + KL -- and bake in
    independently-tuned per-model loss weights like beta/gamma).

    Draws a fresh type-2 mask *n_draws* times (via impute()) over every
    batch in loader_test and pools all (true, predicted) pairs from every
    draw before computing MSE/correlation, matching how
    sample_efficiency.py's original correlation-only metric was computed.

    Args:
        model:         any model type impute() supports; must already be
                       trained (this puts it in eval mode itself).
        loader_test:   DataLoader over the held-out test set.
        mask_fraction: fraction of observed positions to hold out for
                       scoring on each draw (impute()'s `frac`).
        n_draws:       number of independent mask draws to average over --
                       a single draw is noisy (see benchmark_plan.md's
                       "Metrics" section).
        device:        torch device to run inference on.

    Returns:
        {'imputation_mse_mean':  float, 'imputation_mse_std':  float,
         'imputation_corr_mean': float, 'imputation_corr_std': float}
        -- mean/std are taken across the n_draws per-draw values (matching
        sample_efficiency.py's original semantics for the correlation
        fields), std is 0.0 if n_draws == 1.
    """
    mses, corrs = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(n_draws):
            true_all, pred_all = [], []
            for x in loader_test:
                recon, type2_mask = impute(model, x, mask_fraction, device=device)
                if type2_mask.any():
                    true_all.append(x[type2_mask].numpy())
                    pred_all.append(recon[type2_mask].numpy())
            if true_all:
                true_vals = np.concatenate(true_all)
                pred_vals = np.concatenate(pred_all)
                mses.append(float(np.mean((pred_vals - true_vals) ** 2)))
                corrs.append(np.corrcoef(true_vals, pred_vals)[0, 1] if len(true_vals) > 1 else float("nan"))
            else:
                mses.append(float("nan"))
                corrs.append(float("nan"))

    return {
        "imputation_mse_mean":  float(np.nanmean(mses)),
        "imputation_mse_std":   float(np.nanstd(mses)),
        "imputation_corr_mean": float(np.nanmean(corrs)),
        "imputation_corr_std":  float(np.nanstd(corrs)),
    }


# ---------------------------------------------------------------------------
# Per-cell / per-class diagnostics
#
# Built on top of impute() to support grouping per-cell imputation error (and
# DEQ/VampPrior internal diagnostics) by an external class/cluster label, to
# investigate questions like: "is DEQEncoderVampVAE's higher variance in
# imputation error across classes coming from uneven VampPrior mixture
# coverage, uneven DEQ fixed-point convergence, or something else?"
# ---------------------------------------------------------------------------

def per_cell_masked_error(x_raw: torch.Tensor,
                           recon: torch.Tensor,
                           type2_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean squared imputation error per cell, over that cell's type-2-masked
    (deliberately held-out, ground-truth-known) genes only.

    Args:
        x_raw, recon, type2_mask : as returned by / passed to impute() --
            (B, n_genes), aligned row-for-row. x_raw may contain NaN
            (type-1/genuinely-missing positions); these are never counted
            here because type2_mask is constructed to exclude them already
            (see impute()).
    Returns:
        (B,) float tensor, CPU. NaN for any cell with zero masked genes in
        this draw (can happen at very low mask fractions/small n_genes).
    """
    x_raw  = x_raw.detach().cpu()
    recon  = recon.detach().cpu()
    type2_mask = type2_mask.detach().cpu()

    # NaN-safety: positions outside type2_mask can be NaN in x_raw (genuine
    # type-1 missing values), and 0 * NaN == NaN in IEEE754 -- so a plain
    # (sq_err * type2_mask.float()) would silently turn every row that has
    # *any* type-1-missing gene into an all-NaN row, even though those
    # positions are supposed to be excluded. nan_to_num + torch.where keeps
    # the excluded positions from ever contributing/propagating NaN.
    sq_err = (recon - x_raw.nan_to_num(0)).pow(2)
    sq_err = torch.where(type2_mask, sq_err, torch.zeros_like(sq_err))
    n_masked = type2_mask.sum(dim=1).float()
    per_cell = sq_err.sum(dim=1) / n_masked.clamp(min=1)
    per_cell[n_masked == 0] = float('nan')
    return per_cell


def collect_deq_diagnostics(model: nn.Module) -> dict:
    """
    Snapshot the DEQ / VampPrior per-sample diagnostics currently held on
    *model*, populated as a side effect of its most recent forward() call
    (e.g. the call made inside impute()). Call this immediately after
    impute()/model(...) and before any other forward pass on the same model,
    since some of these attributes are overwritten on every call (see
    DEQEncoderVampVAE's Subtlety #4).

    Safe to call on any model type in this file -- entries are None if the
    model doesn't have that diagnostic (e.g. GeneExpressionVAE has no
    deq_cell; VampPriorVAE without a DEQ encoder has no deq_cell either).

    Returns a dict with keys:
        'deq_iters'               : int  or None -- whole-batch iteration count
        'deq_residual'            : float or None -- whole-batch final residual
        'deq_residual_per_sample' : (B,) tensor or None
        'deq_z_star'              : (B, latent_dim) tensor or None -- pre-fc_mu
                                     fixed point, for saturation/range checks
        'effective_K'             : float or None -- VampPrior mixture usage,
                                     whole-batch average
        'effective_K_per_sample'  : (B,) tensor or None
    """
    out = {
        'deq_iters':               None,
        'deq_residual':             None,
        'deq_residual_per_sample': None,
        'deq_z_star':               None,
        'effective_K':              None,
        'effective_K_per_sample':  None,
    }

    if isinstance(model, DEQEncoderVampVAE):
        # Real-batch-specific snapshots taken in forward(), NOT deq_cell's
        # raw attributes -- those get overwritten by the pseudo-input pass
        # triggered inside vampprior_kl() (Subtlety #4).
        out['deq_iters']               = model.last_batch_iters
        out['deq_residual']            = model.last_batch_residual
        out['deq_residual_per_sample'] = model.last_batch_residual_per_sample
        out['deq_z_star']              = model.last_batch_z_star
    else:
        deq_cell = getattr(model, 'deq_cell', None)
        if deq_cell is not None:
            out['deq_iters']               = deq_cell.last_forward_iters
            out['deq_residual']            = deq_cell.last_forward_residual
            out['deq_residual_per_sample'] = deq_cell.last_forward_residual_per_sample
            out['deq_z_star']              = deq_cell.last_z_star

    if isinstance(model, VampPriorVAE):
        effk = getattr(model, '_effective_K', None)
        out['effective_K'] = float(effk) if effk is not None else None
        out['effective_K_per_sample'] = getattr(model, '_effective_K_per_sample', None)

    return out


def _scalar_or_none(x) -> float|None:
    """torch scalar tensor (possibly NaN) or None -> plain float or None."""
    if x is None:
        return None
    return float(x.item()) if torch.is_tensor(x) else float(x)


def collect_vamp_diagnostics(model: nn.Module) -> dict:
    """
    Snapshot VampPriorVAE's collapse-related diagnostics into plain floats,
    populated as a side effect of its most recent forward() call. Designed
    to be called once per epoch (e.g. right after epoch_vae(...)) and
    logged, to track whether the anti-collapse regularizers are actually
    working over the course of training -- see VampPriorVAE's class
    docstring and vampprior_kl's inline comments for what each field
    distinguishes.

    Safe to call on any model type -- returns a dict of Nones if *model* is
    not a VampPriorVAE (or subclass, e.g. DEQEncoderVampVAE).

    Returns a dict with keys:
        'effective_K'                       : float or None -- PER-SAMPLE
            mixture usage, averaged over the last training batch seen.
            IMPORTANT: reads exactly 1.0 even in the theoretical best case
            (see VampPriorVAE class docstring, "effective_K vs
            effective_K_population") -- near 1.0 here is expected/routine
            in a high-dimensional latent space, NOT necessarily collapse.
            Use 'effective_K_population' below to actually judge collapse.
        'effective_K_population'            : float or None -- POPULATION
            mixture usage (entropy computed on the batch-AVERAGED
            responsibility distribution, not averaged per-sample entropies).
            1.0 = true collapse (one pseudo-input absorbs the whole batch),
            up to n_pseudo = every component gets equal aggregate use. This
            is the metric that answers whether the regularizers are
            actually fixing collapse.
        'diversity_loss'                     : float or None -- regularizer 1's
            raw value (population responsibility-entropy term; more negative
            == more uniform aggregate usage, 0 == fully collapsed)
        'repulsion_loss'                     : float or None -- regularizer 2's
            raw value (0 once all pseudo-input pairs clear the margin)
        'repulsion_typical_scale'            : float or None -- real batch's
            own mu pairwise-distance median (the margin's raw ingredient --
            watch for this shrinking over training, see Hypothesis 1)
        'repulsion_margin'                   : float or None -- the derived
            threshold actually used this step (typical_scale * frac)
        'pu_mu_pairwise_median'              : float or None -- what the
            pseudo-inputs actually achieve post-encoding; compare against
            repulsion_margin (gap closing/flat/widening tells you whether
            the regularizer is winning)
        'pseudo_inputs_raw_pairwise_median'  : float or None -- pre-encoding,
            raw gene-expression-space spread of the pseudo_inputs parameters
            themselves; compare against pu_mu_pairwise_median to see whether
            any collapse originates in the raw parameters or is introduced
            by the shared encoder/DEQ mapping them together regardless.
        'coverage_loss'                      : float or None -- regularizer 3's
            raw value (normalized mean squared nearest-distance; shrinks as
            pseudo-inputs land closer to real data).
        'coverage_mean_nearest_dist'          : float or None -- mean (over K
            pseudo-inputs) distance to nearest real-batch cell, in raw
            latent units (unnormalized, unlike coverage_loss) -- should
            trend toward 0 as coverage improves; the actual diagnostic to
            watch to confirm regularizer 3 is fixing effective_K collapse.
    """
    out = {
        'effective_K':                      None,
        'effective_K_population':           None,
        'diversity_loss':                   None,
        'repulsion_loss':                   None,
        'repulsion_typical_scale':          None,
        'repulsion_margin':                 None,
        'pu_mu_pairwise_median':            None,
        'pseudo_inputs_raw_pairwise_median': None,
        'coverage_loss':                    None,
        'coverage_mean_nearest_dist':       None,
    }
    if not isinstance(model, VampPriorVAE):
        return out

    out['effective_K']                      = _scalar_or_none(getattr(model, '_effective_K', None))
    out['effective_K_population']           = _scalar_or_none(getattr(model, '_effective_K_population', None))
    out['diversity_loss']                   = _scalar_or_none(getattr(model, '_vamp_diversity_loss', None))
    out['repulsion_loss']                   = _scalar_or_none(getattr(model, '_vamp_repulsion_loss', None))
    out['repulsion_typical_scale']          = _scalar_or_none(getattr(model, '_vamp_repulsion_typical_scale', None))
    out['repulsion_margin']                 = _scalar_or_none(getattr(model, '_vamp_repulsion_margin', None))
    out['pu_mu_pairwise_median']            = _scalar_or_none(getattr(model, '_vamp_pu_mu_pairwise_median', None))
    out['pseudo_inputs_raw_pairwise_median']= _scalar_or_none(getattr(model, '_vamp_pseudo_inputs_raw_pairwise_median', None))
    out['coverage_loss']                    = _scalar_or_none(getattr(model, '_vamp_coverage_loss', None))
    out['coverage_mean_nearest_dist']       = _scalar_or_none(getattr(model, '_vamp_coverage_mean_nearest_dist', None))

    return out


def standard_gaussian_kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Per-cell analytic KL( q(z|x) || N(0,I) ) for a standard-prior model
    (GeneExpressionVAE, DEQEncoderVAE) -- the direct counterpart of
    VampPriorVAE's self._vamp_kl_per_sample, computed externally since
    mu/logvar are already returned by forward() for every model in this
    file (no model changes needed to use this).

    NOTE: intentionally does NOT apply the free-bits floor
    (`clamp(min=free_bits)`) that masked_loss applies during training --
    this is meant to measure each cell's actual/raw KL cost for comparison
    against VampPriorVAE's self._vamp_kl_per_sample (which has no
    analogous per-dimension floor either, see masked_loss's kl_override
    branch), not to reproduce the training loss exactly.

    Args:
        mu, logvar : (B, D), as returned by encode()/forward().
    Returns:
        (B,) tensor -- per-cell KL cost, in the same units/scale as
        self._vamp_kl_per_sample, so class_grouped_stats' between_class_variance
        is directly comparable across a VampPrior model and a standard-prior
        model at matched latent_dim.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
    return kl_per_dim.sum(dim=1)


def analyze_kl_per_sample(models: list, x: torch.Tensor,
                           mask_fraction: float = 0.05, *,
                           device=device) -> dict:
    """
    Run a single forward pass of `x` through each model in `models` and
    extract each model's per-cell KL cost, so a VampPrior model and a
    standard-N(0,I)-prior model can be compared on equal footing (same
    input batch, same masking draw, same units) via class_grouped_stats.

    Uses the same masking/fill/mask-channel preprocessing as impute() (and
    training), so every model sees input in the format/distribution it was
    trained on -- no caller-side preprocessing needed.

    Args:
        models        : list of trained model instances (any mix of
                         VampPriorVAE-family models -- VampPriorVAE,
                         DEQEncoderVampVAE -- and standard-prior VAEBase
                         models -- GeneExpressionVAE, DEQEncoderVAE, etc.
                         Each model is `eval()`'d internally.
        x             : (B, n_genes) raw gene-expression tensor, may
                         contain NaN for type-1 missing values (same
                         convention as impute()'s x_raw).
        mask_fraction : fraction of observed positions to additionally mask
                         out (get_random_mask-style type-2 masking), same
                         convention/typical value as training.
        device        : torch.device to run inference on.

    Returns:
        dict keyed by f"{type(model).__name__}#{i}" (i = models list index,
        disambiguates multiple instances of the same class), each value a
        (B,) CPU tensor of per-cell KL cost:
          - VampPriorVAE-family: model._vamp_kl_per_sample (log_q - log_p)
          - everything else (assumed standard N(0,I) prior):
            standard_gaussian_kl_per_sample(mu, logvar)

    Usage:
        kl_by_model = analyze_kl_per_sample([vamp_model, baseline_model], x)
        for name, kl in kl_by_model.items():
            per_class_means, between_class_var = class_grouped_stats(kl, labels)
            print(name, between_class_var)
    """
    x = x.to(device)
    nan_mask = torch.isnan(x)                                   # type-1 positions
    type2_mask = (torch.rand_like(x) < mask_fraction) & ~nan_mask
    combined_mask = type2_mask | nan_mask                       # everything the model sees as missing

    x_masked = _random_fill(x, combined_mask)
    x_masked = x_masked + torch.randn_like(x_masked) * 0.05 * (~combined_mask).float()
    x_in = torch.cat((x_masked, combined_mask.float()), dim=1)  # (B, n_genes*2)

    out = {}
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            _, mu, logvar, *_ = model(x_in)

        if isinstance(model, VampPriorVAE):
            kl_per_sample = model._vamp_kl_per_sample
        else:
            kl_per_sample = standard_gaussian_kl_per_sample(mu, logvar)

        out[f"{type(model).__name__}#{i}"] = kl_per_sample.detach().cpu()

    return out


def class_grouped_stats(values: torch.Tensor, labels) -> tuple[dict, float]:
    """
    Group a per-cell diagnostic (e.g. per_cell_masked_error(...) output, or
    collect_deq_diagnostics(...)['effective_K_per_sample']) by an external
    class/cluster label and compute the between-class variance of the
    per-class means -- i.e. the metric this whole investigation is about.

    Args:
        values : (N,) tensor or array-like, may contain NaN (excluded via
                  nanmean within each class).
        labels : (N,) array-like of class ids (int, str, whatever is
                  hashable/sortable via np.unique), aligned row-for-row
                  with values.
    Returns:
        (per_class_means, between_class_variance)
        per_class_means         : {label: float} mean value within each class
        between_class_variance  : float, variance across those per-class
                                   means (population variance, ddof=0)
    """
    values_np = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    labels_np = np.asarray(labels)
    assert values_np.shape[0] == labels_np.shape[0], \
        f"values ({values_np.shape[0]}) and labels ({labels_np.shape[0]}) must align"

    per_class_means = {}
    for lbl in np.unique(labels_np):
        vals = values_np[labels_np == lbl]
        per_class_means[lbl] = float(np.nanmean(vals)) if len(vals) else float('nan')

    between_class_variance = float(np.var(list(per_class_means.values())))
    return per_class_means, between_class_variance


def vamp_usage_by_class(model: nn.Module, labels) -> dict:
    """
    Test whether VampPriorVAE's mixture-usage diversity is aligned with an
    external class/cell-type label, or merely diverse in the aggregate
    irrespective of class (see VampPriorVAE class docstring's note on
    self._vamp_resp).

    Motivation: a healthy self._effective_K_population (e.g. ~40 out of 50)
    confirms the mixture is used diversely *in aggregate*, but says nothing
    about *which* cells route to *which* pseudo-inputs. It's consistent with
    two very different scenarios: (a) different classes are routed to
    different pseudo-inputs (class-aligned specialization -- the mechanism
    that would predict more *consistent* per-class reconstruction error vs.
    a single global N(0,I) prior), or (b) every class independently spreads
    across ~the same broad set of components (diversity with no relation to
    class identity -- which predicts no change in per-class error variance
    relative to a standard prior, despite aggregate diversity being
    perfectly healthy). This function distinguishes the two.

    Populated as a side effect of the model's most recent forward() call
    (self._vamp_resp, set in vampprior_kl). Call this right after a forward
    pass over a batch/eval-set whose row order matches `labels`, same
    convention as class_grouped_stats. Ideally called over an
    accumulated/full eval pass rather than a single small training
    minibatch -- per-class responsibility averages computed from only a
    handful of cells per class (e.g. batch_size=200 split across 8 classes)
    will be noisy.

    Args:
        model  : any model instance; safe to call on non-VampPriorVAE models
                  (returns a dict of empty/None entries, see below).
        labels : (B,) array-like of class ids, aligned row-for-row with the
                  batch that produced model._vamp_resp.

    Returns a dict with keys:
        'per_class_avg_resp'               : {label: (K,) np.ndarray} -- mean
            responsibility vector for cells of that class.
        'per_class_effective_K_population' : {label: float} -- the SAME
            exp(entropy(.)) metric as self._effective_K_population, computed
            on that one class's avg_resp instead of the whole batch's.
            Directly comparable to 'global_effective_K_population' below.
        'global_effective_K_population'    : float or None -- recomputed
            from the same batch's full avg_resp, for reference (should
            match self._effective_K_population from the same forward pass,
            up to floating point).
        'class_pairwise_js_divergence'     : {(label_a, label_b): float} --
            Jensen-Shannon divergence (nats) between each pair of classes'
            avg_resp distributions. ~0 means the two classes use the
            mixture identically; higher means they route to different
            pseudo-inputs.

    Interpretation:
        If every class's per_class_effective_K_population is close to
        global_effective_K_population (and pairwise JS divergences are all
        small/near 0), the mixture's aggregate diversity is NOT
        class-aligned -- explaining a null result on per-class
        reconstruction error variance despite healthy aggregate diversity.

        If per-class values are much LOWER than the global value (each
        class concentrating on a distinct handful of components) and/or
        pairwise JS divergences are large, class-aligned specialization
        does exist -- in which case a null result on per-class error
        variance would need a different explanation (e.g. the weak
        effective beta / latent_dim-mismatch confounds noted alongside
        this investigation).
    """
    out = {
        'per_class_avg_resp':               {},
        'per_class_effective_K_population': {},
        'global_effective_K_population':    None,
        'class_pairwise_js_divergence':     {},
    }
    if not isinstance(model, VampPriorVAE):
        return out

    resp = getattr(model, '_vamp_resp', None)
    if resp is None:
        return out

    resp_np   = resp.detach().cpu().numpy()  # (B, K)
    labels_np = np.asarray(labels)
    assert resp_np.shape[0] == labels_np.shape[0], \
        f"resp ({resp_np.shape[0]}) and labels ({labels_np.shape[0]}) must align"

    # float64 throughout: avg_resp arrays originate from a torch tensor that
    # may be float32, and per-sample responsibility in a high-dimensional
    # (latent_dim=32) mixture is often extremely peaked -- many components'
    # probability mass can be a tiny (near-subnormal) float32 value. Halving
    # such a value while forming the JS midpoint below can round/flush to
    # exactly 0.0 on some numpy/BLAS builds even though it's mathematically
    # nonzero, which -- combined with the missing epsilon guard that used to
    # be here -- produced spurious `inf` JS-divergence values (confirmed:
    # JS divergence between two valid probability distributions is always
    # bounded by ln(2), so `inf` can only be a numerical artifact, never a
    # real value). float64 promotion plus the epsilon guard below closes
    # this off directly regardless of the exact underlying rounding cause.
    _EPS = 1e-20

    def _entropy_effK(p: np.ndarray) -> float:
        p = p.astype(np.float64)
        p = p / p.sum()
        ent = -(p * np.log(p + _EPS)).sum()
        return float(np.exp(ent))

    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        p = p.astype(np.float64); p = p / p.sum()
        q = q.astype(np.float64); q = q / q.sum()
        m = 0.5 * (p + q)
        def _kl(a, b):
            mask = a > 0
            # epsilon guard on both numerator and denominator: guarantees a
            # finite result (KL/JS between valid probability distributions
            # is mathematically always finite) even if b[mask] rounds
            # arbitrarily close to (or, pre-fix, exactly) 0.
            return float((a[mask] * np.log((a[mask] + _EPS) / (b[mask] + _EPS))).sum())
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    out['global_effective_K_population'] = _entropy_effK(resp_np.mean(axis=0))

    unique_labels = np.unique(labels_np)
    for lbl in unique_labels:
        avg = resp_np[labels_np == lbl].mean(axis=0)  # (K,)
        out['per_class_avg_resp'][lbl]               = avg
        out['per_class_effective_K_population'][lbl] = _entropy_effK(avg)

    for i, lbl_a in enumerate(unique_labels):
        for lbl_b in unique_labels[i + 1:]:
            out['class_pairwise_js_divergence'][(lbl_a, lbl_b)] = _js_divergence(
                out['per_class_avg_resp'][lbl_a], out['per_class_avg_resp'][lbl_b])

    return out


# ---------------------------------------------------------------------------
# Model factories
# One callable per named architecture. Each accepts:
#   n_genes : int            -- data-dependent, not a hyperparameter
#   config  : dict | None    -- architecture hyperparameter overrides
# and returns an *un-trained* model instance (not yet moved to device).
#
# Each factory's default hyperparameters live in a named *_CONFIG dict right
# above it -- the single source of truth for that architecture's tunables,
# and the shape a future JSON config file would need to match. `config`, if
# given, only needs to specify the keys it wants to change: it is merged
# over the defaults (`{**DEFAULT_CONFIG, **(config or {})}`) and splatted
# into the model constructor as **kwargs, so no per-key wiring is needed here
# and unknown keys surface naturally as a TypeError from the constructor.
# The resolved dict is stored on the returned model as `model.config`, so a
# model's exact architecture is inspectable/self-documenting after the fact
# (e.g. when reloaded from a checkpoint).
# ---------------------------------------------------------------------------

GENE_EXPRESSION_VAE_CONFIG = {
    'latent_dim':   32,
    'dropout':      0.02,
    'encoder_dims': [512, 256, 128],
    'decoder_dims': [128, 256, 256],
}

def _make_gene_vae(n_genes:int, config:dict|None=None) -> GeneExpressionVAE:
    cfg = {**GENE_EXPRESSION_VAE_CONFIG, **(config or {})}
    model = GeneExpressionVAE(n_genes=n_genes, **cfg)
    model.config = cfg
    return model


DEQ_ENCODER_VAE_CONFIG = {
    # Matched encoder/decoder capacity to GENE_EXPRESSION_VAE_CONFIG for a
    # direct, single-variable (encoder architecture) A/B comparison.
    'latent_dim':   32,
    'dropout':      0.02,
    'encoder_dims': [512, 256, 128],
    'decoder_dims': [128, 256, 256],
    'coeff':        0.9,
    'max_iter':     30,
    'tol':          1e-4,
}

def _make_deq_encoder_vae(n_genes:int, config:dict|None=None) -> DEQEncoderVAE:
    cfg = {**DEQ_ENCODER_VAE_CONFIG, **(config or {})}
    model = DEQEncoderVAE(n_genes=n_genes, **cfg)
    model.config = cfg
    return model


MOVE_VAE_K3_CONFIG = {
    'latent_dim':     30,
    'n_components':   3,
    'gating_net_dim': 32,
    'dropout':        0.05,
    'encoder_dims':   [200, 164, 96],
    'decoder_dims':   [64, 128],
}

def _make_move_vae_k3(n_genes:int, config:dict|None=None) -> MoVEVAE:
    cfg = {**MOVE_VAE_K3_CONFIG, **(config or {})}
    model = MoVEVAE(n_genes=n_genes, **cfg)
    model.config = cfg
    return model


MOVE_VAE_K1_CONFIG = {
    'latent_dim':     80,
    'n_components':   1,
    'gating_net_dim': 32,
    'dropout':        0.05,
    'encoder_dims':   [512, 256, 128],
    'decoder_dims':   [128, 200, 200],
}

def _make_move_vae_k1(n_genes:int, config:dict|None=None) -> MoVEVAE:
    cfg = {**MOVE_VAE_K1_CONFIG, **(config or {})}
    model = MoVEVAE(n_genes=n_genes, **cfg)
    model.config = cfg
    return model


VAMP_PRIOR_VAE_CONFIG = {
    'latent_dim':            24,
    'n_pseudo':              50,
    'dropout':               0.02,
    'encoder_dims':          [512, 256],
    'decoder_dims':          [200, 200],
    'repulsion_margin_frac': 0.5,
}

def _make_vamp_vae(n_genes:int, config:dict|None=None,
                    pseudo_init_samples:torch.Tensor|None=None) -> VampPriorVAE:
    # pseudo_init_samples is data-dependent (sampled from a real batch at
    # construction time) and not JSON-serializable, so it stays a separate
    # runtime kwarg rather than part of the config dict. None -> random
    # pseudo-inputs; a loaded state_dict will overwrite them with the saved
    # values regardless.
    cfg = {**VAMP_PRIOR_VAE_CONFIG, **(config or {})}
    model = VampPriorVAE(n_genes=n_genes, pseudo_init_samples=pseudo_init_samples, **cfg)
    model.config = cfg
    return model


DEQ_ENCODER_VAMP_VAE_CONFIG = {
    # encoder_dims/decoder_dims/latent_dim/coeff/max_iter/tol match
    # DEQ_ENCODER_VAE_CONFIG so DEQEncoderVAE vs DEQEncoderVampVAE isolates
    # only the prior (Gaussian vs Vamp); n_pseudo matches
    # VAMP_PRIOR_VAE_CONFIG so VampPriorVAE vs DEQEncoderVampVAE isolates
    # only the encoder (MLP vs DEQ).
    'latent_dim':            32,
    'n_pseudo':              50,
    'dropout':               0.02,
    'encoder_dims':          [512, 256, 128],
    'decoder_dims':          [128, 256, 256],
    'coeff':                 0.9,
    'max_iter':              30,
    'tol':                   1e-4,
    'repulsion_margin_frac': 0.5,
}

def _make_deq_vamp_vae(n_genes:int, config:dict|None=None,
                        pseudo_init_samples:torch.Tensor|None=None) -> DEQEncoderVampVAE:
    cfg = {**DEQ_ENCODER_VAMP_VAE_CONFIG, **(config or {})}
    model = DEQEncoderVampVAE(n_genes=n_genes, pseudo_init_samples=pseudo_init_samples, **cfg)
    model.config = cfg
    return model


MEANS_MODEL_CONFIG = {}  # no architecture hyperparameters beyond n_genes

def _make_means_model(n_genes:int, config:dict|None=None) -> MeansModel:
    cfg = {**MEANS_MODEL_CONFIG, **(config or {})}
    model = MeansModel(n_genes=n_genes, **cfg)
    model.config = cfg
    return model


# Architecture hyperparameters (small config for ~200 genes) -- consumed by
# _make_imputation_transformer/model_factories, not by the training loop.
IMPUTATION_TRANSFORMER_CONFIG = {
    'n_bins':      24,
    'n_conf_bins': 8,
    'd_gene':      64,
    'd_count':     64,
    'd_conf':      32,
    'd_model':     128,   # smaller than full 256 for 200-gene scale
    'n_heads':     4,
    'n_layers':    3,
    'dropout':     0.1,
}

# Training-loop hyperparameters (not part of the model's architecture config
# -- consumed by epoch_transformer / the training script, not the factory).
it_lr           = 3e-4
it_batch_size   = 128
it_max_epochs   = 100
it_mask_frac    = 0.10
it_lambda_conf  = 0.10

# Bin edges -- shared between training and evaluation
it_bin_edges = make_log_bin_edges(IMPUTATION_TRANSFORMER_CONFIG['n_bins'], max_val=8.5)

def _make_imputation_transformer(n_genes:int, config:dict|None=None) -> ImputationTransformer:
    cfg = {**IMPUTATION_TRANSFORMER_CONFIG, **(config or {})}
    model = ImputationTransformer(n_genes=n_genes, **cfg)
    model.config = cfg
    return model

model_factories = {
    'MeansModel':           _make_means_model,
    'GeneExpressionVAE':    _make_gene_vae,
    'DEQEncoderVAE':        _make_deq_encoder_vae,
    'MoVEVAE_K3':           _make_move_vae_k3,
    'MoVEVAE_K1':           _make_move_vae_k1,
    'VampPriorVAE':         _make_vamp_vae,
    'DEQEncoderVampVAE':    _make_deq_vamp_vae,
    'ImputationTransformer':_make_imputation_transformer,
}

