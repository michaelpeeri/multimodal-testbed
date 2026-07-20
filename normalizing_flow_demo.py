"""
Normalizing-flow demo
======================

Self-contained demonstration:
  1. Define a multimodal target distribution in R^N (a Gaussian mixture with
     `n_modes` components scattered in N-dimensional space).
  2. Draw a training/test set of samples from it.
  3. Fit *two* normalizing-flow architectures to the samples via maximum
     likelihood (i.e. minimize -mean(log p_model(x))):
       (A) RealNVP -- an explicit stack of affine coupling layers.
       (B) A Deep-Equilibrium (DEQ) flow -- an implicit "infinite depth"
           residual layer whose output is the fixed point of
           z* = x + g(z*), trained with implicit differentiation instead
           of unrolling the fixed-point solver.
  4. Compare both learned models against the ground truth and against each
     other: loss curves, log-likelihood on held-out data, and (for the
     first two dimensions) a scatter-plot comparison of real vs.
     flow-generated samples.

This file has no dependency on the rest of the repo -- it only needs
torch, numpy, matplotlib and tqdm.
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import (
    Normal, Independent, Categorical, MixtureSameFamily,
)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. Multimodal target distribution in R^N
# ---------------------------------------------------------------------------

N        = 6     # dimensionality of the space
N_MODES  = 5      # number of mixture components
RADIUS   = 6.0     # how far apart the mode centers are, roughly
MODE_STD = 0.6     # per-mode (diagonal) standard deviation


def make_target_distribution(n=N, n_modes=N_MODES, radius=RADIUS, mode_std=MODE_STD):
    """Build a Gaussian-mixture target distribution living in R^n."""
    means   = torch.randn(n_modes, n) * radius / math.sqrt(n)
    stds    = torch.full((n_modes, n), mode_std)
    weights = torch.rand(n_modes)
    weights = weights / weights.sum()

    components = Independent(Normal(means, stds), 1)          # batch=n_modes, event=n
    mixture    = Categorical(probs=weights)
    return MixtureSameFamily(mixture, components)


target_dist = make_target_distribution()

# ---------------------------------------------------------------------------
# 2. Sample training / test data from the target
# ---------------------------------------------------------------------------

N_TRAIN = 8_000
N_TEST  = 2_000

with torch.no_grad():
    data_train = target_dist.sample((N_TRAIN,)).to(device)
    data_test  = target_dist.sample((N_TEST,)).to(device)

loader_train = DataLoader(TensorDataset(data_train), batch_size=256, shuffle=True)

# ---------------------------------------------------------------------------
# 3. RealNVP normalizing flow
# ---------------------------------------------------------------------------
#
# Each affine-coupling layer splits the N dimensions with a fixed binary
# mask. The masked-in half is passed through small MLPs to produce a scale
# (s) and shift (t) that are applied to the masked-out half:
#
#     z_masked_out = x_masked_out * exp(s(x_masked_in)) + t(x_masked_in)
#
# This is invertible in closed form and has a triangular Jacobian, so
# log|det dz/dx| = sum(s). Alternating the mask across layers lets every
# dimension eventually get transformed.


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer("mask", mask)

        def make_net(out_activation=None):
            layers = [
                nn.Linear(dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            ]
            if out_activation is not None:
                layers.append(out_activation)
            return nn.Sequential(*layers)

        self.net_s = make_net(nn.Tanh())   # bounded scale for training stability
        self.net_t = make_net()

    def forward(self, x):
        """x -> z. Returns (z, log_det_dz_dx)."""
        x_id = x * self.mask
        s = self.net_s(x_id) * (1 - self.mask)
        t = self.net_t(x_id) * (1 - self.mask)
        z = x_id + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        """z -> x (no log-det needed for sampling)."""
        z_id = z * self.mask
        s = self.net_s(z_id) * (1 - self.mask)
        t = self.net_t(z_id) * (1 - self.mask)
        x = z_id + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x


class RealNVP(nn.Module):
    def __init__(self, dim, n_layers=8, hidden_dim=128):
        super().__init__()
        layers = []
        for i in range(n_layers):
            # checkerboard mask, alternating parity each layer
            mask = torch.arange(dim) % 2
            if i % 2 == 1:
                mask = 1 - mask
            layers.append(AffineCoupling(dim, hidden_dim, mask.float()))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("base_loc",   torch.zeros(dim))
        self.register_buffer("base_scale", torch.ones(dim) )

    @property
    def base_dist(self):
        return Independent(Normal(self.base_loc, self.base_scale), 1)

    def forward(self, x):
        """x -> z, accumulating total log|det dz/dx|."""
        log_det_total = x.new_zeros(x.shape[0])
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return self.base_dist.log_prob(z) + log_det

    def sample(self, n):
        z = self.base_dist.sample((n,))
        return self.inverse(z)


flow_realnvp = RealNVP(dim=N, n_layers=8, hidden_dim=128).to(device)

# ---------------------------------------------------------------------------
# 3b. DEQ-based flow (Deep Equilibrium Model)
# ---------------------------------------------------------------------------
#
# Instead of stacking many *explicit* invertible layers (as RealNVP does),
# a Deep Equilibrium (DEQ) layer represents an "infinite-depth" weight-tied
# network implicitly, as the fixed point of a single update rule. Here each
# block defines an invertible residual transform:
#
#     z* solves   z* = x + g(z*)        (forward: x -> z, used for log_prob)
#     x         =    z  - g(z)          (inverse: z -> x, used for sampling)
#
# where g is constrained to be a contraction (Lipschitz constant < 1). This
# guarantees:
#   - the forward fixed point exists and is unique (Banach fixed-point
#     theorem), and can be found by plain repeated substitution
#     (Picard iteration): z_{k+1} = x + g(z_k);
#   - the inverse is *explicit* (no iteration needed) -- exactly the
#     invertible-residual-network (i-ResNet) trick;
#   - the Jacobian dz/dx = (I - dg/dz)^-1 is well defined, so
#     log|det(dz/dx)| = -log|det(I - dg/dz)|.
#
# Because the fixed point is needed on *every* training step (it's what
# feeds log_prob), we backprop through the solver using the implicit
# function theorem (the defining DEQ trick from Bai et al., 2019) instead
# of unrolling every solver iteration -- this keeps backward-pass cost
# independent of how many iterations the forward solve needed.
#
# Note on the log-determinant gradient: for simplicity we compute the local
# Jacobian dg/dz needed for log|det(...)| at a *detached* copy of the fixed
# point. This gives the exact log-det *value*, and correct gradients into
# g's parameters through their direct effect on dg/dz, but it drops the
# (typically small, second-order) correction term coming from how the
# equilibrium point itself shifts as parameters change. Rigorously handling
# that term is possible (see "Implicit Normalizing Flows", Lu et al. 2021)
# but adds real complexity for little benefit in a demo like this one.
# Also note: because dg/dz here is computed exactly via N backward passes,
# this does *not* scale to very high-dimensional problems -- at large N one
# would instead use the stochastic power-series trace estimator from
# Residual Flows (Chen et al., 2019).


def fixed_point_iterate(f, z0, max_iter=50, tol=1e-5):
    """Repeated substitution: iterate z <- f(z) until convergence.

    Valid whenever f is a contraction (Lipschitz constant < 1), which is
    guaranteed here by construction (see LipschitzMLP below).

    Returns (z, n_iter, final_rel_change) so callers can check whether the
    solve actually converged within budget, instead of silently trusting
    that `max_iter` was enough.
    """
    z = z0
    n_iter = max_iter
    rel_change = float("nan")
    for i in range(max_iter):
        z_next = f(z)
        rel_change = ((z_next - z).norm() / (z.norm() + 1e-6)).item()
        z = z_next
        if rel_change < tol:
            n_iter = i + 1
            break
    return z, n_iter, rel_change


class LipschitzMLP(nn.Module):
    """A small MLP with Lipschitz constant strictly below 1.

    Every linear layer is spectrally normalized (operator norm <= 1), and
    ReLU/ELU activations have Lipschitz constant 1, so the composed network
    (before the final `coeff` scaling) already has Lipschitz constant <= 1.
    Scaling the output by `coeff < 1` gives the whole map a Lipschitz
    constant <= coeff, with margin to spare for the contraction to be
    well-behaved numerically.
    """

    def __init__(self, dim, hidden_dim=64, coeff=0.8, n_power_iterations=5):
        super().__init__()
        self.coeff = coeff

        def sn(layer):
            return nn.utils.spectral_norm(layer, n_power_iterations=n_power_iterations)

        self.net = nn.Sequential(
            sn(nn.Linear(dim, hidden_dim)), nn.ELU(),
            sn(nn.Linear(hidden_dim, hidden_dim)), nn.ELU(),
            sn(nn.Linear(hidden_dim, dim)),
        )

    def forward(self, z):
        return self.coeff * self.net(z)


class DEQBlock(nn.Module):
    """One invertible equilibrium layer built around a contraction g."""

    def __init__(self, dim, hidden_dim=64, coeff=0.8, max_iter=150, tol=1e-5):
        super().__init__()
        self.g = LipschitzMLP(dim, hidden_dim, coeff)
        self.max_iter = max_iter
        self.tol = tol

        # Diagnostics populated on every forward()/backward() call, purely
        # for inspection -- not used in the actual computation. See the
        # "Diagnostics" section below for how these are reported.
        self.last_forward_iters = None
        self.last_forward_residual = None
        self.last_backward_iters = None
        self.last_backward_residual = None

    def forward(self, x):
        """x -> z, with an implicit-differentiation backward pass.

        Standard DEQ pattern (Bai, Kolter & Koltun, 2019):
          1. Solve for the fixed point z* under no_grad (cheap, no graph).
          2. Re-apply g *once* at the (detached) fixed point, with grad
             enabled, to attach a shallow graph connecting z to x and to
             g's parameters.
          3. Register a backward hook on z that replaces the incoming
             gradient with the solution of the adjoint fixed-point
             equation (I - J^T) v = grad -- this is what makes the shallow
             graph from step 2 produce the *true* total-derivative
             gradient, without ever unrolling the forward solver.
        """
        with torch.no_grad():
            z_star, n_iter, residual = fixed_point_iterate(
                lambda z: x + self.g(z), x, self.max_iter, self.tol)
        self.last_forward_iters = n_iter
        self.last_forward_residual = residual

        z = x + self.g(z_star)  # value ~= z_star, but now differentiable

        if z.requires_grad:
            z0 = z.detach().requires_grad_()
            g0 = self.g(z0)

            def backward_hook(grad):
                v, n_iter_bwd, residual_bwd = fixed_point_iterate(
                    lambda v: torch.autograd.grad(g0, z0, grad_outputs=v, retain_graph=True)[0] + grad,
                    grad, self.max_iter, self.tol,
                )
                self.last_backward_iters = n_iter_bwd
                self.last_backward_residual = residual_bwd
                return v

            z.register_hook(backward_hook)

        return z

    def inverse(self, z):
        """z -> x, explicit -- no iteration needed."""
        return z - self.g(z)

    def log_det(self, z):
        """log|det(dz/dx)| = -log|det(I - dg/dz)|, evaluated exactly at z.

        Only practical for modest N (see module-level note above).

        Note: `create_graph=True` is required here (not just
        `retain_graph=True`) because the row-extraction below is itself a
        gradient computation, and its *output* (`logabsdet`) is later
        differentiated again w.r.t. g's parameters when the outer training
        loss calls `.backward()`. Without `create_graph=True` that second
        round of differentiation would silently see zero gradient from
        this branch.

        The whole computation is also wrapped in `torch.enable_grad()`
        since `log_prob` (and hence this method) is routinely called
        inside a `torch.no_grad()` block during evaluation -- without
        locally re-enabling grad tracking here, building the Jacobian
        would raise an error (the detached input would have no grad_fn to
        differentiate).
        """
        batch, dim = z.shape
        with torch.enable_grad():
            z_ = z.detach().requires_grad_()
            gz = self.g(z_)
            rows = []
            for i in range(dim):
                grad_outputs = torch.zeros_like(gz)
                grad_outputs[:, i] = 1.0
                row = torch.autograd.grad(gz, z_, grad_outputs=grad_outputs,
                                           create_graph=True, retain_graph=True)[0]
                rows.append(row)
            jac = torch.stack(rows, dim=1)                  # (batch, dim, dim): jac[:, i, :] = d g_i / d z
            eye = torch.eye(dim, device=z.device).unsqueeze(0).expand(batch, -1, -1)
            _, logabsdet = torch.linalg.slogdet(eye - jac)
        return -logabsdet

    def jacobian_singular_values(self, z):
        """Diagnostic only (not used for training): singular values of
        dg/dz and of (I - dg/dz) at z.

        If the largest singular value of dg/dz is close to `coeff`, the
        network is pushing right up against its allowed Lipschitz budget.
        If the smallest singular value of (I - dg/dz) is close to 0, the
        log-det Jacobian term `-log|det(I - dg/dz)|` is close to a
        singularity and can blow up to arbitrarily large values -- this is
        the "cheat" that lets log_prob report an unrealistically good
        likelihood without the model actually fitting the data well.
        """
        batch, dim = z.shape
        with torch.enable_grad():
            z_ = z.detach().requires_grad_()
            gz = self.g(z_)
            rows = []
            for i in range(dim):
                grad_outputs = torch.zeros_like(gz)
                grad_outputs[:, i] = 1.0
                row = torch.autograd.grad(gz, z_, grad_outputs=grad_outputs, retain_graph=True)[0]
                rows.append(row.detach())
        jac = torch.stack(rows, dim=1)
        eye = torch.eye(dim, device=z.device).unsqueeze(0).expand(batch, -1, -1)
        sv_g = torch.linalg.svdvals(jac)             # (batch, dim): singular values of dg/dz
        sv_resid = torch.linalg.svdvals(eye - jac)   # (batch, dim): singular values of (I - dg/dz)
        return sv_g, sv_resid


class DEQFlow(nn.Module):
    """A stack of DEQ equilibrium blocks, exposing the same API as RealNVP.

    Unlike coupling layers, each block already sees and transforms *all*
    N dimensions at once (g maps R^N -> R^N directly) -- no masking or
    permutation trick is needed for the dimensions to mix.
    """

    def __init__(self, dim, n_blocks=4, hidden_dim=64, coeff=0.8):
        super().__init__()
        self.blocks = nn.ModuleList([
            DEQBlock(dim, hidden_dim=hidden_dim, coeff=coeff) for _ in range(n_blocks)
        ])
        self.register_buffer("base_loc", torch.zeros(dim))
        self.register_buffer("base_scale", torch.ones(dim))

    @property
    def base_dist(self):
        return Independent(Normal(self.base_loc, self.base_scale), 1)

    def forward(self, x):
        """x -> z, accumulating total log|det dz/dx| across blocks."""
        log_det_total = x.new_zeros(x.shape[0])
        z = x
        for block in self.blocks:
            z = block.forward(z)
            log_det_total = log_det_total + block.log_det(z)
        return z, log_det_total

    def inverse(self, z):
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return self.base_dist.log_prob(z) + log_det

    def sample(self, n):
        with torch.no_grad():
            z = self.base_dist.sample((n,))
            return self.inverse(z)

    @torch.no_grad()
    def diagnostics(self, x):
        """Collect the diagnostics discussed for tracking down the
        NLL/sample-variance mismatch: per-block fixed-point solver
        convergence, per-block Jacobian singular values, round-trip
        (forward/inverse) consistency, and -- crucially -- a decomposition
        of log_prob into its base_dist term vs. its log_det term, plus the
        empirical mean/std of the final latent z. This lets us see
        directly whether z actually resembles the assumed N(0, I) prior,
        and which term is responsible for an anomalous NLL, rather than
        just inferring it indirectly from Jacobian bounds. Returns a plain
        dict; see `print_deq_diagnostics` below for a formatted report.
        """
        report = {"blocks": []}

        # --- per-block solver convergence + Jacobian singular values ---
        z = x
        log_det_total = x.new_zeros(x.shape[0])
        for block in self.blocks:
            z = block.forward(z)  # populates block.last_forward_iters/residual
            sv_g, sv_resid = block.jacobian_singular_values(z)
            block_log_det = block.log_det(z)
            log_det_total = log_det_total + block_log_det
            report["blocks"].append({
                "forward_iters":    block.last_forward_iters,
                "forward_residual": block.last_forward_residual,
                "max_sv_g":         sv_g.max().item(),        # closer to coeff => pushing the Lipschitz budget
                "min_sv_resid":     sv_resid.min().item(),    # closer to 0 => (I - dg/dz) near-singular
                "coeff":            block.g.coeff,
                "mean_log_det":     block_log_det.mean().item(),
            })
        z_final = z

        # --- decompose log_prob = base_dist.log_prob(z) + log_det ---
        base_logprob = self.base_dist.log_prob(z_final)
        report["mean_base_logprob"]  = base_logprob.mean().item()
        report["mean_log_det_total"] = log_det_total.mean().item()
        report["mean_log_prob"]      = (base_logprob + log_det_total).mean().item()
        report["z_mean_per_dim"] = z_final.mean(dim=0).cpu().tolist()
        report["z_std_per_dim"]  = z_final.std(dim=0).cpu().tolist()

        # --- round trip: x -> z -> x_hat ---
        x_hat = self.inverse(z_final)
        report["data_round_trip_error"] = (x - x_hat).norm(dim=-1).mean().item()

        # --- round trip: z ~ base -> x -> z_hat ---
        z_sample = self.base_dist.sample((x.shape[0],))
        x_gen = self.inverse(z_sample)
        z_hat, _ = self.forward(x_gen)
        report["latent_round_trip_error"] = (z_sample - z_hat).norm(dim=-1).mean().item()

        return report


def print_deq_diagnostics(report):
    print("\nDEQ flow diagnostics:")
    for i, b in enumerate(report["blocks"]):
        print(f"  block {i}: solver iters={b['forward_iters']:3d}  "
              f"final rel_change={b['forward_residual']:.2e}  |  "
              f"max sv(dg/dz)={b['max_sv_g']:.4f} (coeff={b['coeff']:.2f})  "
              f"min sv(I - dg/dz)={b['min_sv_resid']:.4f}  |  mean log_det={b['mean_log_det']:.4f}")
    z_mean_str = ", ".join(f"{v:+.3f}" for v in report["z_mean_per_dim"])
    z_std_str  = ", ".join(f"{v:.3f}"  for v in report["z_std_per_dim"])
    print(f"  final latent z: per-dim mean = [{z_mean_str}]")
    print(f"  final latent z: per-dim std  = [{z_std_str}]  (should be ~0 / ~1 if z matches N(0,I))")
    print(f"  mean base_dist.log_prob(z) : {report['mean_base_logprob']:.4f}")
    print(f"  mean log_det (all blocks)  : {report['mean_log_det_total']:.4f}")
    print(f"  mean log_prob (sum)        : {report['mean_log_prob']:.4f}   "
          f"(=> NLL on this batch: {-report['mean_log_prob']:.4f})")
    print(f"  round-trip error   x -> z -> x_hat : {report['data_round_trip_error']:.4e}")
    print(f"  round-trip error   z -> x -> z_hat : {report['latent_round_trip_error']:.4e}")


flow_deq = DEQFlow(dim=N, n_blocks=4, hidden_dim=64, coeff=0.8).to(device)

# ---------------------------------------------------------------------------
# 4. Train via maximum likelihood: minimize -mean(log_prob(x))
# ---------------------------------------------------------------------------

n_epochs = 100
lr       = 1e-3


def train_flow(model, label, n_epochs=n_epochs, lr=lr):
    """Train `model` via MLE on loader_train; report progress like the
    original single-model script did. Returns (train_losses, test_losses).
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses  = []

    for epoch in tqdm(range(n_epochs), desc=label):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for (x_batch,) in loader_train:
            opt.zero_grad()
            loss = -model.log_prob(x_batch).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x_batch.shape[0]
            n_seen += x_batch.shape[0]
        train_losses.append(epoch_loss / n_seen)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_loss = -model.log_prob(data_test).mean().item()
            test_losses.append((epoch, test_loss))

    return train_losses, test_losses


train_losses_realnvp, test_losses_realnvp = train_flow(flow_realnvp, "RealNVP")
train_losses_deq,     test_losses_deq     = train_flow(flow_deq,     "DEQ flow")

# ---------------------------------------------------------------------------
# 5. Evaluate: compare log-likelihood of both flows vs. the true generator
# ---------------------------------------------------------------------------

flow_realnvp.eval()
flow_deq.eval()
with torch.no_grad():
    true_nll    = -target_dist.log_prob(data_test.cpu()).mean().item()
    realnvp_nll = -flow_realnvp.log_prob(data_test).mean().item()
    deq_nll     = -flow_deq.log_prob(data_test).mean().item()

print(f"\nHeld-out negative log-likelihood (nats/sample), N={N}, N_TRAIN={N_TRAIN}:")
print(f"  true generator : {true_nll:.4f}")
print(f"  RealNVP        : {realnvp_nll:.4f}")
print(f"  DEQ flow       : {deq_nll:.4f}")

# ---------------------------------------------------------------------------
# 5b. DEQ diagnostics -- is log_prob actually self-consistent with sample()?
# ---------------------------------------------------------------------------
#
# A valid model can never (in expectation, on data drawn from the true
# distribution) beat the true generator's own NLL -- if it appears to, the
# reported log_prob isn't a properly normalized density. This block checks
# the three most likely causes: (1) the fixed-point solver not actually
# converging within its iteration budget, (2) dg/dz being pushed close to
# its Lipschitz ceiling, making (I - dg/dz) close to singular and
# log_det artificially huge, and (3) forward/inverse not being consistent
# inverses of each other at the current parameters.
deq_report = flow_deq.diagnostics(data_test[:256])
print_deq_diagnostics(deq_report)

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

# --- Training curves (both models on one axis, spanning the top row) ---
ax = fig.add_subplot(gs[0, :])
ax.plot(train_losses_realnvp, color="tab:orange", label="RealNVP train")
te_epochs, te_vals = zip(*test_losses_realnvp)
ax.plot(te_epochs, te_vals, color="tab:orange", marker="o", ms=4, linestyle="--", label="RealNVP test")
ax.plot(train_losses_deq, color="tab:green", label="DEQ train")
te_epochs, te_vals = zip(*test_losses_deq)
ax.plot(te_epochs, te_vals, color="tab:green", marker="o", ms=4, linestyle="--", label="DEQ test")
ax.axhline(true_nll, color="k", linestyle=":", label="true generator NLL")
ax.set_xlabel("epoch")
ax.set_ylabel("negative log-likelihood")
ax.set_title("Maximum-likelihood training: RealNVP vs. DEQ")
ax.legend(fontsize=8)

with torch.no_grad():
    realnvp_samples = flow_realnvp.sample(N_TEST).cpu().numpy()
    deq_samples     = flow_deq.sample(N_TEST).cpu().numpy()
real_samples = data_test.cpu().numpy()

# --- Scatter comparison on the first two dimensions (bottom row) ---
scatter_data = [
    ("True samples", real_samples, "tab:blue"),
    ("RealNVP samples", realnvp_samples, "tab:orange"),
    ("DEQ samples", deq_samples, "tab:green"),
]
scatter_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
for ax, (title, samples, color) in zip(scatter_axes, scatter_data):
    ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.4, color=color)
    ax.set_title(f"{title} (dims 0,1)")
    ax.set_xlabel("x0"); ax.set_ylabel("x1")

xlim = (min(ax.get_xlim()[0] for ax in scatter_axes), max(ax.get_xlim()[1] for ax in scatter_axes))
ylim = (min(ax.get_ylim()[0] for ax in scatter_axes), max(ax.get_ylim()[1] for ax in scatter_axes))
for ax in scatter_axes:
    ax.set_xlim(xlim); ax.set_ylim(ylim)

plt.tight_layout()
plt.savefig("normalizing_flow_demo.png", dpi=120)
plt.show()
