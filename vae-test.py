# Dual masking:
# Type 1. Some values in each sample are missing from the generated training/test data-sets. Loss cannot be calculated for them(because the target is unknown), and they will be imputed based on the values of other variables in that sample.  
# Type 2. Some values in each sample are masked during each training epoch (or during evaluation). Loss can be calculated for them.
# The model sees both types as masked values - the difference is only in the loss.  
# 
# TODOs:
# 1. Fix masking  
# 2. [Done] Fix missing value initialization -- was mean-fill, now per-gene N(mean, std) random fill (see get_random_mask/_random_fill in synthetic_data.py)
# 3. Use negative binomial distribution, Zero-Inflated Negative Binomial (ZINB) Loss, log(1+x)+MSE  
# 3b. Fix distribution shift on reconstructed output  
# 4. Replace mask concatenatenation with canonical style  
# 5. Use type2_mask in loss, add lambda to balance loss on masked/unmasked data  
# 
# --
# 
# The model has to learn two separate (but linked) things:
# 1. The sparse but strong correlation structure between genes that allows imputing missing values
# 2. The implicit "sample state" that allows different samples to be imputed differently  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from synthetic_data import make_synthetic_data4, make_synthetic_data5, random_points_on_hypersphere
from models import GeneExpressionVAE, MoVEVAE, VampPriorVAE, VAEBase, epoch_vae, get_random_mask, BetaSSMController, World, Controller, rl_training_loop, plot_rl_results, MeansModel, ImputationTransformer, epoch_transformer, impute_transformer, make_log_bin_edges, model_factories
import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_cfg = {'n_blocks': 32, 'rho_within': 0.8,
          'n_genes': 200, 'n_cells': 2_000, 'missing_rate': 0.05}
vtc = {'max_epochs': 50, 'batch_size': 200,
        'mask_fraction': 0.05, 'lr': 3e-4, 'n_pseudo': 50}
vlc = {'gamma_recon': 4.0, 'min_free_bits': 0.01, 'lambda_entropy': 0.01}

world      = World(vtc, vlc, base_cfg, beta_ref=1e-4, lambda_K=0.1)
ssm        = BetaSSMController(obs_dim=12, hidden_dim=32).to(device)
controller = Controller(ssm, lr=1e-3)

controller, replay_buffer, episode_log = rl_training_loop(
  world, controller,
  n_episodes=30,
  n_pretrain_episodes=10,   # collect 10 waveform episodes first
  pretrain_steps=200,       # world-model steps before RL begins
  )
plot_rl_results(episode_log)
# problem

#_train, _test = GeneExpressionDataset(
#    make_synthetic_data(n_cells, n_genes, n_types=5, missing_rate=0.05)
#)
#data_train, mask_train = _train
#data_test,  mask_test  = _test

selected_problem='standard4'

if selected_problem=='easy':
  n_cells =   500
  n_genes =   20

  ret = make_synthetic_data2(n_cells, n_genes, n_blocks=2, n_types=1, rho_within=0.7, missing_rate=0.02)

elif selected_problem=='easy_plus':
  n_cells =   2_000
  n_genes =   20

  ret = make_synthetic_data2(n_cells, n_genes, n_blocks=3, n_types=3, rho_within=0.7, missing_rate=0.02)

elif selected_problem=='standard':

  n_cells =   2_000
  n_genes =     200

  ret = make_synthetic_data2(n_cells, n_genes, n_blocks=18, n_types=4, rho_within=0.7, missing_rate=0.05)

elif selected_problem=='standard4':

  n_cells  = 2_000
  n_genes  =   200
  n_blocks =    32

  if False:
    # draw block state randomly (i.e., all cells come from the same distribution)
    block_state_train=torch.randn(n_cells, n_blocks)
    block_state_test =torch.randn(n_cells, n_blocks)

  else:
    block_state_both, _ = random_points_on_hypersphere(
        num_points=n_cells,
        K=n_blocks,
        N=16, device=device)  # generate K block-state values (for each cell), constrained to lie on a N-d hypershere (N DoF)
    corr_scale = 4 # 0.7   # highest correlation between blocks (approximately)
    scale = block_state_both.abs().max(dim=0).values
    block_state_both /= scale / corr_scale  # normalize (otherwise higher-K problems have weaker correlations)
    assert(block_state_both.shape==(n_cells, n_blocks))

  ret = []
  for block_state in (block_state_both, block_state_both):  # generate train/test data from the same population of cell-states

    _data, _ = make_synthetic_data4(
        block_state=block_state,
        n_genes=n_genes,
        n_blocks=n_blocks,
        rho_within=0.8,
        allow_negative=True,
        missing_rate=0.05
    )
    ret.append( _data )
    ret.append(torch.zeros(n_cells)) # all cells belong to the same type...
  ret.append(None)

else:
  assert(False)

data_train, types_train, data_test, types_test, _  = ret

loader_test           = DataLoader(data_test,  batch_size=n_cells, shuffle=False)
loader_train_no_batch = DataLoader(data_train, batch_size=n_cells, shuffle=True)  # for evaluation

# ---------------------------------------------------------------------------
# load_model: restore a pre-trained model from a checkpoint file.
#
# Expected checkpoint format (saved with torch.save):
#   {
#     'model_type': <key from model_factories>,
#     'state_dict': <model.state_dict()>,
#     'config':     <optional dict of architecture hyperparameters, as found
#                    on model.config after construction -- merged over the
#                    factory's documented *_CONFIG defaults if present>,
#   }
# ---------------------------------------------------------------------------

def load_model(path, n_genes, factories=model_factories, device=device):
    """Load a pre-trained model from *path* and return it in eval mode.

    Expected checkpoint format (saved with torch.save):
        {
            'model_type': <key in model_factories>,
            'state_dict': <model.state_dict()>,
            'config':     <optional dict, see module docstring above>,
        }

    Post-load fixups applied automatically:
      - MeansModel  : means re-fitted from data_train (the parameter has no
                      meaning outside the dataset it was computed on).
      - ImputationTransformer : bin_edges attached from it_bin_edges so that
                      impute() can find them without a separate argument.
                      (Call load_model only after it_bin_edges is defined.)
    """
    ckpt       = torch.load(path, map_location=device)
    model_type = ckpt['model_type']
    if model_type not in factories:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Available: {list(factories)}")
    m = factories[model_type](n_genes, ckpt.get('config')).to(device)
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    if isinstance(m, MeansModel):
        m.means.data = data_train.nanmean(dim=0).to(device)
    if isinstance(m, ImputationTransformer):
        m.bin_edges = it_bin_edges
    return m

# ---------------------------------------------------------------------------
# Active model for the training loop below.
# Swap the key to select a different architecture.
# ---------------------------------------------------------------------------

_active_model_type = 'GeneExpressionVAE'
model = model_factories[_active_model_type](n_genes).to(device)
if isinstance(model, MeansModel):
    model.means.data = data_train.nanmean(dim=0).to(device)
learning_graph_train = []
learning_graph_test  = []
next_epoch = 0
#opt = optim.AdamW(model.parameters(), lr=3e-4)
beta_final = 5.0e-2 # 0.4 # 1e-4 # 1e-2
#beta_sched = 0.5 * beta_final * (1 - np.cos(np.pi*np.linspace(0.002,0.999,1500)) )
#beta_sched = np.hstack( (np.full((100,), fill_value=1e-6),
#                    0.5 * beta_final * (1 - np.cos(np.pi*np.linspace(0.01,0.999,100)) ) ) )
beta_sched = torch.full( size=(200,), fill_value=beta_final )
plt.plot(beta_sched)
plt.gca().set_yscale('log');
max_epocs     =    100 # 150 #100
batch_size    =    200 # 200 #200
lr            =   3e-4 # 5e-4
min_free_bits =   0.005 # 0.8
gamma         =   4.0 # 1.6
mask_fraction =   0.05
lambda_entropy=   0.01 # 0.5


opt = optim.AdamW([{'params':[p for n,p in model.named_parameters() if 'pseudo' not in n], 'lr':lr}],
                  lr=lr
                  )
#opt = optim.AdamW([{'params':[p for n,p in model.named_parameters() if 'pseudo' not in n], 'lr':lr},
#                   {'params':model.pseudo_inputs, 'lr':1.0*lr}],
#                  lr=lr
#                  )

loader_train          = DataLoader(data_train, batch_size=batch_size, shuffle=True)

for g in opt.param_groups: g['lr'] = lr

#for x, mask in loader_train:
#  pred_train, mu, logvar = model(x)

for i in tqdm(range(max_epocs)):
  beta = beta_sched[min(next_epoch,beta_sched.shape[0]-1)]

  train_loss, train_norm, loss1, loss1_type2, loss2 = epoch_vae(
      model,
      loader_train,
      opt,
      mask_fraction=mask_fraction,
      beta=beta,
      gamma=gamma,
      min_free_bits=min_free_bits,
      lambda_entropy=lambda_entropy)

  if isinstance(model, VampPriorVAE):
    effk = model._effective_K
  else:
    effk = None

  learning_graph_train.append( (next_epoch, train_loss, train_norm, loss1, loss2, loss1_type2, effk) )

  if next_epoch%10 == 0:
    test_loss, _a, _b, _c, _d = epoch_vae(
        model,
        loader_test,
        mask_fraction=mask_fraction,
        beta=beta,
        gamma=gamma,
        min_free_bits=min_free_bits,
        lambda_entropy=lambda_entropy)

    learning_graph_test.append( (next_epoch, test_loss, None, None, None) )

  next_epoch += 1


# ---------------------------------------------------------------------------
# VAE imputation correlation sweep (uses unified impute())
# ---------------------------------------------------------------------------

mask_fractions = np.exp(np.linspace(-0.35,-4.5,100+1))

ret = []
model.eval()
for frac in tqdm(mask_fractions):
    for x in loader_test:
        recon, type2_mask = impute(model, x, frac, device=device)
        if type2_mask.any():
            corr_data = np.vstack((x[type2_mask].numpy(),
                                   recon[type2_mask].numpy()))
            corr = np.corrcoef(corr_data, rowvar=True)[0, 1]
        else:
            corr = float('nan')
        ret.append((type2_mask.float().mean().item(), corr))
ret = np.array(ret)
plt.plot( ret[:,0], ret[:,1])
plt.gca().set_xscale('log')
plt.title('Imputed reconstruction')
plt.xlabel('Masked fraction')
plt.ylabel('Imputation correlation');

# Use the same data_train / data_test already built by the standard4 block above.
# (data_train and data_test are (n_cells, n_genes) float tensors with NaN for missing.)
it_loader_train = DataLoader(data_train, batch_size=it_batch_size, shuffle=True)
it_loader_test  = DataLoader(data_test,  batch_size=n_cells,       shuffle=False)

it_model = models._make_imputation_transformer(n_genes).to(device)
# Attach bin_edges as a model attribute so load_model and impute() can access
# them without needing a separate argument.
it_model.bin_edges = models.it_bin_edges

it_opt = optim.AdamW(it_model.parameters(), lr=it_lr)

it_log_train = []
it_log_test  = []
_ep = 0  # epoch counter
it_max_epochs = 20

print('\n=== ImputationTransformer training ===')
for _ in tqdm(range(it_max_epochs)):
    tr_loss, tr_count_ce, tr_conf_ce, tr_gnorm = epoch_transformer(
        it_model, it_loader_train, it_bin_edges,
        opt=it_opt,
        mask_fraction=it_mask_frac,
        lambda_conf=it_lambda_conf,
        device=device,
    )
    it_log_train.append((_ep, tr_loss, tr_count_ce, tr_conf_ce, tr_gnorm))

    if _ep % 5 == 0:
        te_loss, te_count_ce, te_conf_ce, _ = epoch_transformer(
            it_model, it_loader_test, it_bin_edges,
            opt=None,
            mask_fraction=it_mask_frac,
            lambda_conf=it_lambda_conf,
            device=device,
        )
        it_log_test.append((_ep, te_loss, te_count_ce, te_conf_ce))
        print(f'  ep {_ep:3d}  '
              f'train_ce={tr_count_ce:.4f}  conf_ce={tr_conf_ce:.4f}  gnorm={tr_gnorm:.3f}  '
              f'| test_ce={te_count_ce:.4f}')
    _ep += 1

# Plot training curves
_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
_ep_tr = [r[0] for r in it_log_train]
_ep_te = [r[0] for r in it_log_test]
_axes[0].plot(_ep_tr, [r[2] for r in it_log_train], label='train count CE')
_axes[0].plot(_ep_te, [r[2] for r in it_log_test],  label='test count CE', marker='o', ms=4)
_axes[0].set_xlabel('Epoch'); _axes[0].set_ylabel('Cross-entropy')
_axes[0].set_title('ImputationTransformer -- count loss')
_axes[0].legend()
_axes[1].plot(_ep_tr, [r[3] for r in it_log_train], label='train conf CE', color='tab:orange')
_axes[1].set_xlabel('Epoch'); _axes[1].set_ylabel('Cross-entropy')
_axes[1].set_title('ImputationTransformer -- confidence loss')
_axes[1].legend()
plt.tight_layout()
plt.savefig('it_training_curves.png', dpi=120)
plt.show()
# ---------------------------------------------------------------------------------------------
# 7. Evaluation -- imputation correlation sweep (on type-2 masked positions) vs. mask fraction
# ---------------------------------------------------------------------------------------------

it_model.eval()

it_mask_fractions = np.exp(np.linspace(-0.35, -4.5, 10))
it_corr_results   = []

for _frac in tqdm(it_mask_fractions):
    for x_raw in it_loader_test:
        recon, type2_eval = impute(it_model, x_raw, _frac, device=device)
        if type2_eval.any():
            true_vals = x_raw[type2_eval].numpy()
            pred_vals = recon[type2_eval].numpy()
            corr = np.corrcoef(true_vals, pred_vals)[0, 1] if len(true_vals) > 1 else float('nan')
        else:
            corr = float('nan')
        it_corr_results.append((float(type2_eval.float().mean()), corr))

it_corr_arr = np.array(it_corr_results)

plt.plot( it_corr_arr[:,0],  it_corr_arr[:,1] );
