# Extending make_synthetic_data5 for DEQ VAE data generation

**Session ID:** ses_0948dd3bcffedAlisEFiT58MhJ
**Created:** 7/16/2026, 6:01:12 PM
**Updated:** 7/17/2026, 1:07:36 AM

---

## User

make_synthetic_data5 (synthetic_data.py) is used to generate gene-expression-like data that is unevenly distributed on a low-dimensional manifold. I'd like to use this data to train architectures containing a DEQ model (e.g. use a DEQ as a prior for a VAE model). The objective is to explore configurations that have a good inductive bias and are sample-efficient while preventing mode collapse, and handle noisy and incomplete data without requiring imputation of missing genes as preprocessing. What are possible approaches for extending make_synthetic_data5 to generate values for z and x?

---

## Assistant (Plan · Claude Sonnet 4.6 · 7.9s)

I'll start by reading the relevant files to understand the current codebase.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/michael/src/vae-test/synthetic_data.py"
}
```

**Output:**
```
<path>/home/michael/src/vae-test/synthetic_data.py</path>
<type>file</type>
<content>
1: import math
2: import torch
3: import torch.nn as nn
4: import torch.nn.functional as F
5: import torch.optim as optim
6: import torch.autograd as autograd
7: from torch.utils.data import Dataset, DataLoader
8: import numpy as np
9: 
10: # Synthetic data for gene expression imputation
11: def make_synthetic_data(
12:     n_cells: int   = 800,
13:     n_genes: int   = 1000,
14:     n_types: int   = 5,
15:     missing_rate: float = 0.3,
16:     seed: int      = 42,
17: ):
18:   """
19:   Simulate a log-normalised gene-expression matrix with:
20:     - n_types latent cell-type programmes
21:     - ~missing_rate dropout (set to NaN)
22: 
23:   Masking (type 1): missing as np.nan
24:   """
25:   rng = np.random.default_rng(seed)
26: 
27:   # Low-rank structure: cells × types  ×  types × genes
28:   programs    = rng.standard_normal((n_types, n_genes)).clip(0)    # cell types/patterns
29:   expr_sets = []
30:   for _ in ('train', 'test'):
31:     assignments = rng.dirichlet(np.ones(n_types), size=n_cells)      # type composition of each sample
32:     expr        = assignments @ programs
33:     expr        = np.log1p(expr + rng.exponential(0.1, expr.shape))
34: 
35:     # Introduce dropout (MCAR for simplicity)
36:     dropout = rng.random(expr.shape) < missing_rate
37:     expr[dropout] = float('nan')
38: 
39:     expr_sets.append( expr.astype(np.float32) )
40: 
41:   return expr_sets[0], expr_sets[1]
42: 
43: 
44: 
45: def build_modular_cov(
46:     N:int,
47:     K:int,
48:     generator,
49:     rho_between:float=0.05,
50:     rho_within:float=0.5,
51:     allow_negative=True,
52: ):
53:   block_bounds = np.linspace(0,N,K+1).round().astype(int)
54:   Sigma = torch.full((N,N), rho_between, requires_grad=False)
55:   for k in range(K):
56:     s = block_bounds[k]
57:     e = block_bounds[k+1]
58:     assert(e>s)
59:     block = torch.full((e-s, e-s), rho_within, requires_grad=False)
60:     block.fill_diagonal_(1.0)
61: 
62:     if allow_negative:
63:       signs = torch.randint(0, 2, (e-s,), generator=generator) * 2 - 1
64:       sign_mat = signs.unsqueeze(0) * signs.unsqueeze(1)
65:       block = block * sign_mat.float()
66:       block.fill_diagonal_(1.0)  # diagonal stays 1
67: 
68:     Sigma[s:e,s:e] = block
69: 
70:   # nudge towards positive semi-definitiveness if needed
71:   min_eig = torch.linalg.eigvalsh(Sigma).min()
72:   if min_eig <= 0:
73:     while True:
74:       jitter = (-min_eig + 1e-6) * torch.eye(N)
75:       Sigma = Sigma + jitter
76:       # Re-normalise to keep unit diagonal (correlation matrix)
77:       d = Sigma.diagonal().sqrt()
78:       Sigma = Sigma / d.unsqueeze(0) / d.unsqueeze(1)
79:       min_eig = torch.linalg.eigvalsh(Sigma).min()
80: 
81:       if min_eig > 0: break
82: 
83:   return Sigma, block_bounds
84: 
85: def make_synthetic_data2(
86:     n_cells: int        = 800,
87:     n_genes: int        = 200,
88:     n_blocks: int       = 5,
89:     n_types: int        = 1,
90:     missing_rate: float = 0.1,
91:     rho_within:float    = 0.5,
92:     rho_between:float   = 0.05,
93:     means:torch.Tensor|None= None,
94:     stds:torch.Tensor|None= None,
95:     allow_negative:bool = True,
96:     seed: int           = 42,
97:     device:str          = None):
98:   """
99:   Masking (type 1): missing as np.nan
100:   """
101:   generator = torch.Generator(device=device)
102:   if seed is not None:
103:     generator.manual_seed(seed)
104: 
105:   #if means is not None or stds is not None:
106:   dtype = torch.float32
107:   mu    = means.to(dtype=dtype, device=device) if means is not None else torch.zeros(n_genes, dtype=dtype, device=device)
108:   sigma = stds.to( dtype=dtype, device=device) if stds  is not None else torch.ones( n_genes, dtype=dtype, device=device)
109: 
110: 
111:   Sigma = []
112:   block_bounds = []
113:   L = []
114:   for _ in range(n_types):
115:     _Sigma, _block_bounds = build_modular_cov(
116:       N=n_genes,
117:       K=n_blocks,
118:       rho_between=rho_between,
119:       rho_within=rho_within,
120:       allow_negative=True,
121:       generator=generator)
122: 
123:     _L = torch.linalg.cholesky(_Sigma)  # Sigma = L @ L.T
124: 
125:     Sigma.append( _Sigma )
126:     block_bounds.append( _block_bounds)
127:     L.append( _L )
128: 
129:   out = []
130:   for _ in [0,1]:
131:     Xcandidates = []
132: 
133:     for t in range(n_types):
134:       # Sample standard normals, then rotate with Cholesky factor
135:       Z = torch.randn(n_cells, n_genes, device=device, generator=generator)
136:       _Xt = Z @ L[t].T  # shape (n_cells, N)
137:       Xcandidates.append(_Xt)
138:     Xcandidates = torch.stack( Xcandidates )
139:     print(f'Xcandidates:{Xcandidates.shape}')
140: 
141:     # pick single 'type' for each sample; data will be an even mixture of the types
142:     x_types = torch.randint(0,n_types,size=(n_cells,))
143:     print(f'x_types:{x_types.shape}')
144:     # for each sample, fetch the data generated using the chosen type
145:     X = Xcandidates[x_types,torch.arange(n_cells),:]
146:     print(f'X:{X.shape}')
147: 
148:     # Map latent normal values to NB means and sample counts
149:     mean_nb = torch.exp(X * sigma.unsqueeze(0) + mu.unsqueeze(0))
150:     r = torch.tensor(10.0, device=device)
151:     probs = r / (r + mean_nb)
152:     counts = torch.distributions.NegativeBinomial(total_count=r, probs=probs).sample()
153:     X = torch.log1p(counts)
154: 
155:     # Introduce dropout (MCAR for simplicity)
156:     dropout = torch.rand(X.shape, generator=generator) < missing_rate
157:     X[dropout] = float('nan')
158: 
159:     out.append((X, x_types))
160: 
161: 
162:   #modules = [
163:   #    list(range(k * vars_per_module, (k + 1) * vars_per_module))
164:   #    for k in range(K)
165:   #]
166: 
167:   return (out[0][0], out[0][1],
168:           out[1][0], out[1][1],
169:           block_bounds)
170: 
171: 
172: def get_random_mask(x:torch.Tensor, masked_fraction:float) -> torch.Tensor:
173:   """
174:   """
175:   type1_mask = torch.isnan(x)
176: 
177:   type2_mask = torch.rand(x.shape) < masked_fraction
178:   type2_mask = torch.logical_and( type2_mask, ~type1_mask )
179:   mask  = type2_mask | type1_mask
180: 
181:   #print(f'x:{x.shape} mean_vals:{mean_vals.shape}')
182:   x_masked = x.clone()
183:   #x_masked[mask] = x.mean(dim=0).expand((x.shape[0],-1)) # 0 # torch.nan # TODO use random values
184:   x_masked = x_masked.where( ~mask, x.nanmean(dim=0).expand((x.shape[0],-1)) )
185:   x_masked += torch.randn_like(x_masked) * 0.05  # Augmentation: add noise to the input
186:   # TODO use better noise
187:   return x_masked, mask, type2_mask
188: 
189: class GeneExpressionDataset(Dataset):
190:   def __init__(self, matrix:np.ndarray|torch.Tensor):
191:     if isinstance(matrix, np.ndarray):
192:       self.data = torch.tensor(matrix, dtype=torch.float32)
193:     else:
194:       self.data = matrix.to(dtype=torch.float32)
195:     self.mask = torch.tensor( ~np.isnan(matrix), dtype=torch.bool)
196:     self.data[~self.mask] = torch.randn((~self.mask).sum()).abs() * 0.05  # replace masked values with noise
197: 
198:   def __len__(self) -> int:
199:     return len(self.data)
200: 
201:   def __getitem__(self, idx):
202:     return self.data[idx], self.mask[idx]
203: 
204: _debug_loss_bits = None
205: def masked_loss(
206:     x:            torch.Tensor,
207:     pred:         torch.Tensor,
208:     mask:         torch.Tensor,
209:     mu:           torch.Tensor,
210:     logvar:       torch.Tensor,
211:     gate_probs:   torch.Tensor|None = None,
212:     soft_weights: torch.Tensor|None = None,
213:     mask_type2:   torch.Tensor|None = None,
214:     beta:         float=1.0,
215:     gamma:        float=0.2,
216:     free_bits:    float=0.1,
217:     lambda_entropy:float=0.2,
218:     kl_override: torch.Tensor|None = None,
219: ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
220:   """
221:   Masked ELBO
222:   mask:       0=known, 1=masked/missing
223:   mask_type2: 0=observed/missing, 1=masked  (only deliberately masked values)
224:   beta:       VAE balance (low beta: focus on reconstruction quality; high beta: focus on structured latent space representation)
225:   gamma:      imputation balance (recon loss on known vs. recon loss on imputed positions)
226: 
227:   Handles both standard VAE (mu:(B,D) and MoVE (mu:(K,B,D)))
228:   """
229:   global _debug_loss_bits  # DEBUG ONLY
230: 
231:   assert(x.shape[-1]==pred.shape[-1])
232:   #x = x.reshape((x.shape[0],2,-1))[:,0,:]
233: 
234:   # MSE over observed positions
235:   _mask = ~mask
236:   sq_err = (pred - x).pow(2)
237:   recon_loss = (sq_err * _mask).sum() / _mask.sum().clamp(min=1)
238: 
239:   # Type 2 reconstruction
240:   if mask_type2 is not None:
241:     recon_loss_type2 = (sq_err * mask_type2.float()).sum() / mask_type2.sum().clamp(min=1)
242:     recon_loss += recon_loss_type2 * gamma
243:   else:
244:     recon_loss_type2 = torch.tensor(0.0)
245: 
246:   # KL:  -½ Σ (1 + log σ² - μ² - σ²)
247:   if kl_override is not None:
248:     # Pre-computed KL (e.g. VampPrior); free_bits clamping is skipped because
249:     # the VampPrior KL is a difference of log-densities and does not decompose
250:     # per dimension.
251:     kl_loss = kl_override
252: 
253:   elif mu.ndim==2:
254:     #kl_loss   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
255:     kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
256:     kl_loss    = kl_per_dim.clamp(min=free_bits).sum(dim=1).mean()
257:     #_debug_loss_bits = kl_per_dim.detach()  # DEBUG ONLY
258: 
259:   elif mu.ndim==3:
260:     # for MoVE: weighted KL over the components
261:     assert(soft_weights is not None)
262:     kl_per_dim  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) # (K,B,D)
263:     kl_per_comp = kl_per_dim.clamp(min=free_bits).sum(dim=-1)
264:     kl_loss     = (kl_per_comp.T * soft_weights).sum(dim=0).mean()  # weighted sum over components, then average over batch
265: 
266:   else:
267:     assert(False)
268: 
269:   # entropy loss - encourage uniform gate assignment
270:   if gate_probs is not None:
271:     avg_gate = gate_probs.mean(dim=0) # (K,)
272:     gating_entropy_loss = (avg_gate * avg_gate.log()).sum() # encourage uniform load
273:   else:
274:     gating_entropy_loss = torch.tensor(0.0)
275: 
276:   loss = recon_loss + beta * kl_loss + lambda_entropy * gating_entropy_loss
277:   return loss, recon_loss, recon_loss_type2, kl_loss
278: 
279: 
280: def random_points_on_hypersphere(
281:     num_points:    int,
282:     K:             int,
283:     N:             int,
284:     concentration: float = 0.0,
285:     n_clusters:    int   = 1,
286:     centre:        bool  = True,
287:     device:        str   = 'cpu',
288: ) -> tuple[torch.Tensor, torch.Tensor]:
289:   """
290:   Generate random K-dimensional points constrained to an N-dimensional
291:   hypersphere (N < K), rotated so all K dimensions carry signal.
292: 
293:   Args:
294:       num_points:    number of points to generate.
295:       K:             dimensionality of the output space.
296:       N:             intrinsic dimensionality of the manifold (N < K).
297:       concentration: von Mises-Fisher concentration kappa >= 0.
298:                      0  => uniform on the sphere (original behaviour).
299:                      >0 => points are pulled towards their cluster centre;
300:                           higher values produce tighter clusters.
301:       n_clusters:    number of modes.  Each point is assigned to one cluster
302:                      uniformly at random, then sampled from a vMF distribution
303:                      centred on that cluster's mean direction.
304:       centre:        if True (default), subtract the empirical column mean from
305:                      the rotated output.  This ensures zero-mean cell states so
306:                      that non-interacting genes do not acquire a spurious DC
307:                      correlation from the offset of the cluster centres.
308:       device:        torch device string.
309: 
310:   Returns:
311:       rotated : (num_points, K) float tensor of cell-state coordinates.
312:       labels  : (num_points,)   int tensor of cluster assignments (0-indexed).
313:   """
314:   import warnings
315:   assert N < K, "N must be strictly less than K"
316: 
317:   if n_clusters > 1 and concentration < 1e-6:
318:     warnings.warn(
319:         "random_points_on_hypersphere: n_clusters > 1 but concentration ~ 0. "
320:         "Cluster centres have no effect on the distribution at concentration=0; "
321:         "all points will be drawn uniformly regardless of cluster assignment.",
322:         stacklevel=2,
323:     )
324: 
325:   # --- 1. Sample n_clusters centre directions on S^N ---
326:   centres = torch.randn(n_clusters, N + 1, device=device)
327:   centres = centres / torch.norm(centres, dim=-1, keepdim=True)  # (n_clusters, N+1)
328: 
329:   # --- 2. Assign each point to a cluster uniformly at random ---
330:   labels = torch.randint(0, n_clusters, (num_points,), device=device)  # (num_points,)
331: 
332:   # --- 3. Sample points from vMF(centre[label], concentration) ---
333:   if concentration < 1e-6:
334:     # Uniform on S^N — identical to the original implementation
335:     coords = torch.randn(num_points, N + 1, device=device)
336:     coords = coords / torch.norm(coords, dim=-1, keepdim=True)
337:   else:
338:     mu = centres[labels]  # (num_points, N+1) — each point's cluster centre
339: 
340:     # Tangent-plane perturbation: sample isotropic noise then project out the
341:     # component along mu so that z lies in the tangent plane at mu.
342:     z = torch.randn(num_points, N + 1, device=device)
343:     z = z - (z * mu).sum(dim=-1, keepdim=True) * mu   # project onto tangent plane
344:     z = z / torch.norm(z, dim=-1, keepdim=True)        # unit tangent vector
345: 
346:     # Mix: high concentration => near mu; concentration -> 0 => dominated by z (uniform)
347:     coords = mu + z / concentration
348:     coords = coords / torch.norm(coords, dim=-1, keepdim=True)  # renormalise to sphere
349: 
350:   # --- 4. Embed in K-dimensional space ---
351:   embedded = torch.zeros(num_points, K, device=device)
352:   embedded[:, :N+1] = coords
353: 
354:   # --- 5. Random rotation in R^K (Haar-uniform via QR) ---
355:   random_matrix = torch.randn(K, K, device=device)
356:   Q, R = torch.linalg.qr(random_matrix)
357:   signs = torch.sign(torch.diag(R))   # fix sign ambiguity -> uniform over SO(K)
358:   Q = Q * signs
359: 
360:   rotated = embedded @ Q.T  # (num_points, K)
361: 
362:   # --- 6. Optionally remove empirical mean ---
363:   if centre:
364:     rotated = rotated - rotated.mean(dim=0, keepdim=True)
365: 
366:   return rotated, labels
367: 
368: 
369: def make_synthetic_data4(
370:     block_state:    torch.Tensor,
371:     n_genes:        int = 200,
372:     n_blocks:       int = 5,
373:     rho_within:     float = 0.5,
374:     means:          torch.Tensor|None = None,
375:     stds:           torch.Tensor|None = None,
376:     allow_negative: bool = False,
377:     missing_rate:   float = 0.1,
378:     seed:           int = 42,
379:     device:         str|None = None,
380: ) -> tuple[torch.Tensor, np.ndarray]:
381:   """
382:   Explicit-factor variant of make_synthetic_data2/3 with caller-supplied block states.
383: 
384:   Each sample receives its block-state vector from `block_state`; gene-specific noise
385:   is sampled internally.  There is no global shared factor.
386: 
387:   For block k spanning genes [s, e):
388: 
389:       X[:, s:e] = sqrt(rho_within) * block_state[:, k:k+1]
390:                   + sqrt(1 - rho_within) * noise[:, s:e]
391: 
392:   If block_state[:, k] ~ N(0,1), the within-block correlation is rho_within.
393:   Between-block correlation is fully determined by the correlation structure the
394:   caller puts into block_state (no n_genes^2 covariance matrix is formed).
395: 
396:   Note: This version has no explicit support for discrete cell types (n_types==1)
397: 
398:   Masking (type 1): missing as np.nan.
399: 
400:   Returns:
401:       data: torch.Tensor(n_cells, n_genes)
402:       block_bounds
403:   """
404:   generator = torch.Generator(device=device)
405:   if seed is not None:
406:       generator.manual_seed(seed)
407: 
408:   #if means is not None or stds is not None:
409:   dtype = torch.float32
410:   mu    = means.to(dtype=dtype, device=device) if means is not None else torch.zeros(n_genes, dtype=dtype, device=device)
411:   sigma = stds.to( dtype=dtype, device=device) if stds  is not None else torch.ones( n_genes, dtype=dtype, device=device)
412: 
413:   assert block_state.shape[1] == n_blocks, (
414:       f"block_state has {block_state.shape[1]} columns but n_blocks={n_blocks}"
415:   )
416:   n_cells = block_state.shape[0]
417:   block_state = block_state.to(device=device)
418: 
419:   block_bounds = np.linspace(0, n_genes, n_blocks + 1).round().astype(int)
420: 
421:   std_block = np.sqrt(rho_within)
422:   std_noise = np.sqrt(max(1.0 - rho_within, 0.0))
423: 
424:   noise = torch.randn(n_cells, n_genes, device=device, generator=generator)
425:   X = torch.zeros(n_cells, n_genes, device=device)
426: 
427:   for k in range(n_blocks):
428:       s = block_bounds[k]
429:       e = block_bounds[k + 1]
430: 
431:       # Common component for this block (broadcast over genes)
432:       X[:, s:e] = std_block * block_state[:, k : k + 1] + std_noise * noise[:, s:e]
433: 
434:       if allow_negative:
435:           signs = torch.randint(0, 2, (e - s,), generator=generator, device=device) * 2 - 1
436:           X[:, s:e] = X[:, s:e] * signs.unsqueeze(0)
437: 
438:   # Map latent normal values to NB means and sample counts
439:   mean_nb = torch.exp(X * sigma.unsqueeze(0) + mu.unsqueeze(0))
440:   r = torch.tensor(10.0, device=device)  # use constant over-dispersion factor for all genes
441:   probs = r / (r + mean_nb)
442:   counts = torch.distributions.NegativeBinomial(total_count=r, probs=probs).sample()
443:   X = torch.log1p(counts)
444: 
445:   dropout = torch.rand(X.shape, generator=generator, device=device) < missing_rate
446:   X[dropout] = float('nan')
447: 
448:   return X, block_bounds
449: 
450: 
451: 
452: # Define group hierarchy, with a 'membership fraction' associated with each group
453: groups_def = \
454:      {0: (0.20, { 6:(0.7,{23:(0.4,{})}),
455:                   7:(0.5,{24:(0.7,{})})}),
456:       1: (0.15, { 8:(0.6,{25:(0.7,{})}), 
457:                   9:(0.7,{26:(0.5,{})})}),
458:       2: (0.30, {10:(0.8,{27:(0.5,{})}), 
459:                  11:(0.4,{28:(0.7,{})}), 
460:                  12:(0.4,{})}),
461:       3: (0.30, {13:(0.4,{}), 
462:                  14:(0.3,{29:(0.6,{})}), 
463:                  15:(0.5,{})}),
464:       4: (0.35, {16:(0.6,{30:(0.5,{})}), 
465:                  17:(0.2,{})}),
466:       5: (0.20, {18:(0.3,{}),
467:                  19:(0.6,{}),
468:                  20:(0.4,{31:(0.6,{})}),
469:                  21:(0.4,{}),
470:                  22:(0.4,{})})
471:       }
472: 
473: def _draw_group_membership(g:dict, device=None):
474:   ret = []
475:   for gid, ginfo in g.items():
476:     membership_prob, descendants = ginfo
477: 
478:     if torch.rand(1, device=device).item() >= membership_prob: continue
479: 
480:     ret.append(gid)
481:     ret.extend( _draw_group_membership(descendants, device=device) )
482:   return tuple(ret)
483: 
484: def make_synthetic_data5(
485:     cell_state:     torch.Tensor,
486:     n_genes:        int = 200,
487:     corr_scale:     float = 1.0,
488:     noise_scale:    float = 0.001,
489:     n_gene_groups:  int = 32,  # must match groups_def
490:     means:          torch.Tensor|None = None,
491:     stds:           torch.Tensor|None = None,
492:     allow_negative: bool = False,
493:     missing_rate:   float = 0.1,
494:     seed:           int = 42,
495:     device:         str|None = None,
496:     groups_def:     dict = groups_def, # must match n_gene_groups
497: ) -> tuple[torch.Tensor, np.ndarray]:
498:   """
499:   Hierarchical sparse-interaction synthetic gene-count data.
500: 
501:   Genes are stochastically assigned to groups in a 3-level hierarchy
502:   (groups_def).  Each cell's expression is driven by a cell_state vector
503:   that lives on a low-dimensional manifold (typically produced by
504:   random_points_on_hypersphere).
505: 
506:   Signal model (latent space, before NB sampling):
507: 
508:       group_membership : (n_genes, n_gene_groups)  -- sparse binary
509:       cell_state       : (n_cells, n_gene_groups)  -- manifold coords
510: 
511:       latent[:, g] = corr_scale * (group_membership @ cell_state.T)[g, :]
512:                      * influence_weight[g]           -- Gamma(10,10) ~ 1
513:                      + noise                         -- N(0, noise_scale)
514: 
515:   Then latent values are mapped to NegativeBinomial counts and log1p
516:   transformed, exactly as in make_synthetic_data4.
517: 
518:   Args:
519:       cell_state:    (n_cells, n_gene_groups) tensor of cell states.
520:       n_genes:       number of genes to simulate.
521:       corr_scale:    scales cell_state before the projection; controls the
522:                      overall signal amplitude / within-module correlation.
523:       noise_scale:   std of gene-level Gaussian noise added before NB sampling.
524:       n_gene_groups: number of groups (must match the IDs in groups_def).
525:       means:         optional per-gene offset (n_genes,) applied before exp().
526:       stds:          optional per-gene scale  (n_genes,) applied before exp().
527:       allow_negative: if True, each gene gets an independent ±1 sign flip.
528:       missing_rate:  fraction of entries set to NaN (MCAR dropout).
529:       seed:          RNG seed.
530:       device:        torch device string.
531:       groups_def:    hierarchy definition dict; must cover exactly n_gene_groups IDs.
532: 
533:   Returns:
534:       X             : (n_cells, n_genes) float32 tensor, log1p counts, NaN for missing.
535:       group_membership_np : (n_genes, n_gene_groups) numpy bool array.
536:   """
537:   generator = torch.Generator(device=device)
538:   if seed is not None:
539:       generator.manual_seed(seed)
540: 
541:   # n_cells is authoritative from the supplied tensor
542:   n_cells = cell_state.shape[0]
543:   assert cell_state.shape[1] == n_gene_groups, (
544:       f"cell_state has {cell_state.shape[1]} columns but n_gene_groups={n_gene_groups}"
545:   )
546: 
547:   # --- sparse gene-group membership matrix ---
548:   # group_membership[i, k] = 1 iff gene i belongs to group k
549:   group_membership = torch.zeros((n_genes, n_gene_groups), requires_grad=False)
550:   for gene_idx in range(n_genes):
551:     m = _draw_group_membership(groups_def)
552:     for mi in m:
553:       group_membership[gene_idx, mi] = 1.0
554:   group_membership = group_membership.to(device=device)
555: 
556:   ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
557:   #block_state_both = block_state_both[0,:].unsqueeze(0).expand(block_state_both.shape[0],block_state_both.shape[1])  # copy the first cell's state for all cells
558:   #block_state_both[:,:2] = 1 + torch.randn(10_000,2)*0.1
559:   #block_state_both[:,1:] = 0   # only group 1 has any genes
560:   ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
561: 
562:   # Scale cell state by corr_scale so the caller controls signal amplitude
563:   cell_state = cell_state.to(device=device) * corr_scale  # (n_cells, n_gene_groups)
564: 
565:   # Per-(gene, cell) interaction-strength weights drawn from Gamma with mean~1
566:   # Gamma(concentration=10, rate=10) => mean=1, std~0.32; tight enough to be
567:   # roughly uniform but non-degenerate.
568:   influence_dist = torch.distributions.Gamma(
569:       torch.tensor([10.0], device=device),
570:       torch.tensor([10.0], device=device),
571:   )
572: 
573:   # (n_genes, n_cells): project cell states through sparse membership
574:   out = group_membership @ cell_state.T                   # (n_genes, n_cells)
575:   out *= influence_dist.sample(out.shape).squeeze(-1)     # per-entry strength
576:   out += torch.randn(out.shape, device=device,
577:                      generator=generator) * noise_scale   # gene-level noise
578: 
579:   # Transpose to (n_cells, n_genes) -- codebase convention
580:   X = out.T.contiguous()   # (n_cells, n_genes)
581: 
582:   # Optional per-gene sign flips (makes some genes anti-correlated within a group)
583:   if allow_negative:
584:       signs = (torch.randint(0, 2, (n_genes,),
585:                              generator=generator, device=device) * 2 - 1).float()
586:       X = X * signs.unsqueeze(0)
587: 
588:   # Per-gene offset / scale (consistent with make_synthetic_data4)
589:   dtype = torch.float32
590:   mu    = means.to(dtype=dtype, device=device) if means is not None \
591:           else torch.zeros(n_genes, dtype=dtype, device=device)
592:   sigma = stds.to( dtype=dtype, device=device) if stds  is not None \
593:           else torch.ones( n_genes, dtype=dtype, device=device)
594: 
595:   # Map latent values to NegativeBinomial counts, then log1p-normalise
596:   mean_nb = torch.exp(X * sigma.unsqueeze(0) + mu.unsqueeze(0))
597:   r       = torch.tensor(10.0, device=device)
598:   probs   = r / (r + mean_nb)
599:   counts  = torch.distributions.NegativeBinomial(total_count=r, probs=probs).sample()
600:   X       = torch.log1p(counts)
601: 
602:   # MCAR dropout -> NaN
603:   dropout = torch.rand(X.shape, generator=generator, device=device) < missing_rate
604:   X[dropout] = float('nan')
605: 
606:   return X, group_membership.cpu().numpy()
607: 

(End of file - total 607 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/michael/src/vae-test/vae-test.py"
}
```

**Output:**
```
<path>/home/michael/src/vae-test/vae-test.py</path>
<type>file</type>
<content>
1: # Dual masking:
2: # Type 1. Some values in each sample are missing from the generated training/test data-sets. Loss cannot be calculated for them(because the target is unknown), and they will be imputed based on the values of other variables in that sample.  
3: # Type 2. Some values in each sample are masked during each training epoch (or during evaluation). Loss can be calculated for them.
4: # The model sees both types as masked values - the difference is only in the loss.  
5: # 
6: # TODOs:
7: # 1. Fix masking  
8: # 2. Fix missing value initialization (mean)  
9: # 3. Use negative binomial distribution, Zero-Inflated Negative Binomial (ZINB) Loss, log(1+x)+MSE  
10: # 3b. Fix distribution shift on reconstructed output  
11: # 4. Replace mask concatenatenation with canonical style  
12: # 5. Use type2_mask in loss, add lambda to balance loss on masked/unmasked data  
13: # 
14: # --
15: # 
16: # The model has to learn two separate (but linked) things:
17: # 1. The sparse but strong correlation structure between genes that allows imputing missing values
18: # 2. The implicit "sample state" that allows different samples to be imputed differently  
19: 
20: import torch
21: import torch.nn as nn
22: import torch.nn.functional as F
23: import torch.optim as optim
24: import torch.autograd as autograd
25: from torch.utils.data import Dataset, DataLoader
26: import numpy as np
27: import matplotlib.pyplot as plt
28: from tqdm import tqdm
29: from synthetic_data import make_synthetic_data4, make_synthetic_data5, random_points_on_hypersphere
30: from models import GeneExpressionVAE, MoVEVAE, VampPriorVAE, VAEBase, epoch_vae, get_random_mask, BetaSSMController, World, Controller, rl_training_loop, plot_rl_results, MeansModel, ImputationTransformer, epoch_transformer, impute_transformer, make_log_bin_edges, model_factories
31: import models
32: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
33: base_cfg = {'n_blocks': 32, 'rho_within': 0.8,
34:           'n_genes': 200, 'n_cells': 2_000, 'missing_rate': 0.05}
35: vtc = {'max_epochs': 50, 'batch_size': 200,
36:         'mask_fraction': 0.05, 'lr': 3e-4, 'n_pseudo': 50}
37: vlc = {'gamma_recon': 4.0, 'min_free_bits': 0.01, 'lambda_entropy': 0.01}
38: 
39: world      = World(vtc, vlc, base_cfg, beta_ref=1e-4, lambda_K=0.1)
40: ssm        = BetaSSMController(obs_dim=12, hidden_dim=32).to(device)
41: controller = Controller(ssm, lr=1e-3)
42: 
43: controller, replay_buffer, episode_log = rl_training_loop(
44:   world, controller,
45:   n_episodes=30,
46:   n_pretrain_episodes=10,   # collect 10 waveform episodes first
47:   pretrain_steps=200,       # world-model steps before RL begins
48:   )
49: plot_rl_results(episode_log)
50: # problem
51: 
52: #_train, _test = GeneExpressionDataset(
53: #    make_synthetic_data(n_cells, n_genes, n_types=5, missing_rate=0.05)
54: #)
55: #data_train, mask_train = _train
56: #data_test,  mask_test  = _test
57: 
58: selected_problem='standard4'
59: 
60: if selected_problem=='easy':
61:   n_cells =   500
62:   n_genes =   20
63: 
64:   ret = make_synthetic_data2(n_cells, n_genes, n_blocks=2, n_types=1, rho_within=0.7, missing_rate=0.02)
65: 
66: elif selected_problem=='easy_plus':
67:   n_cells =   2_000
68:   n_genes =   20
69: 
70:   ret = make_synthetic_data2(n_cells, n_genes, n_blocks=3, n_types=3, rho_within=0.7, missing_rate=0.02)
71: 
72: elif selected_problem=='standard':
73: 
74:   n_cells =   2_000
75:   n_genes =     200
76: 
77:   ret = make_synthetic_data2(n_cells, n_genes, n_blocks=18, n_types=4, rho_within=0.7, missing_rate=0.05)
78: 
79: elif selected_problem=='standard4':
80: 
81:   n_cells  = 2_000
82:   n_genes  =   200
83:   n_blocks =    32
84: 
85:   if False:
86:     # draw block state randomly (i.e., all cells come from the same distribution)
87:     block_state_train=torch.randn(n_cells, n_blocks)
88:     block_state_test =torch.randn(n_cells, n_blocks)
89: 
90:   else:
91:     block_state_both, _ = random_points_on_hypersphere(
92:         num_points=n_cells,
93:         K=n_blocks,
94:         N=16, device=device)  # generate K block-state values (for each cell), constrained to lie on a N-d hypershere (N DoF)
95:     corr_scale = 4 # 0.7   # highest correlation between blocks (approximately)
96:     scale = block_state_both.abs().max(dim=0).values
97:     block_state_both /= scale / corr_scale  # normalize (otherwise higher-K problems have weaker correlations)
98:     assert(block_state_both.shape==(n_cells, n_blocks))
99: 
100:   ret = []
101:   for block_state in (block_state_both, block_state_both):  # generate train/test data from the same population of cell-states
102: 
103:     _data, _ = make_synthetic_data4(
104:         block_state=block_state,
105:         n_genes=n_genes,
106:         n_blocks=n_blocks,
107:         rho_within=0.8,
108:         allow_negative=True,
109:         missing_rate=0.05
110:     )
111:     ret.append( _data )
112:     ret.append(torch.zeros(n_cells)) # all cells belong to the same type...
113:   ret.append(None)
114: 
115: else:
116:   assert(False)
117: 
118: data_train, types_train, data_test, types_test, _  = ret
119: 
120: loader_test           = DataLoader(data_test,  batch_size=n_cells, shuffle=False)
121: loader_train_no_batch = DataLoader(data_train, batch_size=n_cells, shuffle=True)  # for evaluation
122: 
123: # ---------------------------------------------------------------------------
124: # load_model: restore a pre-trained model from a checkpoint file.
125: #
126: # Expected checkpoint format (saved with torch.save):
127: #   {
128: #     'model_type': <key from model_factories>,
129: #     'state_dict': <model.state_dict()>,
130: #   }
131: # ---------------------------------------------------------------------------
132: 
133: def load_model(path, factories=model_factories, device=device):
134:     """Load a pre-trained model from *path* and return it in eval mode.
135: 
136:     Expected checkpoint format (saved with torch.save):
137:         {
138:             'model_type': <key in model_factories>,
139:             'state_dict': <model.state_dict()>,
140:         }
141: 
142:     Post-load fixups applied automatically:
143:       - MeansModel  : means re-fitted from data_train (the parameter has no
144:                       meaning outside the dataset it was computed on).
145:       - ImputationTransformer : bin_edges attached from it_bin_edges so that
146:                       impute() can find them without a separate argument.
147:                       (Call load_model only after it_bin_edges is defined.)
148:     """
149:     ckpt       = torch.load(path, map_location=device)
150:     model_type = ckpt['model_type']
151:     if model_type not in factories:
152:         raise ValueError(f"Unknown model_type '{model_type}'. "
153:                          f"Available: {list(factories)}")
154:     m = factories[model_type]().to(device)
155:     m.load_state_dict(ckpt['state_dict'])
156:     m.eval()
157:     if isinstance(m, MeansModel):
158:         m.means.data = data_train.nanmean(dim=0).to(device)
159:     if isinstance(m, ImputationTransformer):
160:         m.bin_edges = it_bin_edges
161:     return m
162: 
163: # ---------------------------------------------------------------------------
164: # Active model for the training loop below.
165: # Swap the key to select a different architecture.
166: # ---------------------------------------------------------------------------
167: 
168: _active_model_type = 'GeneExpressionVAE'
169: model = model_factories[_active_model_type]().to(device)
170: if isinstance(model, MeansModel):
171:     model.means.data = data_train.nanmean(dim=0).to(device)
172: learning_graph_train = []
173: learning_graph_test  = []
174: next_epoch = 0
175: #opt = optim.AdamW(model.parameters(), lr=3e-4)
176: beta_final = 5.0e-2 # 0.4 # 1e-4 # 1e-2
177: #beta_sched = 0.5 * beta_final * (1 - np.cos(np.pi*np.linspace(0.002,0.999,1500)) )
178: #beta_sched = np.hstack( (np.full((100,), fill_value=1e-6),
179: #                    0.5 * beta_final * (1 - np.cos(np.pi*np.linspace(0.01,0.999,100)) ) ) )
180: beta_sched = torch.full( size=(200,), fill_value=beta_final )
181: plt.plot(beta_sched)
182: plt.gca().set_yscale('log');
183: max_epocs     =    100 # 150 #100
184: batch_size    =    200 # 200 #200
185: lr            =   3e-4 # 5e-4
186: min_free_bits =   0.005 # 0.8
187: gamma         =   4.0 # 1.6
188: mask_fraction =   0.05
189: lambda_entropy=   0.01 # 0.5
190: 
191: 
192: opt = optim.AdamW([{'params':[p for n,p in model.named_parameters() if 'pseudo' not in n], 'lr':lr}],
193:                   lr=lr
194:                   )
195: #opt = optim.AdamW([{'params':[p for n,p in model.named_parameters() if 'pseudo' not in n], 'lr':lr},
196: #                   {'params':model.pseudo_inputs, 'lr':1.0*lr}],
197: #                  lr=lr
198: #                  )
199: 
200: loader_train          = DataLoader(data_train, batch_size=batch_size, shuffle=True)
201: 
202: for g in opt.param_groups: g['lr'] = lr
203: 
204: #for x, mask in loader_train:
205: #  pred_train, mu, logvar = model(x)
206: 
207: for i in tqdm(range(max_epocs)):
208:   beta = beta_sched[min(next_epoch,beta_sched.shape[0]-1)]
209: 
210:   train_loss, train_norm, loss1, loss1_type2, loss2 = epoch_vae(
211:       model,
212:       loader_train,
213:       opt,
214:       mask_fraction=mask_fraction,
215:       beta=beta,
216:       gamma=gamma,
217:       min_free_bits=min_free_bits,
218:       lambda_entropy=lambda_entropy)
219: 
220:   if isinstance(model, VampPriorVAE):
221:     effk = model._effective_K
222:   else:
223:     effk = None
224: 
225:   learning_graph_train.append( (next_epoch, train_loss, train_norm, loss1, loss2, loss1_type2, effk) )
226: 
227:   if next_epoch%10 == 0:
228:     test_loss, _a, _b, _c, _d = epoch_vae(
229:         model,
230:         loader_test,
231:         mask_fraction=mask_fraction,
232:         beta=beta,
233:         gamma=gamma,
234:         min_free_bits=min_free_bits,
235:         lambda_entropy=lambda_entropy)
236: 
237:     learning_graph_test.append( (next_epoch, test_loss, None, None, None) )
238: 
239:   next_epoch += 1
240: 
241: 
242: # ---------------------------------------------------------------------------
243: # VAE imputation correlation sweep (uses unified impute())
244: # ---------------------------------------------------------------------------
245: 
246: mask_fractions = np.exp(np.linspace(-0.35,-4.5,100+1))
247: 
248: ret = []
249: model.eval()
250: for frac in tqdm(mask_fractions):
251:     for x in loader_test:
252:         recon, type2_mask = impute(model, x, frac, device=device)
253:         if type2_mask.any():
254:             corr_data = np.vstack((x[type2_mask].numpy(),
255:                                    recon[type2_mask].numpy()))
256:             corr = np.corrcoef(corr_data, rowvar=True)[0, 1]
257:         else:
258:             corr = float('nan')
259:         ret.append((type2_mask.float().mean().item(), corr))
260: ret = np.array(ret)
261: plt.plot( ret[:,0], ret[:,1])
262: plt.gca().set_xscale('log')
263: plt.title('Imputed reconstruction')
264: plt.xlabel('Masked fraction')
265: plt.ylabel('Imputation correlation');
266: 
267: # Use the same data_train / data_test already built by the standard4 block above.
268: # (data_train and data_test are (n_cells, n_genes) float tensors with NaN for missing.)
269: it_loader_train = DataLoader(data_train, batch_size=it_batch_size, shuffle=True)
270: it_loader_test  = DataLoader(data_test,  batch_size=n_cells,       shuffle=False)
271: 
272: it_model = models._make_imputation_transformer().to(device)
273: # Attach bin_edges as a model attribute so load_model and impute() can access
274: # them without needing a separate argument.
275: it_model.bin_edges = models.it_bin_edges
276: 
277: it_opt = optim.AdamW(it_model.parameters(), lr=it_lr)
278: 
279: it_log_train = []
280: it_log_test  = []
281: _ep = 0  # epoch counter
282: it_max_epochs = 20
283: 
284: print('\n=== ImputationTransformer training ===')
285: for _ in tqdm(range(it_max_epochs)):
286:     tr_loss, tr_count_ce, tr_conf_ce, tr_gnorm = epoch_transformer(
287:         it_model, it_loader_train, it_bin_edges,
288:         opt=it_opt,
289:         mask_fraction=it_mask_frac,
290:         lambda_conf=it_lambda_conf,
291:         device=device,
292:     )
293:     it_log_train.append((_ep, tr_loss, tr_count_ce, tr_conf_ce, tr_gnorm))
294: 
295:     if _ep % 5 == 0:
296:         te_loss, te_count_ce, te_conf_ce, _ = epoch_transformer(
297:             it_model, it_loader_test, it_bin_edges,
298:             opt=None,
299:             mask_fraction=it_mask_frac,
300:             lambda_conf=it_lambda_conf,
301:             device=device,
302:         )
303:         it_log_test.append((_ep, te_loss, te_count_ce, te_conf_ce))
304:         print(f'  ep {_ep:3d}  '
305:               f'train_ce={tr_count_ce:.4f}  conf_ce={tr_conf_ce:.4f}  gnorm={tr_gnorm:.3f}  '
306:               f'| test_ce={te_count_ce:.4f}')
307:     _ep += 1
308: 
309: # Plot training curves
310: _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
311: _ep_tr = [r[0] for r in it_log_train]
312: _ep_te = [r[0] for r in it_log_test]
313: _axes[0].plot(_ep_tr, [r[2] for r in it_log_train], label='train count CE')
314: _axes[0].plot(_ep_te, [r[2] for r in it_log_test],  label='test count CE', marker='o', ms=4)
315: _axes[0].set_xlabel('Epoch'); _axes[0].set_ylabel('Cross-entropy')
316: _axes[0].set_title('ImputationTransformer -- count loss')
317: _axes[0].legend()
318: _axes[1].plot(_ep_tr, [r[3] for r in it_log_train], label='train conf CE', color='tab:orange')
319: _axes[1].set_xlabel('Epoch'); _axes[1].set_ylabel('Cross-entropy')
320: _axes[1].set_title('ImputationTransformer -- confidence loss')
321: _axes[1].legend()
322: plt.tight_layout()
323: plt.savefig('it_training_curves.png', dpi=120)
324: plt.show()
325: # ---------------------------------------------------------------------------------------------
326: # 7. Evaluation -- imputation correlation sweep (on type-2 masked positions) vs. mask fraction
327: # ---------------------------------------------------------------------------------------------
328: 
329: it_model.eval()
330: 
331: it_mask_fractions = np.exp(np.linspace(-0.35, -4.5, 10))
332: it_corr_results   = []
333: 
334: for _frac in tqdm(it_mask_fractions):
335:     for x_raw in it_loader_test:
336:         recon, type2_eval = impute(it_model, x_raw, _frac, device=device)
337:         if type2_eval.any():
338:             true_vals = x_raw[type2_eval].numpy()
339:             pred_vals = recon[type2_eval].numpy()
340:             corr = np.corrcoef(true_vals, pred_vals)[0, 1] if len(true_vals) > 1 else float('nan')
341:         else:
342:             corr = float('nan')
343:         it_corr_results.append((float(type2_eval.float().mean()), corr))
344: 
345: it_corr_arr = np.array(it_corr_results)
346: 
347: plt.plot( it_corr_arr[:,0],  it_corr_arr[:,1] );

(End of file - total 347 lines)
</content>
```

---

## Assistant (Plan · Claude Sonnet 4.6 · 4.3s)

Now let me look at the models file and the AGENTS.md for more context.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/michael/src/vae-test/AGENTS.md"
}
```

**Output:**
```
<path>/home/michael/src/vae-test/AGENTS.md</path>
<type>file</type>
<content>
1: # VAE Gene Expression Imputation
2: 
3: Single-file PyTorch research script (`vae-test.py`). No tests, no package structure. Modules not installable in this environment, so tests beyond syntax checking should not be run.
4: 
5: ## Known Issues
6: 

(End of file - total 6 lines)
</content>
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "/home/michael/src/vae-test/*.py"
}
```

**Output:**
```
No files found
```

---

## Assistant (Plan · Claude Sonnet 4.6 · 4.8s)

**Tool: bash**

**Input:**
```json
{
  "command": "ls /home/michael/src/vae-test/"
}
```

**Output:**
```
AGENTS.md
models.py
__pycache__
session-ses_09a0.md
synthetic_data.py
vae-test.py

```

---

## Assistant (Plan · Claude Sonnet 4.6 · 2.7s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/michael/src/vae-test/models.py"
}
```

**Output:**
```
<path>/home/michael/src/vae-test/models.py</path>
<type>file</type>
<content>
1: from abc import ABC, abstractmethod  # re-evaluate this
2: import math
3: import torch
4: import torch.nn as nn
5: import torch.nn.functional as F
6: import torch.optim as optim
7: import torch.autograd as autograd
8: from torch.utils.data import Dataset, DataLoader
9: import numpy as np
10: from synthetic_data import masked_loss, get_random_mask
11: 
12: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
13: 
14: def _block(in_dim:int, out_dim:int, dropout:float) -> nn.Sequential:
15:   return nn.Sequential(
16:       nn.Linear(in_dim, out_dim),
17:       nn.BatchNorm1d(out_dim),
18:       nn.ReLU(),
19:       nn.Dropout(dropout)
20:   )
21: 
22: def build_encoder(n_genes:int, encoder_dims:list[int], dropout:float) -> tuple[nn.Sequential, int]:
23:     """Returns encoder backbone and output dimension."""
24:     layers = []
25:     prev = n_genes * 2
26:     for h in encoder_dims:
27:         layers.append(_block(prev, h, dropout))
28:         prev = h
29:     return nn.Sequential(*layers)
30: 
31: 
32: def build_decoder(n_genes:int, latent_dim:int, decoder_dims:list[int],  dropout:float) -> nn.Sequential:
33:     """
34:     Returns decoder network.
35:     """
36:     # TODO use _block ?
37:     layers = []
38:     prev = latent_dim
39:     for h in decoder_dims:
40:         layers.extend([
41:             nn.Linear(prev, h),
42:             nn.BatchNorm1d(h),
43:             nn.ReLU(),
44:             nn.Dropout(dropout)
45:         ])
46:         prev = h
47:     layers.append(nn.Linear(prev, n_genes))
48:     layers.append(nn.ReLU())
49:     return nn.Sequential(*layers)
50: 
51: 
52: class VAEBase(nn.Module, ABC): #nn.Module
53:   """Shared encoder backbone for both single and mixture VAEs."""
54:   def __init__(self,
55:                 n_genes:int,
56:                 latent_dim:int,
57:                 hidden_dims:  list[int]|None = None,
58:                 encoder_dims: list[int]|None = None,
59:                 decoder_dims: list[int]|None = None,
60:                 dropout:float = 0.05):
61: 
62:     super().__init__()
63: 
64:     # hidden_dims | encoder_dims | decoder_dims | Note
65:     #-------------+--------------+--------------+-------------------------------
66:     #      +      |      -       |      -       | valid (hidden used for both)
67:     #      -      |      +       |      +       | valid
68:     #      -      |      -       |      -       | valid (default used)
69:     if encoder_dims is not None: assert hidden_dims is None
70:     if decoder_dims is not None: assert hidden_dims is None
71:     if hidden_dims is not None:
72:       assert encoder_dims is None and decoder_dims is None
73:       encoder_dims = hidden_dims
74:       decoder_dims = list(reversed(hidden_dims))
75:     else:
76:       if encoder_dims is None and decoder_dims is None:
77:         encoder_dims = [124, 64]
78:         decoder_dims = [64, 124]
79: 
80:     self.latent_dim   = latent_dim
81:     self.encoder_dims = encoder_dims
82:     self.decoder_dims = decoder_dims
83:     self.enc_out_dim  = encoder_dims[-1]  # encoder output dim (!= latent dim)
84:     self.n_genes      = n_genes           # input dimension (input features)
85:     self.dropout      = dropout
86: 
87:     # descendants will initialize model elements
88: 
89: 
90:   def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
91:     if self.training:
92:       std = torch.exp(0.5 * logvar)
93:       return mu + torch.randn_like(std) * std
94:     else:
95:       return mu
96: 
97:   #def get_encoder_features(self, x:torch.Tensor) -> torch.Tensor:
98:   #    """Get encoder hidden representation before mu/logvar."""
99:   #    return self.encoder(x)
100: 
101: class GeneExpressionVAE(VAEBase):
102:   """
103:   Standard VAE for gene expression imputation.
104:   """
105:   def __init__(self,
106:                 n_genes:int,
107:                 latent_dim:int = 8,
108:                 hidden_dims:  list[int]|None = None,
109:                 encoder_dims: list[int]|None = None,
110:                 decoder_dims: list[int]|None = None,
111:                 dropout:float = 0.05):
112: 
113:     super().__init__(n_genes,
114:                       latent_dim,
115:                       hidden_dims,
116:                       encoder_dims,
117:                       decoder_dims,
118:                       dropout)
119: 
120:     # Build encoder
121:     self.encoder = build_encoder(n_genes, encoder_dims, dropout)
122: 
123:     # Build latent
124:     self.fc_mu     = nn.Linear(self.enc_out_dim, latent_dim)
125:     self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)
126: 
127:     # Build decoder
128:     self.decoder = build_decoder(n_genes, latent_dim, decoder_dims, dropout)
129: 
130: 
131:   def encode(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
132:     h = self.encoder(x)
133:     return h, self.fc_mu(h), self.fc_logvar(h)
134: 
135:   def decode(self, z:torch.Tensor) -> torch.Tensor:
136:     return self.decoder(z)
137: 
138:   def forward(self, x:torch.Tensor):
139:     _, mu, logvar = self.encode(x)
140:     z = self.reparameterize(mu, logvar)
141:     recon = self.decode(z)
142:     return recon, mu, logvar, None, None, None
143: 
144: class MoVEVAE(VAEBase):
145:   """
146:   Mixture of Variational Encoders using GeneExpressionVAE decoders.
147: 
148:   Architecture:
149:     - Shared encoder backbone (from VAEBase)
150:     - Per-component μ/logvar heads (component-specific recognition)
151:     - Per-component decoders (component-specific generative model)
152:     - Gating network for component selection
153:   """
154:   def __init__(self,
155:                n_genes:int,
156:                latent_dim:int=8,
157:                n_components:int=3,
158:                gating_net_dim:int=32,
159:                hidden_dims:  list[int]|None = None,
160:                encoder_dims: list[int]|None = None,
161:                decoder_dims: list[int]|None = None,
162:                dropout:float = 0.05):
163: 
164:     super().__init__(n_genes, latent_dim, hidden_dims, encoder_dims, decoder_dims, dropout)
165: 
166:     self.n_components = n_components
167: 
168:     #self.component_encoders = nn.ModuleList([
169:     #    nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
170:     #])
171:     #self.component_encoders_logvar = nn.ModuleList([
172:     #    nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
173:     #])
174: 
175:     # Build encoder
176:     self.encoder = build_encoder(n_genes, encoder_dims, dropout)
177: 
178:     # Build latent
179:     self.fc_mu     = nn.ModuleList([
180:         nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
181:     ])
182:     self.fc_logvar = nn.ModuleList([
183:         nn.Linear(self.enc_out_dim, latent_dim) for _ in range(n_components)
184:     ])
185: 
186:     # Build gating network
187:     self.gating = nn.Sequential(
188:         nn.Linear(self.enc_out_dim, gating_net_dim),
189:         nn.ReLU(),
190:         nn.Linear(gating_net_dim,    n_components)
191:     )
192: 
193:     # Build decoders
194:     self.decoders = nn.ModuleList([
195:         build_decoder(n_genes, latent_dim, decoder_dims, dropout)
196:         for _ in range(n_components)
197:     ])
198: 
199: 
200:   #def _encode_single_ecomponent(self, h:torch.Tensor, component:int) -> tuple[torch.Tensor, torch.Tensor]:
201:   #  """Encode for a specific component."""
202:   #  return self.component_encoders[component](h), self.component_encoders_logvar[component](h)
203: 
204:   def encode(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
205:     h = self.encoder(x)
206:     #return h, self.fc_mu(h), self.fc_logvar(h)
207:     return (h,
208:       torch.stack( [mu(h)     for mu     in self.fc_mu    ] ),
209:       torch.stack( [logvar(h) for logvar in self.fc_logvar] ))
210: 
211:   def decode(self, z:torch.Tensor, component:int) -> torch.Tensor:
212:     """Decode using specific component."""
213:     return self.decoders[component](z)
214: 
215:   def forward(self, x:torch.Tensor):
216:     """
217:     Returns: (recons_all_components, mu, logvar, gate_probs, soft_weights, selected_component)
218:     """
219:     h, mu, logvar = self.encode(x)
220: 
221:     gate_logits = self.gating(h)
222:     gate_probs = F.softmax(gate_logits, dim=-1)
223: 
224:     if self.training:
225:         # Gumbel-Softmax: straight-through estimator
226:         gumbel = -torch.log(-torch.log(torch.rand_like(gate_probs) + 1e-20) + 1e-20)
227:         gate_logits_st = gate_logits + gumbel
228:         soft_weights = F.softmax(gate_logits_st, dim=-1)  # (B, K)
229:         k_indices = soft_weights.argmax(dim=-1)  # for logging
230:         hard = F.one_hot( k_indices, self.n_components).float()
231:         soft_weights = hard - soft_weights.detach() + soft_weights
232:     else:
233:         k_indices = gate_probs.argmax(dim=-1)
234:         soft_weights = gate_probs
235: 
236: 
237:     #z = self.reparameterize(mu, logvar)
238: 
239:     #recons_all = torch.stack([self.decoders[k](z) for k in range(self.n_components)])
240:     recons_all = torch.stack([
241:         self.decoders[k]( self.reparameterize(mu[k], logvar[k]))
242:         for k in range(self.n_components)
243:         ]).permute(1, 0, 2)  # (B,K,G)
244:     batch_indices = torch.arange(x.shape[0], device=x.device)
245:     #print(f'recons_all:{recons_all.shape} batch_indices:{batch_indices.shape}')
246:     #recons = recons_all[k_indices, batch_indices]
247:     recons = (recons_all * soft_weights.unsqueeze(-1)).sum(dim=1)  # marginalize over k decoders
248: 
249:     return recons, mu, logvar, gate_probs, soft_weights, k_indices
250: 
251: 
252: def gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
253:     """
254:     Element-wise log N(z; mu, exp(logvar)).
255:     Supports broadcasting: z/mu/logvar can be (..., D).
256:     Returns a tensor of the same shape as inputs (before any .sum()).
257:     """
258:     return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp())
259: 
260: class VampPriorVAE(VAEBase):
261:   """
262:   VAE with Variational Mixture of Posteriors (VampPrior).
263: 
264:   The prior p(z) = (1/K) sum_k q_phi(z | u_k) is a mixture of K encoder
265:   posteriors evaluated at K learnable pseudo-inputs u_k (in gene-expression
266:   space).  The KL divergence KL(q(z|x) || p(z)) is estimated with a single
267:   Monte-Carlo sample z ~ q(z|x) using log-sum-exp over the K components.
268: 
269:   Architecture is identical to GeneExpressionVAE (single encoder head, single
270:   decoder); all expressiveness comes from the flexible prior.
271: 
272:   forward() returns the same 6-tuple as MoVEVAE so it is a drop-in replacement
273:   in epoch_vae.  The VampPrior KL scalar is stored in self._vamp_kl after each
274:   forward pass for use by the training loop.
275:   """
276: 
277:   def __init__(self,
278:                n_genes:      int,
279:                latent_dim:   int = 8,
280:                n_pseudo:     int = 20,
281:                hidden_dims:  list[int]|None = None,
282:                encoder_dims: list[int]|None = None,
283:                decoder_dims: list[int]|None = None,
284:                dropout:      float = 0.05,
285:                pseudo_init_samples: torch.Tensor|None = None):
286: 
287:     super().__init__(n_genes, latent_dim, hidden_dims, encoder_dims, decoder_dims, dropout)
288: 
289:     self.n_pseudo = n_pseudo
290: 
291:     # Encoder (single head — same as GeneExpressionVAE)
292:     self.encoder   = build_encoder(n_genes, self.encoder_dims, dropout)
293:     self.fc_mu     = nn.Linear(self.enc_out_dim, latent_dim)
294:     self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)
295: 
296:     # Decoder
297:     self.decoder = build_decoder(n_genes, latent_dim, self.decoder_dims, dropout)
298: 
299:     # Learnable pseudo-inputs in gene-expression space (K, n_genes).
300:     # A zero mask (all-observed) is appended before encoding so the
301:     # encoder sees a valid augmented input.
302:     if pseudo_init_samples is None:
303:       self.pseudo_inputs = nn.Parameter( torch.randn(n_pseudo, n_genes) * 0.01 )
304:     else:
305:       sample = torch.randint( 0, pseudo_init_samples.shape[0], size=(n_pseudo,) )
306:       self.pseudo_inputs = nn.Parameter( pseudo_init_samples[sample] )
307:     assert(self.pseudo_inputs.data.shape==(n_pseudo, n_genes))
308: 
309:     # Populated by forward(); read by epoch_vae.
310:     self._vamp_kl: torch.Tensor|None = None
311: 
312:   def _pseudo_encoded(self) -> tuple[torch.Tensor, torch.Tensor]:
313:     """Encode all K pseudo-inputs; return (pu_mu, pu_logvar) each (K, D)."""
314:     zero_mask  = torch.zeros(self.n_pseudo, self.n_genes, device=self.pseudo_inputs.device)
315:     pseudo_aug = torch.cat([self.pseudo_inputs, zero_mask], dim=1)  # (K, n_genes*2)
316:     h = self.encoder(pseudo_aug)
317:     return self.fc_mu(h), self.fc_logvar(h)  # (K, D), (K, D)
318: 
319:   def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
320:     h = self.encoder(x)
321:     return h, self.fc_mu(h), self.fc_logvar(h)
322: 
323:   def decode(self, z: torch.Tensor) -> torch.Tensor:
324:     return self.decoder(z)
325: 
326:   def vampprior_kl(self,
327:                    mu:     torch.Tensor,
328:                    logvar: torch.Tensor,
329:                    z:      torch.Tensor) -> torch.Tensor:
330:     """
331:     KL( q(z|x) || p_vamp(z) ) estimated with a single sample z ~ q(z|x).
332: 
333:     KL = E_q[log q(z|x)] - E_q[log p(z)]
334:        = E_q[log q(z|x)] - E_q[log (1/K) sum_k q(z|u_k)]
335: 
336:     mu, logvar, z : (B, D)
337:     returns scalar
338:     """
339:     # log q(z | x) : (B,)
340:     log_q = gaussian_log_prob(z, mu, logvar).sum(dim=-1)
341: 
342:     # Encode all pseudo-inputs once per forward pass
343:     pu_mu, pu_logvar = self._pseudo_encoded()  # (K, D)
344: 
345:     # Broadcast: z (B,1,D), pu_mu/pu_logvar (1,K,D)
346:     log_p_k = gaussian_log_prob(
347:         z.unsqueeze(1),           # (B, 1, D)
348:         pu_mu.unsqueeze(0),       # (1, K, D)
349:         pu_logvar.unsqueeze(0),   # (1, K, D)
350:     ).sum(dim=-1)                 # (B, K)
351: 
352:     # effective K (diagnostic)
353:     # range: 1.0 - n_pseudo
354:     with torch.no_grad():
355:       log_resp = log_p_k - torch.logsumexp( log_p_k, dim=1, keepdim=True ) # (B, K)
356:       resp = log_resp.exp()
357:       # effective K = exp(mean_batch[ entropy(resp)])
358:       ent = -(resp * (resp + 1e-20).log()).sum(dim=1)  # (B,)
359:       self._effective_K = torch.exp(ent.mean())
360: 
361:     log_p = torch.logsumexp(log_p_k, dim=1) - math.log(self.n_pseudo)  # (B,)
362: 
363:     return (log_q - log_p).mean()  # scalar
364: 
365:   def forward(self, x: torch.Tensor):
366:     """
367:     Returns: (recon, mu, logvar, None, None, None)
368:     Side-effect: stores VampPrior KL in self._vamp_kl after each call.
369:     """
370:     _, mu, logvar = self.encode(x)
371:     z = self.reparameterize(mu, logvar)
372:     recon = self.decode(z)
373:     self._vamp_kl = self.vampprior_kl(mu, logvar, z)
374:     return recon, mu, logvar, None, None, None
375: 
376: 
377: 
378: def epoch_vae(model, loader, opt=None, mask_fraction=0.1, beta=1.0, gamma=0.2, min_free_bits=0.05, lambda_entropy=0.1):
379:   total_loss, total_err, grad_norm, recon_loss, kl_loss = 0.,0.,0.,0.,0.
380: 
381:   model.eval() if opt is None else model.train()
382: 
383:   loss_func = masked_loss # nn.MSELoss()
384: 
385:   if opt:
386:     opt.zero_grad()
387: 
388:   #curr_step = 0
389: 
390:   recon_loss_type2 = torch.tensor(0.0)
391: 
392:   for x in loader:
393: 
394:     x_masked,  mask, type2_mask = get_random_mask(x, mask_fraction)
395:     x_masked = torch.cat( (x_masked, mask.float()), dim=1 )
396:     x, x_masked ,mask, type2_mask = x.to(device), x_masked.to(device), mask.to(device), type2_mask.to(device)
397: 
398:     ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
399:     #return x_masked, mask
400:     ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
401: 
402:     #assert( ~torch.isnan( x_masked ).any() )
403:     #assert( ~torch.isnan( mask ).any() )
404: 
405:     if opt:
406:       pred, mu, logvar, gate_probs, soft_weights, _ = model(x_masked)
407:     else:
408:       with torch.no_grad():
409:         pred, mu, logvar, gate_probs, soft_weights, _ = model(x_masked)
410: 
411:     #assert( ~torch.isnan( pred ).any() )
412:     #assert( ~torch.isnan( mu ).any() )
413:     #assert( ~torch.isnan( logvar ).any() )
414: 
415:     x.nan_to_num_(nan=0)
416: 
417:     kl_override = model._vamp_kl if isinstance(model, VampPriorVAE) else None
418: 
419:     loss, recon_loss, recon_loss_type2, kl_loss = loss_func(
420:         x,
421:         pred,
422:         mask,
423:         mu,
424:         logvar,
425:         gate_probs,
426:         soft_weights,
427:         beta=beta,
428:         gamma=gamma,
429:         mask_type2=type2_mask,
430:         free_bits=min_free_bits,
431:         lambda_entropy=lambda_entropy,
432:         kl_override=kl_override)
433: 
434: 
435:     if opt:
436:       loss.backward()
437: 
438:       # record gradient histogram
439:       #if (curr_step+5)==len(loader):
440:       #  _debug_grads = collect_grads(model, (z.shape[1],z.shape[1]))
441: 
442:       # update weights
443:       grad_norm += torch.nn.utils.clip_grad_norm_( model.parameters(), max_norm=1 ) # clip in-place
444: 
445:       opt.step()
446:       opt.zero_grad()
447: 
448:     total_loss += loss.item() #* accumulate_steps # undo accumulation scaling  # / z_train.shape[0]
449: 
450:     #curr_step += 1
451: 
452: 
453:   total_loss /= len(loader)
454:   grad_norm  /= len(loader)
455: 
456:   return total_loss, grad_norm, recon_loss.detach(), recon_loss_type2.detach(), kl_loss.detach()*beta
457: 
458: 
459: # ---------------------------------------------------------------------------
460: # SSM Beta Controller, World, Controller, and REINFORCE training loop
461: # ---------------------------------------------------------------------------
462: 
463: class RunningNorm:
464:     """
465:     Per-episode exponential moving-average normalizer.
466: 
467:     Tracks mean and variance of a D-dimensional signal using EMA.
468:     Call .update(x) each step, then .normalize(x) to z-score.
469:     Reset with .reset() at the start of each episode.
470:     """
471:     def __init__(self, dim: int, alpha: float = 0.1, eps: float = 1e-6):
472:         self.dim   = dim
473:         self.alpha = alpha
474:         self.eps   = eps
475:         self.reset()
476: 
477:     def reset(self):
478:         self.mean = torch.zeros(self.dim)
479:         self.var  = torch.ones(self.dim)
480:         self._initialized = False
481: 
482:     def update(self, x: torch.Tensor):
483:         """x: (dim,) — one observation vector (detached, on CPU)."""
484:         x = x.detach().cpu().float()
485:         if not self._initialized:
486:             self.mean = x.clone()
487:             self.var  = torch.ones(self.dim)
488:             self._initialized = True
489:         else:
490:             self.mean = (1 - self.alpha) * self.mean + self.alpha * x
491:             self.var  = (1 - self.alpha) * self.var  + self.alpha * (x - self.mean).pow(2)
492: 
493:     def normalize(self, x: torch.Tensor) -> torch.Tensor:
494:         """Returns z-scored x using current EMA statistics."""
495:         mean = self.mean.to(x.device)
496:         std  = self.var.clamp(min=self.eps).sqrt().to(x.device)
497:         return (x - mean) / std
498: 
499: 
500: class BetaSSMController(nn.Module):
501:     """
502:     Tiny diagonal linear SSM that outputs a beta distribution for VAE training.
503: 
504:     h_t = hidden state
505:     x_t = observations
506:     y_t = control output
507:     A   = hidden state evolution
508:     B   = observed variable effects
509:     C   = hidden state control contribution
510:     D   = observed variable control contribution
511: 
512:     State-space model (diagonal A for efficiency and stability):
513:         h_t = diag(A) * h_{t-1} + B_ctrl @ x_t
514:         y_t = C @ h_t + D @ x_t          → (mu_log_beta, log_sigma)
515: 
516:     A is parameterized via softplus to keep eigenvalues in (0, 1) (stable).
517: 
518:     Action: beta ~ LogNormal(mu_log_beta, exp(log_sigma)), clipped to [1e-5, 0.1].
519: 
520:     Two input paths:
521:       B_ctrl  (hidden_dim × obs_dim)     — used during control (act)
522:       B_world (hidden_dim × obs_dim+1)   — used during world-model pretraining;
523:                                            the extra column encodes log(beta_t),
524:                                            first obs_dim columns tied to B_ctrl
525:                                            at init (but trained independently).
526: 
527: 
528:     Args:
529:         obs_dim:    dimension of the observation vector (default 12)
530:         hidden_dim: SSM state dimension (default 32)
531:     """
532:     def __init__(self, obs_dim: int = 12, hidden_dim: int = 32):
533:         super().__init__()
534:         self.obs_dim    = obs_dim
535:         self.hidden_dim = hidden_dim
536: 
537:         # Diagonal A: A = exp(-softplus(a_raw)) ∈ (0, 1)
538:         self.a_raw   = nn.Parameter(torch.zeros(hidden_dim))
539: 
540:         # Control path input matrix (obs_dim → hidden_dim)
541:         self.B_ctrl  = nn.Parameter(torch.randn(hidden_dim, obs_dim) * 0.01)
542: 
543:         # World-model pretraining input matrix (obs_dim+1 → hidden_dim).
544:         # Initialised so first obs_dim columns match B_ctrl.
545:         B_world_init = torch.randn(hidden_dim, obs_dim + 1) * 0.01
546:         B_world_init[:, :obs_dim] = self.B_ctrl.data.clone()
547:         self.B_world = nn.Parameter(B_world_init)
548: 
549:         # Output projection (shared by both paths)
550:         self.C        = nn.Parameter(torch.randn(2, hidden_dim) * 0.01)
551:         self.D        = nn.Parameter(torch.zeros(2, obs_dim))
552: 
553:         # Output bias: mu_log_beta initialised to centre of log-range
554:         # log([1e-5, 0.1]) midpoint = (log(1e-5)+log(0.1))/2 ≈ -6.9
555:         # log_sigma initialised to 1.5 → sigma≈4.5, covering the full range
556:         self.out_bias = nn.Parameter(torch.tensor([-6.9, 1.5]))
557: 
558:     def _step(self, x_proj: torch.Tensor, x_obs: torch.Tensor,
559:               h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
560:         """Shared SSM core given a pre-projected input x_proj (hidden_dim,)."""
561:         A     = torch.exp(-F.softplus(self.a_raw))          # (hidden_dim,)
562:         h_new = A * h + x_proj                               # (hidden_dim,)
563:         y     = self.C @ h_new + self.D @ x_obs + self.out_bias  # (2,)
564:         mu_log_beta = y[0]
565:         log_sigma   = y[1].clamp(-4.0, 2.0)
566:         return mu_log_beta, log_sigma, h_new
567: 
568: 
569:     def forward(self, x: torch.Tensor, h: torch.Tensor
570:                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
571:         """
572:         Control path: single SSM step using B_ctrl.
573: 
574:         Args:
575:             x: observation vector (obs_dim,)
576:             h: hidden state       (hidden_dim,)
577:         Returns:
578:             mu_log_beta, log_sigma, h_new
579:         """
580:         x = x.view(-1)
581:         h = h.view(-1)
582:         return self._step(self.B_ctrl @ x, x, h)
583: 
584:     def forward_world(self, x: torch.Tensor, log_beta: torch.Tensor,
585:                       h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
586:         """
587:         World-model pretraining path: single SSM step using B_world.
588:         Input is augmented with log(beta_t) so the model can learn how
589:         beta affects future observations.
590: 
591:         Args:
592:             x:        observation vector (obs_dim,)
593:             log_beta: scalar log of the beta applied this step
594:             h:        hidden state (hidden_dim,)
595:         Returns:
596:             mu_log_beta, log_sigma, h_new
597:         """
598:         x        = x.view(-1)
599:         h        = h.view(-1)
600:         log_beta = log_beta.view(1)
601:         x_aug    = torch.cat([x, log_beta], dim=0)   # (obs_dim+1,)
602:         return self._step(self.B_world @ x_aug, x, h)
603: 
604:     def initial_state(self, device=None) -> torch.Tensor:
605:         return torch.zeros(self.hidden_dim, device=device)
606: 
607: # ---------------------------------------------------------------------------
608: # Beta source functions
609: # (t: int, obs: Tensor) -> float   — all share this signature
610: # ---------------------------------------------------------------------------
611: 
612: def beta_triangle(t: int, obs: torch.Tensor, *,
613:                   period: int = 31, lo: float = 1e-5, hi: float = 0.1) -> float:
614:     """Triangle wave in log-space between lo and hi."""
615:     phase = (t % period) / period          # [0, 1)
616:     frac  = 1.0 - 2.0 * abs(phase - 0.5)  # triangle: 0→1→0
617:     return lo * (hi / lo) ** frac
618: 
619: 
620: def beta_sine(t: int, obs: torch.Tensor, *,
621:               period: int = 12, lo: float = 1e-5, hi: float = 0.5) -> float:
622:     """Sine wave in log-space between lo and hi."""
623:     frac = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / period)
624:     return lo * (hi / lo) ** frac
625: 
626: 
627: def beta_impulse(t: int, obs: torch.Tensor, *,
628:                  impulse_epochs: tuple = (5, 12, 25, 32,50, 75),
629:                  lo: float = 1e-5, hi: float = 0.1) -> float:
630:     """Low beta everywhere except brief spikes at specified epochs."""
631:     return hi if t in impulse_epochs else lo
632: 
633: def beta_flip(t:int, obs:torch.Tensor, orig, lo: float = 1e-5, hi: float = 0.1) -> float:
634:   return orig(t, obs, hi=lo, lo=hi)
635: 
636: def beta_constant(t: int, obs: torch.Tensor, *,
637:                   value: float = 1e-4) -> float:
638:     """Fixed beta for every epoch."""
639:     return value
640: 
641: 
642: def beta_cosine_warmup(t: int, obs: torch.Tensor, *,
643:                        T: int = 30, lo: float = 1e-5, hi: float = 0.1) -> float:
644:     """Cosine annealing from lo to hi over T epochs (mirrors existing schedule)."""
645:     frac = 0.5 * (1.0 - math.cos(math.pi * min(t, T - 1) / T))
646:     return lo * (hi / lo) ** frac
647: 
648: 
649: def beta_from_controller(controller: 'Controller'):
650:     """
651:     Adapter: wraps a Controller so it can be used as a BetaSource.
652:     log_prob is discarded — only beta is returned.
653:     The caller is responsible for resetting controller state before the episode.
654:     """
655:     def _src(t: int, obs: torch.Tensor) -> float:
656:         beta, _ = controller.act(obs)
657:         return beta
658:     return _src
659: 
660: 
661: # Convenience list of all built-in waveform sources (excluding controller adapter)
662: BUILTIN_BETA_SOURCES = [beta_triangle, beta_sine, beta_impulse, lambda t,obs: beta_flip(t, obs, beta_impulse),
663:                          ]
664: 
665: 
666: 
667: # ---------------------------------------------------------------------------
668: # Helpers: data and VAE construction
669: # ---------------------------------------------------------------------------
670: 
671: def _build_obs_raw(metrics: dict, meta: dict) -> torch.Tensor:
672:     """
673:     Assemble the raw (un-normalized) 12-d observation vector.
674: 
675:     Dynamic features (9):
676:         recon_loss, kl_loss, recon_loss_type2, grad_norm, effective_K,
677:         delta_recon, delta_kl, delta_eff_K, log(epoch+1)
678: 
679:     Fixed meta-parameter features (3 — fixed normalization, not z-scored):
680:         log(n_genes)/log(1000), log(n_cells)/log(10000), missing_rate
681: 
682:     Returns a CPU float32 tensor of shape (12,).
683:     """
684:     recon      = float(metrics.get('recon_loss',       0.0))
685:     kl         = float(metrics.get('kl_loss',          0.0))
686:     recon2     = float(metrics.get('recon_loss_type2', 0.0))
687:     gnorm      = float(metrics.get('grad_norm',        0.0))
688:     eff_k      = float(metrics.get('effective_K',      1.0))
689:     prev_recon = float(metrics.get('prev_recon',       recon))
690:     prev_kl    = float(metrics.get('prev_kl',          kl))
691:     prev_eff_k = float(metrics.get('prev_eff_K',       eff_k))
692:     epoch      = float(metrics.get('epoch',            0))
693: 
694:     dyn = torch.tensor([
695:         recon, kl, recon2, gnorm, eff_k,
696:         recon - prev_recon,
697:         kl    - prev_kl,
698:         eff_k - prev_eff_k,
699:         math.log(epoch + 1),
700:     ], dtype=torch.float32)
701: 
702:     meta_feat = torch.tensor([
703:         math.log(meta['n_genes']) / math.log(1000.0),
704:         math.log(meta['n_cells']) / math.log(10000.0),
705:         meta['missing_rate'],
706:     ], dtype=torch.float32)
707: 
708:     return torch.cat([dyn, meta_feat], dim=0)   # (12,) on CPU
709: 
710: 
711: 
712: def _normalize_obs(raw_obs: torch.Tensor,
713:                    running_norm: RunningNorm,
714:                    device) -> torch.Tensor:
715:     """
716:     Update running stats with the 9 dynamic features of raw_obs,
717:     then return the fully normalized 12-d vector on `device`.
718:     Meta features (last 3) are already on a fixed scale and are passed through.
719:     """
720:     dyn = raw_obs[:9]
721:     running_norm.update(dyn)
722:     dyn_normed  = running_norm.normalize(dyn).to(device)
723:     meta_normed = raw_obs[9:].to(device)
724:     return torch.cat([dyn_normed, meta_normed], dim=0)
725: 
726: 
727: 
728: def _make_loaders(problem_config: dict, batch_size: int, device):
729:     """Build train / test / train-full DataLoaders for one episode."""
730:     n_cells  = problem_config['n_cells']
731:     n_blocks = problem_config['n_blocks']
732: 
733:     block_state = random_points_on_hypersphere(
734:         num_points=n_cells,
735:         K=n_blocks,
736:         N=min(24, n_blocks - 1),
737:         device=device,
738:     )
739:     scale = block_state.abs().max(dim=0).values.clamp(min=1e-6)
740:     block_state = block_state / scale * 4.0
741: 
742:     datasets = []
743:     for _ in range(2):
744:         data, _ = make_synthetic_data4(
745:             block_state=block_state,
746:             n_genes=problem_config['n_genes'],
747:             n_blocks=n_blocks,
748:             rho_within=problem_config['rho_within'],
749:             allow_negative=True,
750:             missing_rate=problem_config['missing_rate'],
751:             seed=None,
752:             device=device,
753:         )
754:         datasets.append(data)
755: 
756:     data_train, data_test = datasets
757:     return (
758:         DataLoader(data_train, batch_size=batch_size,  shuffle=True),
759:         DataLoader(data_test,  batch_size=n_cells,     shuffle=False),
760:         DataLoader(data_train, batch_size=n_cells,     shuffle=False),
761:     )
762: 
763: 
764: def _make_vae(problem_config: dict, loader_train_full,
765:               vae_training_config: dict, device):
766:     """Instantiate a fresh VampPriorVAE + AdamW optimizer."""
767:     for x in loader_train_full:
768:         x_init = x.nan_to_num(0).to(device)
769:         break
770:     vae = VampPriorVAE(
771:         n_genes=problem_config['n_genes'],
772:         n_pseudo=vae_training_config['n_pseudo'],
773:         dropout=0.02,
774:         encoder_dims=[512, 256, 128],
775:         decoder_dims=[128, 200, 200],
776:         latent_dim=64,
777:         pseudo_init_samples=x_init,
778:     ).to(device)
779:     opt = optim.AdamW(vae.parameters(), lr=vae_training_config['lr'])
780:     return vae, opt
781: 
782: # ---------------------------------------------------------------------------
783: # World
784: # ---------------------------------------------------------------------------
785: class World:
786:     """
787:     Encapsulates one VAE training problem and the controlled VAE model.
788: 
789:     Lifecycle per episode:
790:         obs = world.reset(problem_config)  # fresh data + VAE + ref epoch
791:         for t in range(max_epochs):
792:             beta = <some source>
793:             obs  = world.step(beta)        # train one epoch, return next obs
794:         reward, recon_term, eff_k_term = world.get_reward()
795: 
796:     world.obs_history: list of (raw_obs_tensor_CPU, beta_float) — one entry
797:                        per step, suitable for the replay buffer.
798:     """
799: 
800:     def __init__(self,
801:                  vae_training_config: dict,
802:                  vae_loss_config:     dict,
803:                  base_problem_config: dict,
804:                  beta_ref:  float = 1e-4,
805:                  lambda_K:  float = 0.1,
806:                  device            = None):
807:         """
808:         Args:
809:             vae_training_config: max_epochs, batch_size, mask_fraction, lr, n_pseudo
810:             vae_loss_config:     gamma_recon, min_free_bits, lambda_entropy
811:             base_problem_config: fixed keys shared across all episodes
812:                                  (e.g. n_blocks, rho_within); varied keys
813:                                  (n_genes, n_cells, missing_rate) are
814:                                  overridden per episode in reset().
815:             beta_ref:   beta used for the reference epoch (reward normalisation)
816:             lambda_K:   weight for effective_K term in reward
817:             device:     torch device
818:         """
819:         self.vtc              = vae_training_config
820:         self.vlc              = vae_loss_config
821:         self.base_problem_config = base_problem_config   # fixed across episodes
822:         self.beta_ref         = beta_ref
823:         self.lambda_K         = lambda_K
824:         self.device           = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
825:         self.max_epochs       = vae_training_config['max_epochs']
826: 
827:         # Set during reset()
828:         self.meta:         dict           = {}
829:         self.problem_config: dict         = {}
830:         self.vae                          = None
831:         self.vae_opt                      = None
832:         self.loader_train                 = None
833:         self.loader_test                  = None
834:         self.loader_train_full            = None
835:         self.ref_recon:    float          = 1.0
836:         self.running_norm: RunningNorm    = RunningNorm(dim=9)
837:         self.obs_history:  list           = []
838:         self._metrics:     dict           = {}
839:         self._last_eff_k:  float          = 1.0
840: 
841:     def reset(self, problem_config: dict) -> torch.Tensor:
842:         """
843:         Prepare a fresh episode:
844:           1. Build dataset from problem_config.
845:           2. Run reference epoch (1 epoch, beta_ref) → self.ref_recon.
846:           3. Instantiate a second fresh VAE for the controlled episode.
847:           4. Reset running normalizer and obs_history.
848: 
849:         Returns the initial (all-zeros) observation tensor.
850:         """
851:         self.problem_config = problem_config
852:         self.meta = dict(
853:             n_genes      = problem_config['n_genes'],
854:             n_cells      = problem_config['n_cells'],
855:             missing_rate = problem_config['missing_rate'],
856:         )
857: 
858:         # Build loaders
859:         self.loader_train, self.loader_test, self.loader_train_full = _make_loaders(
860:             problem_config, self.vtc['batch_size'], self.device)
861: 
862:         # Reference epoch
863:         vae_ref, opt_ref = _make_vae(problem_config, self.loader_train_full,
864:                                      self.vtc, self.device)
865:         epoch_vae(vae_ref, self.loader_train, opt_ref,
866:                   mask_fraction=self.vtc['mask_fraction'], beta=self.beta_ref,
867:                   gamma=self.vlc['gamma_recon'], min_free_bits=self.vlc['min_free_bits'],
868:                   lambda_entropy=self.vlc['lambda_entropy'])
869:         _, _, ref_test_recon, _, _ = epoch_vae(
870:             vae_ref, self.loader_test, opt=None,
871:             mask_fraction=self.vtc['mask_fraction'], beta=0.0,
872:             gamma=self.vlc['gamma_recon'], min_free_bits=self.vlc['min_free_bits'],
873:             lambda_entropy=self.vlc['lambda_entropy'])
874:         self.ref_recon = float(ref_test_recon)
875:         del vae_ref, opt_ref
876: 
877:         # Fresh controlled VAE
878:         self.vae, self.vae_opt = _make_vae(problem_config, self.loader_train_full,
879:                                            self.vtc, self.device)
880: 
881:         # Reset per-episode state
882:         self.running_norm.reset()
883:         self.obs_history = []
884:         self._metrics    = dict(recon_loss=0., kl_loss=0., recon_loss_type2=0.,
885:                                 grad_norm=0., effective_K=1.,
886:                                 prev_recon=0., prev_kl=0., prev_eff_K=1., epoch=0)
887:         self._last_eff_k = 1.0
888: 
889:         print(f'  ref_recon_test={self.ref_recon:.4f}  '
890:               f'n_genes={problem_config["n_genes"]}  '
891:               f'n_cells={problem_config["n_cells"]}  '
892:               f'missing_rate={problem_config["missing_rate"]:.3f}')
893: 
894:         # Return initial (zero) observation
895:         raw_obs = _build_obs_raw(self._metrics, self.meta)
896:         return _normalize_obs(raw_obs, self.running_norm, self.device)
897: 
898:     def step(self, beta: float) -> torch.Tensor:
899:         """
900:         Train the VAE for one epoch with the given beta.
901:         Appends (raw_obs_before_step, beta) to obs_history.
902:         Returns the normalized observation AFTER this epoch.
903:         """
904:         t = int(self._metrics['epoch'])
905: 
906:         # Build raw obs BEFORE the step (what the controller acted on)
907:         raw_obs_pre = _build_obs_raw(self._metrics, self.meta)
908:         self.obs_history.append((raw_obs_pre.clone(), beta))
909: 
910:         # Train one epoch
911:         _, grad_norm, recon_l, recon2_l, kl_l = epoch_vae(
912:             self.vae, self.loader_train, self.vae_opt,
913:             mask_fraction=self.vtc['mask_fraction'],
914:             beta=beta,
915:             gamma=self.vlc['gamma_recon'],
916:             min_free_bits=self.vlc['min_free_bits'],
917:             lambda_entropy=self.vlc['lambda_entropy'],
918:         )
919:         eff_k = float(getattr(self.vae, '_effective_K', torch.tensor(1.0)))
920:         self._last_eff_k = eff_k
921: 
922:         # Update metrics for the next observation
923:         self._metrics = dict(
924:             recon_loss       = float(recon_l),
925:             kl_loss          = float(kl_l),
926:             recon_loss_type2 = float(recon2_l),
927:             grad_norm        = float(grad_norm),
928:             effective_K      = eff_k,
929:             prev_recon       = self._metrics['recon_loss'],
930:             prev_kl          = self._metrics['kl_loss'],
931:             prev_eff_K       = self._metrics['effective_K'],
932:             epoch            = t + 1,
933:         )
934: 
935:         raw_obs_post = _build_obs_raw(self._metrics, self.meta)
936:         return _normalize_obs(raw_obs_post, self.running_norm, self.device)
937: 
938:     def get_reward(self) -> tuple[float, float, float]:
939:         """
940:         Evaluate test-set reconstruction (beta=0) on the current VAE.
941:         Returns (reward, recon_term, eff_k_term).
942:         """
943:         _, _, final_recon_t, _, _ = epoch_vae(
944:             self.vae, self.loader_test, opt=None,
945:             mask_fraction=self.vtc['mask_fraction'], beta=0.0,
946:             gamma=self.vlc['gamma_recon'], min_free_bits=self.vlc['min_free_bits'],
947:             lambda_entropy=self.vlc['lambda_entropy'])
948:         final_recon = float(final_recon_t)
949: 
950:         n_pseudo   = self.vtc['n_pseudo']
951:         recon_term = (self.ref_recon - final_recon) / (self.ref_recon + 1e-8)
952:         eff_k_term = math.log(max(self._last_eff_k, 1.0)) / math.log(n_pseudo)
953:         reward     = recon_term + self.lambda_K * eff_k_term
954:         return reward, recon_term, eff_k_term
955: 
956: # ---------------------------------------------------------------------------
957: # Controller
958: # ---------------------------------------------------------------------------
959: 
960: class Controller:
961:     """
962:     Wraps BetaSSMController with:
963:       - REINFORCE policy update
964:       - World-model pretraining from a replay buffer
965:       - EMA reward baseline
966:     """
967: 
968:     def __init__(self,
969:                  ssm:            BetaSSMController,
970:                  lr:             float = 1e-3,
971:                  baseline_alpha: float = 0.1,
972:                  device                = None):
973:         self.ssm            = ssm
974:         self.device         = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
975:         self.baseline_alpha = baseline_alpha
976:         self.baseline       = 0.0
977:         self._opt           = optim.Adam(ssm.parameters(), lr=lr)
978:         self.h              = ssm.initial_state(device=self.device)
979: 
980:     def reset_state(self):
981:         """Reset SSM hidden state to zeros (call at the start of each episode)."""
982:         self.h = self.ssm.initial_state(device=self.device)
983: 
984:     def act(self, obs: torch.Tensor) -> tuple[float, torch.Tensor]:
985:         """
986:         Step the SSM with obs, sample beta ~ LogNormal(mu, sigma).
987: 
988:         Args:
989:             obs: normalized observation (12,) on device
990: 
991:         Returns:
992:             beta_float: sampled beta, clipped to [1e-5, 0.1]
993:             log_prob:   log-probability of the sample (scalar tensor, has grad)
994:         """
995:         mu_log_beta, log_sigma, self.h = self.ssm(obs, self.h)
996:         dist           = torch.distributions.Normal(mu_log_beta, log_sigma.exp())
997:         log_beta_samp  = dist.rsample()
998:         log_prob       = dist.log_prob(log_beta_samp)
999:         beta           = log_beta_samp.exp().clamp(1e-5, 0.1).item()
1000:         return beta, log_prob
1001: 
1002:     def update(self, log_probs: list, reward: float):
1003:         """
1004:         REINFORCE gradient step.
1005: 
1006:         Args:
1007:             log_probs: list of scalar log-prob tensors (one per epoch)
1008:             reward:    scalar episode reward
1009:         """
1010:         advantage   = reward - self.baseline
1011:         policy_loss = -advantage * torch.stack(log_probs).sum()
1012: 
1013:         self._opt.zero_grad()
1014:         policy_loss.backward()
1015:         torch.nn.utils.clip_grad_norm_(self.ssm.parameters(), max_norm=1.0)
1016:         self._opt.step()
1017: 
1018:         self.baseline = ((1 - self.baseline_alpha) * self.baseline
1019:                          + self.baseline_alpha * reward)
1020: 
1021:     def pretrain(self,
1022:                  replay_buffer: list,
1023:                  n_steps:       int   = 500,
1024:                  lr_pretrain:   float = 1e-3,
1025:                  seq_len:       int   = 50):
1026:         """
1027:         World-model pretraining via next-observation prediction.
1028: 
1029:         For each gradient step:
1030:           1. Sample a random episode from the replay buffer.
1031:           2. Sample a random start within it (subsequence of length seq_len).
1032:           3. Re-run the SSM forward pass using forward_world(obs_t, log_beta_t).
1033:           4. Predict obs_{t+1} with a linear head.
1034:           5. Loss = MSE(prediction, actual_obs_{t+1}).
1035: 
1036:         A separate AdamW optimizer is used so pretraining LR is independent
1037:         of the REINFORCE LR.  The prediction head (self._pred_head) is created
1038:         lazily on first call.
1039: 
1040:         Args:
1041:             replay_buffer: list of dicts with keys 'obs_sequence', 'meta'
1042:             n_steps:       number of gradient steps
1043:             lr_pretrain:   learning rate for the prediction head + SSM params
1044:             seq_len:       BPTT truncation length
1045:         """
1046:         if len(replay_buffer) == 0:
1047:             return
1048: 
1049:         obs_dim    = self.ssm.obs_dim
1050:         hidden_dim = self.ssm.hidden_dim
1051: 
1052:         # Lazy-init prediction head (hidden_dim → obs_dim) and its optimizer
1053:         if not hasattr(self, '_pred_head'):
1054:             self._pred_head = nn.Linear(hidden_dim, obs_dim).to(self.device)
1055:             self._pretrain_opt = optim.AdamW(
1056:                 list(self.ssm.parameters()) + list(self._pred_head.parameters()),
1057:                 lr=lr_pretrain)
1058: 
1059:         rng = np.random.default_rng()
1060: 
1061:         pretrain_log = []
1062: 
1063:         for _ in tqdm(range(n_steps)):
1064:             # Sample episode
1065:             ep   = replay_buffer[rng.integers(len(replay_buffer))]
1066:             seq  = ep['obs_sequence']    # list of (raw_obs_CPU, beta_float)
1067:             meta = ep['meta']
1068:             T    = len(seq)
1069:             if T < 2:
1070:                 continue
1071: 
1072:             # Sample start position; clip so we have at least 2 steps
1073:             start = int(rng.integers(0, max(1, T - 1)))
1074:             end   = min(start + seq_len, T - 1)   # we need obs[end] as target
1075: 
1076:             # Re-run RunningNorm from scratch up to `start` to get correct stats
1077:             rn = RunningNorm(dim=9)
1078:             for i in range(start):
1079:                 rn.update(seq[i][0][:9])
1080: 
1081:             # Forward pass over the subsequence
1082:             h    = self.ssm.initial_state(device=self.device)
1083:             loss = torch.tensor(0.0, device=self.device)
1084:             n    = 0
1085: 
1086:             for i in range(start, end):
1087:                 raw_obs_t, beta_t = seq[i]
1088:                 raw_obs_next, _   = seq[i + 1]
1089: 
1090:                 obs_t    = _normalize_obs(raw_obs_t,    rn, self.device)
1091:                 obs_next = _normalize_obs(raw_obs_next, rn, self.device)
1092: 
1093:                 log_beta = torch.tensor(math.log(max(beta_t, 1e-10)),
1094:                                         dtype=torch.float32, device=self.device)
1095:                 _, _, h = self.ssm.forward_world(obs_t, log_beta, h)
1096:                 pred    = self._pred_head(h)
1097: 
1098:                 loss = loss + F.mse_loss(pred, obs_next.detach())
1099:                 n   += 1
1100: 
1101:             if n == 0:
1102:                 continue
1103: 
1104:             self._pretrain_opt.zero_grad()
1105:             (loss / n).backward()
1106:             torch.nn.utils.clip_grad_norm_(
1107:                 list(self.ssm.parameters()) + list(self._pred_head.parameters()),
1108:                 max_norm=1.0)
1109:             self._pretrain_opt.step()
1110: 
1111:             pretrain_log.append( float(loss/n) )
1112:             #print(f'pretrain loss: {float(loss/n):.3g}')
1113: 
1114:         ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
1115:         plt.subplots()
1116:         plt.plot(pretrain_log)
1117:         plt.gca().set_yscale('log')
1118:         plt.show()
1119:         ## DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY ##
1120: 
1121:         return pretrain_log
1122: 
1123: 
1124: # ---------------------------------------------------------------------------
1125: # run_pretrain_episode
1126: # ---------------------------------------------------------------------------
1127: 
1128: def run_pretrain_episode(world: World, beta_source, problem_config: dict) -> dict:
1129:     """
1130:     Run one full episode using a fixed beta_source (not the controller).
1131:     Populates world.obs_history and returns a replay-buffer entry.
1132:     No gradient updates are performed.
1133: 
1134:     Args:
1135:         world:          World instance
1136:         beta_source:    callable (t, obs) -> float
1137:         problem_config: dict passed to world.reset()
1138: 
1139:     Returns:
1140:         dict with keys 'meta', 'obs_sequence', 'final_obs'
1141:     """
1142:     obs = world.reset(problem_config)
1143:     for t in range(world.max_epochs):
1144:         beta = beta_source(t, obs)
1145:         obs  = world.step(beta)
1146:     return {
1147:         'meta':         world.meta.copy(),
1148:         'obs_sequence': list(world.obs_history),   # list of (raw_obs_CPU, beta)
1149:         'final_obs':    obs.detach().cpu(),
1150:     }
1151: 
1152: 
1153: # ---------------------------------------------------------------------------
1154: # rl_training_loop  (thin orchestrator)
1155: # ---------------------------------------------------------------------------
1156: 
1157: def rl_training_loop(
1158:     world:               World,
1159:     controller:          Controller,
1160:     n_episodes:          int   = 30,
1161:     # Problem meta-parameter ranges (sampled per episode)
1162:     n_genes_choices:     list  = None,
1163:     n_cells_choices:     list  = None,
1164:     missing_rate_range:  tuple = (0.02, 0.15),
1165:     # Optional pretraining
1166:     pretrain_sources:    list  = None,   # list of beta_source callables
1167:     n_pretrain_episodes: int   = 0,      # replay episodes to collect before RL
1168:     pretrain_steps:      int   = 0,      # world-model gradient steps per RL episode
1169:     pretrain_lr:         float = 1e-3,
1170:     pretrain_seq_len:    int   = 20,
1171:     replay_buffer:       list  = None,
1172:     seed:                int   = 0,
1173: ) -> tuple:
1174:     """
1175:     Outer REINFORCE loop.
1176: 
1177:     Phase 0 (optional): collect `n_pretrain_episodes` episodes using random
1178:         beta sources from `pretrain_sources`, then pretrain the controller's
1179:         world model for `pretrain_steps` gradient steps.
1180: 
1181:     Phase 1: for each RL episode —
1182:         1. Sample problem meta-parameters.
1183:         2. world.reset(problem_config)
1184:         3. controller.reset_state()
1185:         4. Loop max_epochs: beta, log_prob = controller.act(obs); obs = world.step(beta)
1186:         5. reward = world.get_reward()
1187:         6. controller.update(log_probs, reward)
1188:         7. Append episode to replay_buffer.
1189:         8. Optional: controller.pretrain(replay_buffer, pretrain_steps)
1190: 
1191:     Returns:
1192:         controller, replay_buffer, episode_log
1193:     """
1194:     if n_genes_choices  is None: n_genes_choices  = [100, 150, 200]
1195:     if n_cells_choices  is None: n_cells_choices  = [1000, 1500, 2000]
1196:     if pretrain_sources is None: pretrain_sources = BUILTIN_BETA_SOURCES
1197:     if replay_buffer    is None: replay_buffer    = []
1198: 
1199:     rng = np.random.default_rng(seed)
1200: 
1201:     def _sample_problem():
1202:         return {
1203:             **world.base_problem_config,   # inherit fixed keys (n_blocks, rho_within, …)
1204:             'n_genes':      int(rng.choice(n_genes_choices)),
1205:             'n_cells':      int(rng.choice(n_cells_choices)),
1206:             'missing_rate': float(rng.uniform(*missing_rate_range)),
1207:         }
1208: 
1209:     # ------------------------------------------------------------------ #
1210:     # Phase 0: pre-populate replay buffer with diverse beta sources
1211:     # ------------------------------------------------------------------ #
1212:     if n_pretrain_episodes > 0:
1213:         print(f'\n--- Pretraining data collection: {n_pretrain_episodes} episodes ---')
1214:         for i in range(n_pretrain_episodes):
1215:             src    = pretrain_sources[i % len(pretrain_sources)]
1216:             pcfg   = _sample_problem()
1217:             entry  = run_pretrain_episode(world, src, pcfg)
1218:             replay_buffer.append(entry)
1219:             print(f'  pretrain ep {i+1}/{n_pretrain_episodes}  '
1220:                   f'src={src.__name__}  n_genes={pcfg["n_genes"]}  '
1221:                   f'missing_rate={pcfg["missing_rate"]:.3f}')
1222: 
1223:         if pretrain_steps > 0:
1224:             print(f'  world-model pretraining: {pretrain_steps} steps ...')
1225:             pretrain_log = controller.pretrain(replay_buffer, n_steps=pretrain_steps,
1226:                                                lr_pretrain=pretrain_lr, seq_len=pretrain_seq_len)
1227: 
1228:     # ------------------------------------------------------------------ #
1229:     # Phase 1: RL episodes
1230:     # ------------------------------------------------------------------ #
1231:     episode_log = []
1232: 
1233:     for ep in range(n_episodes):
1234:         pcfg = _sample_problem()
1235:         print(f'\n=== RL episode {ep+1}/{n_episodes}  '
1236:               f'n_genes={pcfg["n_genes"]}  n_cells={pcfg["n_cells"]}  '
1237:               f'missing_rate={pcfg["missing_rate"]:.3f} ===')
1238: 
1239:         obs = world.reset(pcfg)
1240:         controller.reset_state()
1241: 
1242:         log_probs   = []
1243:         epoch_betas = []
1244: 
1245:         for t in range(world.max_epochs):
1246:             beta, log_prob = controller.act(obs)
1247:             obs = world.step(beta)
1248:             log_probs.append(log_prob)
1249:             epoch_betas.append(beta)
1250: 
1251:             if (t + 1) % 20 == 0:
1252:                 m = world._metrics
1253:                 print(f'  epoch {t+1:3d}  beta={beta:.2e}  '
1254:                       f'recon={m["recon_loss"]:.4f}  kl={m["kl_loss"]:.4f}  '
1255:                       f'eff_K={m["effective_K"]:.1f}')
1256: 
1257:         reward, recon_term, eff_k_term = world.get_reward()
1258:         print(f'  reward={reward:.4f}  recon_term={recon_term:.4f}  '
1259:               f'eff_k_term={eff_k_term:.4f}  baseline={controller.baseline:.4f}')
1260: 
1261:         controller.update(log_probs, reward)
1262: 
1263:         # Add to replay buffer
1264:         replay_buffer.append({
1265:             'meta':         world.meta.copy(),
1266:             'obs_sequence': list(world.obs_history),
1267:             'final_obs':    obs.detach().cpu(),
1268:         })
1269: 
1270:         # Optional interleaved world-model pretraining
1271:         if pretrain_steps > 0:
1272:             controller.pretrain(replay_buffer, n_steps=pretrain_steps,
1273:                                 lr_pretrain=pretrain_lr, seq_len=pretrain_seq_len)
1274: 
1275:         episode_log.append(dict(
1276:             episode      = ep,
1277:             n_genes      = pcfg['n_genes'],
1278:             n_cells      = pcfg['n_cells'],
1279:             missing_rate = pcfg['missing_rate'],
1280:             ref_recon    = world.ref_recon,
1281:             final_eff_k  = world._last_eff_k,
1282:             recon_term   = recon_term,
1283:             eff_k_term   = eff_k_term,
1284:             reward       = reward,
1285:             baseline     = controller.baseline,
1286:             betas        = epoch_betas,
1287:         ))
1288: 
1289:     return controller, replay_buffer, episode_log
1290: 
1291: 
1292: def plot_rl_results(episode_log: list):
1293:     """Plot reward components and beta trajectories from rl_training_loop."""
1294:     episodes    = [e['episode']    for e in episode_log]
1295:     rewards     = [e['reward']     for e in episode_log]
1296:     baselines   = [e['baseline']   for e in episode_log]
1297:     recon_terms = [e['recon_term'] for e in episode_log]
1298:     eff_k_terms = [e['eff_k_term'] for e in episode_log]
1299:     final_eff_ks= [e['final_eff_k'] for e in episode_log]
1300: 
1301:     fig, axes = plt.subplots(1, 3, figsize=(18, 4))
1302: 
1303:     axes[0].plot(episodes, rewards,     label='total reward', marker='o', ms=4)
1304:     axes[0].plot(episodes, baselines,   label='baseline',     linestyle='--')
1305:     axes[0].plot(episodes, recon_terms, label='recon term',   linestyle=':')
1306:     axes[0].plot(episodes, eff_k_terms, label='eff_K term',   linestyle='-.')
1307:     axes[0].axhline(0, color='k', lw=0.5)
1308:     axes[0].set_xlabel('Episode')
1309:     axes[0].set_ylabel('Reward')
1310:     axes[0].set_title('RL controller: reward components')
1311:     axes[0].legend(fontsize=8)
1312: 
1313:     axes[1].plot(episodes, final_eff_ks, marker='o', ms=4, color='tab:orange')
1314:     axes[1].set_xlabel('Episode')
1315:     axes[1].set_ylabel('effective_K (final epoch)')
1316:     axes[1].set_title('VampPrior latent utilization')
1317: 
1318:     for e in episode_log[-5:]:
1319:         axes[2].plot(e['betas'], alpha=0.8,
1320:                      label=f"ep{e['episode']} g{e['n_genes']} m{e['missing_rate']:.2f}")
1321:     axes[2].set_yscale('log')
1322:     axes[2].set_xlabel('Epoch')
1323:     axes[2].set_ylabel('beta (log scale)')
1324:     axes[2].set_title('Beta trajectories (last 5 episodes)')
1325:     axes[2].legend(fontsize=7)
1326: 
1327:     plt.tight_layout()
1328:     plt.savefig('rl_controller_results.png', dpi=120)
1329:     plt.show()
1330: 

(Output capped at 50 KB. Showing lines 1-1330. Use offset=1331 to continue.)
</content>
```

---

## Assistant (Plan · Claude Sonnet 4.6 · 3.1s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/michael/src/vae-test/models.py",
  "offset": 1330
}
```

**Output:**
```
<path>/home/michael/src/vae-test/models.py</path>
<type>file</type>
<content>
1330: 
1331: class MeansModel(nn.Module):
1332:   """
1333:   Model to impute genes based on the average value for each gene (baseline model for comparison)
1334:   """
1335:   def __init__(self,
1336:                n_genes:int):
1337:     super().__init__()
1338:     self.means = nn.Parameter(torch.zeros(n_genes))
1339: 
1340:   def forward(self, x:torch.Tensor):
1341:     if x.ndim==1:
1342:       output = self.means  # (n_genes,)
1343:     else:
1344:       output = self.means.expand( x.shape[0], -1 )  # (batch_size, n_genes)
1345: 
1346:     return output, torch.ones(1), torch.zeros(1), None, None, None
1347: 
1348: 
1349: 
1350: # ===========================================================================
1351: # ImputationTransformer: decoder-focused transformer for gene imputation
1352: # ===========================================================================
1353: #
1354: # Architecture summary:
1355: #   - Each gene is a token; embedding = concat(gene_id_emb, count_bin_emb, conf_bin_emb)
1356: #   - Non-causal attention with a confidence-gated key mask: low-confidence
1357: #     (unknown/imputed) tokens are excluded as keys so they cannot be attended to,
1358: #     but they can still attend to all high-confidence (known) tokens.
1359: #   - Two output heads per token: count bin logits + confidence bin logits.
1360: #   - Confidence head is supervised by the magnitude of the count prediction error
1361: #     (self-supervised, no external labels needed).
1362: #   - Iterative inference: imputed values are re-entered with entropy-derived
1363: #     confidence and the model is run again until convergence.
1364: #
1365: # Training (single-pass, v1):
1366: #   - Type 1 mask (NaN values): set to unknown count token + conf bin 0.
1367: #   - Type 2 mask (random subset of observed): same treatment; loss computed here.
1368: #   - Count loss: cross-entropy on Type-2 masked positions only.
1369: #   - Conf  loss: cross-entropy on Type-2 positions, target = binned |count_error|.
1370: # ===========================================================================
1371: 
1372: 
1373: # ---------------------------------------------------------------------------
1374: # 1. Discretization utilities
1375: # ---------------------------------------------------------------------------
1376: 
1377: def make_log_bin_edges(n_bins: int, max_val: float = 8.5) -> torch.Tensor:
1378:     """
1379:     Produce n_bins+1 edges in log1p space covering [0, max_val].
1380: 
1381:     log1p(5000) ~ 8.52, so max_val=8.5 covers essentially all scRNA-seq counts.
1382:     The edges are linearly spaced in log1p space, which gives log-spacing in
1383:     raw count space (finer resolution at low counts).
1384: 
1385:     Returns: 1-D tensor of shape (n_bins+1,) on CPU.
1386:     """
1387:     return torch.linspace(0.0, max_val, n_bins + 1)
1388: 
1389: 
1390: def discretize(x: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
1391:     """
1392:     Map continuous log1p expression values to integer bin indices.
1393: 
1394:     NaN entries are assigned the special "unknown" index = len(bin_edges) - 1 = n_bins.
1395:     Values below the lowest edge -> bin 0; above the highest -> bin n_bins-1.
1396: 
1397:     Args:
1398:         x:         (B, G) float tensor, possibly containing NaN.
1399:         bin_edges: (n_bins+1,) tensor of bin boundaries (from make_log_bin_edges).
1400: 
1401:     Returns:
1402:         bins: (B, G) long tensor; values in {0, ..., n_bins-1} for observed,
1403:               n_bins for NaN (unknown token index).
1404:     """
1405:     n_bins    = len(bin_edges) - 1
1406:     nan_mask  = torch.isnan(x)
1407: 
1408:     # torch.bucketize assigns index i when bin_edges[i-1] <= x < bin_edges[i].
1409:     # right=False: left-closed intervals; clamp to [0, n_bins-1].
1410:     x_safe    = x.nan_to_num(0.0)
1411:     edges_dev = bin_edges.to(x_safe.device)
1412:     bins      = torch.bucketize(x_safe, edges_dev, right=False).clamp(0, n_bins - 1)
1413:     bins[nan_mask] = n_bins  # unknown token
1414:     return bins
1415: 
1416: 
1417: def bins_to_midpoints(bins: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
1418:     """
1419:     Dequantize bin indices to the midpoint of each bin (for correlation evaluation).
1420: 
1421:     Bins equal to n_bins (unknown token) are mapped to 0.0.
1422: 
1423:     Args:
1424:         bins:      (B, G) long tensor of bin indices.
1425:         bin_edges: (n_bins+1,) tensor of bin boundaries.
1426: 
1427:     Returns:
1428:         (B, G) float tensor of dequantized values.
1429:     """
1430:     n_bins    = len(bin_edges) - 1
1431:     midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # (n_bins,)
1432:     # Clamp unknowns to 0 before indexing
1433:     safe_bins = bins.clamp(0, n_bins - 1)
1434:     vals      = midpoints.to(bins.device)[safe_bins]
1435:     vals[bins == n_bins] = 0.0  # unknown positions -> 0
1436:     return vals
1437: 
1438: 
1439: def entropy_to_conf_bin(logits: torch.Tensor, n_conf_bins: int) -> torch.Tensor:
1440:     """
1441:     Convert count-bin logits to a confidence bin index via normalized entropy.
1442: 
1443:     Low entropy (peaked distribution)  -> high confidence bin.
1444:     High entropy (flat distribution)   -> low confidence bin.
1445: 
1446:     Args:
1447:         logits:      (..., n_bins) float tensor (raw, un-softmaxed).
1448:         n_conf_bins: number of confidence bins.
1449: 
1450:     Returns:
1451:         (...,) long tensor of confidence bin indices in {0, ..., n_conf_bins-1}.
1452:     """
1453:     n_bins   = logits.shape[-1]
1454:     probs    = torch.softmax(logits, dim=-1).clamp(min=1e-9)
1455:     entropy  = -(probs * probs.log()).sum(dim=-1)       # (...,)
1456:     max_ent  = math.log(n_bins)                         # entropy of uniform
1457:     norm_ent = (entropy / max_ent).clamp(0.0, 1.0)     # 0=peaked, 1=flat
1458:     # High confidence = low entropy -> high bin index
1459:     conf_bin = ((1.0 - norm_ent) * (n_conf_bins - 1)).long().clamp(0, n_conf_bins - 1)
1460:     return conf_bin
1461: 
1462: 
1463: def error_to_conf_bin(pred_bins: torch.Tensor,
1464:                       true_bins: torch.Tensor,
1465:                       n_conf_bins: int,
1466:                       max_err_bins: int | None = None) -> torch.Tensor:
1467:     """
1468:     Map absolute count-bin prediction error to a confidence bin index.
1469: 
1470:     Small error -> high confidence bin; large error -> low confidence bin.
1471:     Used to supervise the confidence output head.
1472: 
1473:     Args:
1474:         pred_bins:    (...,) long tensor of predicted count bin indices.
1475:         true_bins:    (...,) long tensor of true count bin indices.
1476:         n_conf_bins:  number of confidence bins.
1477:         max_err_bins: clip errors beyond this many bins (defaults to 4).
1478: 
1479:     Returns:
1480:         (...,) long tensor of target confidence bin indices.
1481:     """
1482:     abs_err = (pred_bins.long() - true_bins.long()).abs().float()
1483:     if max_err_bins is None:
1484:         # Infer a reasonable cap: 4 bins is already a large error for 24 bins
1485:         max_err_bins = 4
1486:     norm_err = (abs_err / max_err_bins).clamp(0.0, 1.0)   # 0=perfect, 1=max_err
1487:     conf_bin = ((1.0 - norm_err) * (n_conf_bins - 1)).long().clamp(0, n_conf_bins - 1)
1488:     return conf_bin
1489: 
1490: 
1491: # ---------------------------------------------------------------------------
1492: # 2. GeneTokenEmbedding
1493: # ---------------------------------------------------------------------------
1494: 
1495: class GeneTokenEmbedding(nn.Module):
1496:     """
1497:     Per-gene token embedding combining three information sources:
1498:       - gene identity  (which gene)
1499:       - count bin      (expression level, discretized)
1500:       - confidence bin (measurement reliability)
1501: 
1502:     All three embeddings are concatenated and projected to d_model.
1503:     """
1504:     def __init__(self,
1505:                  n_genes:     int,
1506:                  n_bins:      int,
1507:                  n_conf_bins: int,
1508:                  d_gene:      int = 64,
1509:                  d_count:     int = 64,
1510:                  d_conf:      int = 32,
1511:                  d_model:     int = 256):
1512:         super().__init__()
1513:         # n_bins+1 count tokens: indices 0..n_bins-1 are observed bins,
1514:         # index n_bins is the special "unknown" token.
1515:         self.emb_gene  = nn.Embedding(n_genes,      d_gene)
1516:         self.emb_count = nn.Embedding(n_bins + 1,   d_count)
1517:         self.emb_conf  = nn.Embedding(n_conf_bins,  d_conf)
1518:         self.proj      = nn.Linear(d_gene + d_count + d_conf, d_model)
1519:         self.norm      = nn.LayerNorm(d_model)
1520: 
1521:     def forward(self,
1522:                 gene_ids:   torch.Tensor,
1523:                 count_bins: torch.Tensor,
1524:                 conf_bins:  torch.Tensor) -> torch.Tensor:
1525:         """
1526:         Args:
1527:             gene_ids:   (B, G) long -- gene index in {0, ..., n_genes-1}
1528:             count_bins: (B, G) long -- count bin in {0, ..., n_bins}
1529:             conf_bins:  (B, G) long -- confidence bin in {0, ..., n_conf_bins-1}
1530:         Returns:
1531:             (B, G, d_model) float
1532:         """
1533:         e = torch.cat([
1534:             self.emb_gene(gene_ids),
1535:             self.emb_count(count_bins),
1536:             self.emb_conf(conf_bins),
1537:         ], dim=-1)
1538:         return self.norm(self.proj(e))
1539: 
1540: 
1541: # ---------------------------------------------------------------------------
1542: # 3. ImputationTransformer
1543: # ---------------------------------------------------------------------------
1544: 
1545: class ImputationTransformer(nn.Module):
1546:     """
1547:     Decoder-focused transformer for gene expression imputation.
1548: 
1549:     Tokens are genes; the attention mask is confidence-gated:
1550:       - Low-confidence tokens (conf_bin == 0) are masked out as keys,
1551:         so they cannot be attended to by any other token.
1552:       - All tokens (including low-confidence) can attend to high-confidence keys.
1553: 
1554:     This means imputed/unknown genes are invisible to the attention of other
1555:     genes but can themselves gather information from all known genes.
1556: 
1557:     Output:
1558:       - count_logits: (B, G, n_bins)      -- predicted count bin distribution
1559:       - conf_logits:  (B, G, n_conf_bins) -- predicted confidence bin distribution
1560:     """
1561: 
1562:     def __init__(self,
1563:                  n_genes:     int,
1564:                  n_bins:      int   = 24,
1565:                  n_conf_bins: int   = 8,
1566:                  d_gene:      int   = 64,
1567:                  d_count:     int   = 64,
1568:                  d_conf:      int   = 32,
1569:                  d_model:     int   = 256,
1570:                  n_heads:     int   = 8,
1571:                  n_layers:    int   = 4,
1572:                  dropout:     float = 0.1):
1573:         super().__init__()
1574: 
1575:         self.n_genes     = n_genes
1576:         self.n_bins      = n_bins
1577:         self.n_conf_bins = n_conf_bins
1578:         self.d_model     = d_model
1579: 
1580:         self.embedding = GeneTokenEmbedding(
1581:             n_genes, n_bins, n_conf_bins,
1582:             d_gene, d_count, d_conf, d_model)
1583: 
1584:         encoder_layer = nn.TransformerEncoderLayer(
1585:             d_model         = d_model,
1586:             nhead           = n_heads,
1587:             dim_feedforward = d_model * 4,
1588:             dropout         = dropout,
1589:             batch_first     = True,   # (B, G, d_model) convention
1590:             norm_first      = True,   # pre-norm: more stable for deeper models
1591:         )
1592:         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
1593: 
1594:         self.head_count = nn.Linear(d_model, n_bins)
1595:         self.head_conf  = nn.Linear(d_model, n_conf_bins)
1596: 
1597:         # Fixed gene-id buffer (0..n_genes-1), registered so it moves with .to(device)
1598:         self.register_buffer('_gene_ids',
1599:                              torch.arange(n_genes).unsqueeze(0))  # (1, G)
1600: 
1601:     def forward(self,
1602:                 count_bins: torch.Tensor,
1603:                 conf_bins:  torch.Tensor
1604:                 ) -> tuple[torch.Tensor, torch.Tensor]:
1605:         """
1606:         Args:
1607:             count_bins: (B, G) long -- current count bin for each gene.
1608:                         Unknown genes have count_bins == n_bins.
1609:             conf_bins:  (B, G) long -- current confidence bin.
1610:                         Unknown/low-confidence genes have conf_bins == 0.
1611:         Returns:
1612:             count_logits: (B, G, n_bins)
1613:             conf_logits:  (B, G, n_conf_bins)
1614:         """
1615:         B, G = count_bins.shape
1616: 
1617:         gene_ids = self._gene_ids.expand(B, -1)   # (B, G)
1618: 
1619:         # Token embeddings
1620:         tokens = self.embedding(gene_ids, count_bins, conf_bins)   # (B, G, d_model)
1621: 
1622:         # Confidence-gated key mask:
1623:         # src_key_padding_mask: (B, G), True = this position is MASKED OUT as a key.
1624:         # We mask out positions where conf_bin == 0 (unknown / lowest confidence).
1625:         key_pad_mask = (conf_bins == 0)  # (B, G), bool
1626: 
1627:         # Guard against all-masked rows (would produce NaN in softmax).
1628:         # If every gene in a sample is unknown (shouldn't happen in practice but
1629:         # be safe), unmask everything for that sample.
1630:         all_masked = key_pad_mask.all(dim=1)  # (B,)
1631:         if all_masked.any():
1632:             key_pad_mask[all_masked] = False
1633: 
1634:         out = self.transformer(tokens, src_key_padding_mask=key_pad_mask)  # (B, G, d_model)
1635: 
1636:         count_logits = self.head_count(out)   # (B, G, n_bins)
1637:         conf_logits  = self.head_conf(out)    # (B, G, n_conf_bins)
1638: 
1639:         return count_logits, conf_logits
1640: 
1641: 
1642: # ---------------------------------------------------------------------------
1643: # 4. epoch_transformer -- training / evaluation loop
1644: # ---------------------------------------------------------------------------
1645: 
1646: def epoch_transformer(
1647:     model:         ImputationTransformer,
1648:     loader:        DataLoader,
1649:     bin_edges:     torch.Tensor,
1650:     opt:           torch.optim.Optimizer | None = None,
1651:     mask_fraction: float = 0.10,
1652:     lambda_conf:   float = 0.10,
1653:     device:        torch.device | str = 'cpu',
1654: ) -> tuple[float, float, float, float]:
1655:     """
1656:     One epoch of training (opt != None) or evaluation (opt == None).
1657: 
1658:     Masking strategy:
1659:       - Type 1 (NaN in raw data):  always set to unknown token + conf bin 0.
1660:       - Type 2 (random ~mask_fraction of *observed* positions): same treatment;
1661:         cross-entropy loss is computed on these positions.
1662: 
1663:     Losses:
1664:       - count_ce:  cross-entropy of predicted count bins vs. true bins,
1665:                    averaged over Type-2 masked positions.
1666:       - conf_ce:   cross-entropy of predicted confidence vs. error-derived target,
1667:                    weighted by lambda_conf.
1668: 
1669:     Args:
1670:         model:         ImputationTransformer instance.
1671:         loader:        DataLoader yielding raw log1p tensors (B, G), possibly NaN.
1672:         bin_edges:     (n_bins+1,) edges from make_log_bin_edges.
1673:         opt:           optimizer (None for eval mode).
1674:         mask_fraction: fraction of observed genes to mask for training.
1675:         lambda_conf:   weight of the confidence auxiliary loss.
1676:         device:        torch device.
1677: 
1678:     Returns:
1679:         (total_loss, count_ce, conf_ce, grad_norm)  -- all floats, averaged over batches.
1680:     """
1681:     model.train() if opt is not None else model.eval()
1682:     n_bins      = model.n_bins
1683:     n_conf_bins = model.n_conf_bins
1684: 
1685:     total_loss = conf_ce_sum = count_ce_sum = grad_norm_sum = 0.0
1686:     n_batches  = 0
1687: 
1688:     ctx = torch.enable_grad() if opt is not None else torch.no_grad()
1689: 
1690:     with ctx:
1691:         for x_raw in loader:
1692:             x_raw = x_raw.to(device)  # (B, G), float, may contain NaN
1693: 
1694:             # --- Discretize observed values ---
1695:             true_count_bins = discretize(x_raw, bin_edges)  # (B, G) long
1696:             # Confidence proxy: higher count -> higher confidence (simple heuristic).
1697:             # Observed positions get conf bins 1..n_conf_bins-1 proportional to
1698:             # their count level; NaN positions start at 0.
1699:             nan_mask  = torch.isnan(x_raw)                  # (B, G) bool
1700:             obs_count = true_count_bins.float().clamp(0, n_bins - 1)
1701:             # Map [0, n_bins-1] -> [1, n_conf_bins-1] for observed genes
1702:             true_conf_bins = (obs_count / (n_bins - 1) * (n_conf_bins - 2) + 1).long()
1703:             true_conf_bins = true_conf_bins.clamp(1, n_conf_bins - 1)
1704:             true_conf_bins[nan_mask] = 0   # unknown -> lowest confidence
1705: 
1706:             # --- Type 2 mask: random subset of observed positions ---
1707:             type2_mask = (torch.rand_like(x_raw) < mask_fraction) & ~nan_mask  # (B, G) bool
1708:             all_mask   = nan_mask | type2_mask                                  # (B, G) bool
1709: 
1710:             # Build model inputs: masked positions get unknown token + conf bin 0
1711:             inp_count = true_count_bins.clone()
1712:             inp_conf  = true_conf_bins.clone()
1713:             inp_count[all_mask] = n_bins   # unknown token
1714:             inp_conf[all_mask]  = 0        # lowest confidence
1715: 
1716:             # --- Forward pass ---
1717:             count_logits, conf_logits = model(inp_count, inp_conf)
1718:             # count_logits: (B, G, n_bins), conf_logits: (B, G, n_conf_bins)
1719: 
1720:             # --- Count cross-entropy loss (Type-2 positions only) ---
1721:             if type2_mask.any():
1722:                 # True count bins for Type-2 positions (exclude the unknown token index)
1723:                 true_count_t2 = true_count_bins[type2_mask].clamp(0, n_bins - 1)
1724:                 pred_count_t2 = count_logits[type2_mask]          # (N2, n_bins)
1725:                 count_ce      = F.cross_entropy(pred_count_t2, true_count_t2)
1726: 
1727:                 # --- Confidence auxiliary loss ---
1728:                 # Target: how accurate was the count prediction at Type-2 positions?
1729:                 with torch.no_grad():
1730:                     pred_count_argmax = pred_count_t2.argmax(dim=-1)
1731:                     conf_target = error_to_conf_bin(
1732:                         pred_count_argmax, true_count_t2, n_conf_bins)
1733:                 pred_conf_t2 = conf_logits[type2_mask]            # (N2, n_conf_bins)
1734:                 conf_ce      = F.cross_entropy(pred_conf_t2, conf_target)
1735:             else:
1736:                 count_ce = torch.tensor(0.0, device=device)
1737:                 conf_ce  = torch.tensor(0.0, device=device)
1738: 
1739:             loss = count_ce + lambda_conf * conf_ce
1740: 
1741:             if opt is not None:
1742:                 opt.zero_grad()
1743:                 loss.backward()
1744:                 gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
1745:                 opt.step()
1746:                 grad_norm_sum += float(gn)
1747: 
1748:             total_loss   += float(loss)
1749:             count_ce_sum += float(count_ce)
1750:             conf_ce_sum  += float(conf_ce)
1751:             n_batches    += 1
1752: 
1753:     n_batches = max(n_batches, 1)
1754:     return (total_loss   / n_batches,
1755:             count_ce_sum / n_batches,
1756:             conf_ce_sum  / n_batches,
1757:             grad_norm_sum / n_batches)
1758: 
1759: 
1760: # ---------------------------------------------------------------------------
1761: # 5. impute_transformer -- iterative inference
1762: # ---------------------------------------------------------------------------
1763: 
1764: def impute_transformer(
1765:     model:               ImputationTransformer,
1766:     x_raw:               torch.Tensor,
1767:     bin_edges:           torch.Tensor,
1768:     device:              torch.device | str = 'cpu',
1769:     max_iters:           int   = 5,
1770:     convergence_thresh:  float = 0.01,
1771: ) -> tuple[torch.Tensor, torch.Tensor]:
1772:     """
1773:     Iteratively impute missing values using the trained ImputationTransformer.
1774: 
1775:     At each iteration:
1776:       1. Forward pass (only unknown positions are predicted).
1777:       2. Update unknown positions with argmax of predicted count logits.
1778:       3. Compute entropy of predicted distribution -> new confidence bin.
1779:       4. Check convergence: fraction of unknown positions that changed bin.
1780: 
1781:     Args:
1782:         model:              trained ImputationTransformer (eval mode).
1783:         x_raw:              (B, G) float, NaN at missing positions.
1784:         bin_edges:          (n_bins+1,) edges from make_log_bin_edges.
1785:         device:             torch device.
1786:         max_iters:          maximum number of refinement passes.
1787:         convergence_thresh: stop when < this fraction of unknown bins change.
1788: 
1789:     Returns:
1790:         imputed_continuous: (B, G) float -- dequantized bin midpoints for all genes.
1791:         imputed_bins:       (B, G) long  -- final bin indices.
1792:     """
1793:     model.eval()
1794:     n_bins      = model.n_bins
1795:     n_conf_bins = model.n_conf_bins
1796:     x_raw       = x_raw.to(device)
1797: 
1798:     nan_mask        = torch.isnan(x_raw)           # (B, G) -- positions to impute
1799:     true_count_bins = discretize(x_raw, bin_edges)  # (B, G); NaN -> n_bins
1800: 
1801:     # Initialise: observed positions keep true bins + proportional conf; unknown = bin 0
1802:     obs_count      = true_count_bins.float().clamp(0, n_bins - 1)
1803:     true_conf_bins = (obs_count / (n_bins - 1) * (n_conf_bins - 2) + 1).long().clamp(1, n_conf_bins - 1)
1804:     true_conf_bins[nan_mask] = 0
1805: 
1806:     cur_count = true_count_bins.clone()   # (B, G) long -- working count bins
1807:     cur_conf  = true_conf_bins.clone()    # (B, G) long -- working conf bins
1808:     # Unknown positions start with the unknown token
1809:     cur_count[nan_mask] = n_bins
1810:     cur_conf[nan_mask]  = 0
1811: 
1812:     with torch.no_grad():
1813:         for _iter in range(max_iters):
1814:             count_logits, _conf_logits = model(cur_count, cur_conf)
1815:             # (B, G, n_bins), (B, G, n_conf_bins)
1816: 
1817:             # Predicted bin for each position
1818:             pred_bins = count_logits.argmax(dim=-1)   # (B, G) long
1819: 
1820:             # New confidence from entropy of count predictions
1821:             new_conf = entropy_to_conf_bin(count_logits, n_conf_bins)  # (B, G) long
1822: 
1823:             # Check convergence: fraction of unknown positions that changed
1824:             prev_unknown_bins = cur_count[nan_mask]
1825:             new_unknown_bins  = pred_bins[nan_mask]
1826:             changed_frac      = (prev_unknown_bins != new_unknown_bins).float().mean().item()
1827: 
1828:             # Update only unknown positions
1829:             cur_count[nan_mask] = pred_bins[nan_mask]
1830:             cur_conf[nan_mask]  = new_conf[nan_mask]
1831: 
1832:             if changed_frac < convergence_thresh:
1833:                 break
1834: 
1835:     # Dequantize to continuous values for correlation evaluation
1836:     imputed_continuous = bins_to_midpoints(cur_count, bin_edges)   # (B, G) float
1837:     # Restore observed positions to their true continuous values
1838:     imputed_continuous[~nan_mask] = x_raw[~nan_mask]
1839: 
1840:     return imputed_continuous, cur_count
1841: 
1842: # ---------------------------------------------------------------------------
1843: # Unified imputation adapter
1844: # ---------------------------------------------------------------------------
1845: 
1846: def impute(model, x_raw, frac, *, device=device):
1847:     """Apply a type-2 mask at rate *frac* and return imputed values.
1848: 
1849:     Works for all model types without any caller-side preprocessing.
1850: 
1851:     Args:
1852:         model   : any of GeneExpressionVAE / MoVEVAE / VampPriorVAE /
1853:                   MeansModel / ImputationTransformer — must be in eval mode.
1854:         x_raw   : (B, n_genes) float tensor with NaN for type-1 missing values.
1855:         frac    : float, fraction of *observed* positions to mask for evaluation.
1856:         device  : torch.device to run inference on.
1857: 
1858:     Returns:
1859:         recon      : (B, n_genes) float tensor on CPU, imputed continuous values.
1860:         type2_mask : (B, n_genes) bool tensor on CPU, positions used for scoring.
1861:     """
1862:     x_raw = x_raw.to(device)
1863:     nan_mask = torch.isnan(x_raw)                          # type-1 positions
1864: 
1865:     # Draw type-2 mask: observed positions only
1866:     type2_mask = (torch.rand_like(x_raw) < frac) & ~nan_mask   # (B, G), CPU-safe
1867: 
1868:     model.eval()
1869:     with torch.no_grad():
1870:         if isinstance(model, ImputationTransformer):
1871:             # Mask type-2 positions by treating them as NaN for the transformer
1872:             x_eval = x_raw.clone()
1873:             x_eval[type2_mask] = float('nan')
1874:             recon, _ = impute_transformer(
1875:                 model, x_eval, model.bin_edges,
1876:                 device=device, max_iters=5)
1877:             recon = recon.cpu()
1878: 
1879:         elif isinstance(model, MeansModel):
1880:             # MeansModel ignores its input; output is the per-gene mean broadcast
1881:             # to batch size — no masking pre-processing needed.
1882:             recon, *_ = model(x_raw.nan_to_num(0))
1883:             recon = recon.detach().cpu()
1884: 
1885:         elif isinstance(model, VAEBase):
1886:             # Apply get_random_mask logic manually so we control the type-2 mask
1887:             combined_mask = type2_mask | nan_mask           # what the model sees as missing
1888:             x_masked = x_raw.clone()
1889:             x_masked = x_masked.where(~combined_mask,
1890:                                       x_raw.nanmean(dim=0).expand_as(x_raw))
1891:             x_masked += torch.randn_like(x_masked) * 0.05  # same augmentation as training
1892:             x_in = torch.cat((x_masked, combined_mask.float()), dim=1)  # (B, n_genes*2)
1893:             recon, *_ = model(x_in)
1894:             recon = recon.detach().cpu()
1895: 
1896:         else:
1897:             raise TypeError(f"impute(): unsupported model type {type(model).__name__}")
1898: 
1899:     return recon, type2_mask.cpu()
1900: 
1901: 
1902: # ---------------------------------------------------------------------------
1903: # Model factories
1904: # One callable per named architecture; each accepts a single argument: n_genes
1905: # an *un-trained* model instance (not yet moved to device).
1906: # ---------------------------------------------------------------------------
1907: 
1908: def _make_gene_vae(n_genes:int):
1909:     return GeneExpressionVAE(
1910:         n_genes=n_genes,
1911:         dropout=0.02,
1912:         encoder_dims=[512, 256, 128],
1913:         decoder_dims=[128, 256, 256],
1914:         latent_dim=32)
1915: 
1916: def _make_move_vae_k3(n_genes:int):
1917:     return MoVEVAE(
1918:         n_genes=n_genes,
1919:         n_components=3,
1920:         dropout=0.05,
1921:         encoder_dims=[200, 164, 96],
1922:         decoder_dims=[64, 128],
1923:         latent_dim=30)
1924: 
1925: def _make_move_vae_k1(n_genes:int):
1926:     return MoVEVAE(
1927:         n_genes=n_genes,
1928:         n_components=1,
1929:         dropout=0.05,
1930:         encoder_dims=[512, 256, 128],
1931:         decoder_dims=[128, 200, 200],
1932:         latent_dim=80)
1933: 
1934: def _make_vamp_vae(n_genes:int):
1935:     # pseudo_init_samples=None → random pseudo-inputs; a loaded state_dict will
1936:     # overwrite them with the saved values.
1937:     return VampPriorVAE(
1938:         n_genes=n_genes,
1939:         n_pseudo=50,
1940:         dropout=0.02,
1941:         encoder_dims=[512, 256],
1942:         decoder_dims=[200, 200],
1943:         latent_dim=24,
1944:         pseudo_init_samples=None)
1945: 
1946: def _make_means_model(n_genes:int):
1947:     return MeansModel(n_genes=n_genes)
1948: 
1949: 
1950: # Hyper-parameters (small config for ~200 genes)
1951: it_n_bins       = 24
1952: it_n_conf_bins  = 8
1953: it_d_gene       = 64
1954: it_d_count      = 64
1955: it_d_conf       = 32
1956: it_d_model      = 128   # smaller than full 256 for 200-gene scale
1957: it_n_heads      = 4
1958: it_n_layers     = 3
1959: it_dropout      = 0.1
1960: it_lr           = 3e-4
1961: it_batch_size   = 128
1962: it_max_epochs   = 100
1963: it_mask_frac    = 0.10
1964: it_lambda_conf  = 0.10
1965: 
1966: # Bin edges -- shared between training and evaluation
1967: it_bin_edges = make_log_bin_edges(it_n_bins, max_val=8.5)
1968: 
1969: def _make_imputation_transformer(n_genes:int):
1970:     return ImputationTransformer(
1971:         n_genes     = n_genes,
1972:         n_bins      = it_n_bins,
1973:         n_conf_bins = it_n_conf_bins,
1974:         d_gene      = it_d_gene,
1975:         d_count     = it_d_count,
1976:         d_conf      = it_d_conf,
1977:         d_model     = it_d_model,
1978:         n_heads     = it_n_heads,
1979:         n_layers    = it_n_layers,
1980:         dropout     = it_dropout)
1981: 
1982: model_factories = {
1983:     'MeansModel':           _make_means_model,
1984:     'GeneExpressionVAE':    _make_gene_vae,
1985:     'MoVEVAE_K3':           _make_move_vae_k3,
1986:     'MoVEVAE_K1':           _make_move_vae_k1,
1987:     'VampPriorVAE':         _make_vamp_vae,
1988:     'ImputationTransformer':_make_imputation_transformer,
1989: }
1990: 

(End of file - total 1990 lines)
</content>
```

---

## Assistant (Plan · Claude Sonnet 4.6 · 62.3s)

Now I have a thorough understanding of the codebase. Let me structure my analysis.

Here is a clear breakdown of the question and the design space:

---

## Context: what `make_synthetic_data5` currently produces

The function takes a caller-supplied `cell_state` tensor `(n_cells, n_gene_groups)` — typically from `random_points_on_hypersphere` — and generates:

- **`X`** `(n_cells, n_genes)`: observed gene-expression data (log1p NB counts, with MCAR NaN dropout)
- **`group_membership`**: sparse binary `(n_genes, n_gene_groups)` — which genes belong to which groups

The generation pipeline is:

```
cell_state  →  group_membership @ cell_state.T  →  influence_weights  →  NB sampling  →  log1p  →  NaN dropout  →  X
```

The function does **not** return `z` (a proper latent code per cell) or make the generative factors explicit. The caller only has `cell_state` (which was the input, not a derived latent), and `X` as output.

---

## The question: what are `z` and `x` in this context?

In the VAE + DEQ framing:

| Symbol | Meaning | Notes |
|--------|---------|-------|
| `z` | True latent state per cell | Could be `cell_state`, or a richer per-cell representation derived from it |
| `x` | Noisy, incomplete observed gene expression | Currently returned as `X` with NaN |

The question is: **what latent structure should `z` have, and what additional structure in `x` would make the dataset a good testbed for DEQ-based VAE priors?**

---

## Approaches to extending `make_synthetic_data5` to return `z` and `x`

### Approach 1: `z = cell_state` directly (minimal change)

The simplest option. The function already receives `cell_state` and uses it as the latent driver. Return it alongside `X`.

```python
return X, group_membership_np, cell_state  # z = cell_state
```

**Pros:**
- Ground truth `z` is available for evaluating posterior collapse, disentanglement, and latent geometry
- The generative model `p(x|z)` is fully explicit (NB with mean = f(group_membership @ z.T))
- Enables computing posterior `q(z|x)` vs. true `z` correlations after training

**Cons:**
- `z` is the raw manifold coordinate, not a normalized latent — may need standardization
- The mapping `z → x` is complex (group membership, influence weights, NB, log1p, NaN) — hard to invert analytically, which is realistic but means you can't easily check decoder accuracy symbolically

**Good for:** ground-truth-guided evaluation of latent recovery, testing DEQ convergence as an implicit encoder

---

### Approach 2: Two-level `z`: global cell type + local continuous state

Introduce a discrete cluster assignment `y` and a continuous within-cluster deviation `delta_z`, so the full latent is `z = (y, delta_z)`.

```python
# Use n_clusters from random_points_on_hypersphere with n_clusters > 1
cell_state, labels = random_points_on_hypersphere(..., n_clusters=K, concentration=c)
```

`random_points_on_hypersphere` already returns `labels` but `make_synthetic_data5` discards them. Return both.

```python
return X, group_membership_np, cell_state, labels  # z = (cell_state, labels)
```

**Pros:**
- Tests whether the model can discover discrete modes without supervision (anti-mode-collapse test)
- Natural for VampPrior / MoVE / MoE-DEQ architectures
- Labels enable purity/ARI evaluation of learned clusters

**Cons:**
- The cluster structure is soft (vMF, not hard boundaries) — need `concentration >> 1` to get well-separated modes
- Adds a hyperparameter (`n_clusters`, `concentration`) that needs to be tuned to avoid trivially easy or trivially hard clustering

**Good for:** testing that DEQ-prior VAEs don't collapse modes and correctly model multi-modal distributions

---

### Approach 3: Hierarchical `z` matching `groups_def` structure

The current `groups_def` has a 3-level hierarchy (6 root groups → intermediate → leaf). The `cell_state` dimensions correspond to these 32 group IDs, but they're treated as flat. A richer extension would make `z` explicitly hierarchical:

- Level 0: 6-dimensional "cell type" vector (coarse)
- Level 1: intermediate program activation (conditioned on level 0)
- Level 2: leaf-level activations (conditioned on level 1)

This mirrors real biology where cell type → sub-type → state hierarchy exists.

```python
# z_coarse: (n_cells, 6) — root group activations
# z_mid:    (n_cells, 14) — intermediate activations
# z_fine:   (n_cells, 12) — leaf activations
# z = concat([z_coarse, z_mid, z_fine])  # (n_cells, 32)
```

The key addition: generate the hierarchy causally (coarse → mid → fine) so that:
- `z_mid` is sampled conditioned on `z_coarse` (linear + noise)
- `z_fine` is sampled conditioned on `z_mid`
- `X` is generated from the full `z`

**Pros:**
- Tests whether DEQ fixed-point iteration can capture hierarchical correlation structure
- Ground truth at each level allows multi-resolution evaluation
- Natural fit for a hierarchical VAE prior or a DEQ with block-structured Jacobian

**Cons:**
- Adds significant complexity to the data generator
- Requires choosing the causal strengths (how much does coarse predict mid?)
- The groups_def hierarchy is already implicit in the membership structure; making it explicit in `z` may be redundant unless the architecture is designed to exploit it

**Good for:** benchmarking hierarchical or structured DEQ priors

---

### Approach 4: Explicit `z` with a DEQ-compatible fixed-point structure

Design `z` so that it is the fixed point of a map `F(z; x)`. This directly mirrors what a DEQ encoder would learn, and makes the synthetic data a natural testbed.

Concretely, define `z` as the solution to:

```
z = tanh(W_xx @ x_obs + W_zz @ z + b)   [fixed point iteration]
```

where `W_xx`, `W_zz`, `b` are fixed random matrices (not learned), and `x_obs` is a partially-observed version of `X`. The "true" latent is this fixed point, and the DEQ encoder's job is to recover it.

**Pros:**
- The data has a ground-truth DEQ structure — the DEQ encoder is provably the right inductive bias
- Enables measuring whether the encoder converges to the correct fixed point
- Can be made to have spectral radius < 1 (by normalizing `W_zz`) so fixed points are unique and well-conditioned
- Tests sample efficiency: fewer observations needed if the DEQ prior is correctly specified

**Cons:**
- The data generation is now tied to a specific DEQ architecture, making it less general as a benchmark
- Fixed random `W_zz` may not reflect realistic biological priors
- Computing the true fixed point during data generation requires running the iteration to convergence (straightforward but adds computation)

**Good for:** a direct test of DEQ encoder correctness; less suitable for broad architecture comparison

---

### Approach 5: Return multiple levels of `x` (clean, noisy, incomplete)

Rather than adding structure to `z`, extend `make_synthetic_data5` to return multiple versions of `x`:

| Return | Description |
|--------|-------------|
| `x_latent` | Pre-NB linear output (the latent signal, no NB noise, no NaN) |
| `x_clean` | log1p NB counts without NaN (noise but no missing) |
| `x_obs` | log1p NB counts with NaN (noise + missing — current output) |

This lets you separately evaluate:
- Whether the model recovers the latent signal (reconstruction of `x_latent`)
- Whether it handles NB noise correctly (reconstruction of `x_clean`)
- Whether it handles missing data without imputation preprocessing (reconstruction of `x_obs`)

**Pros:**
- No architectural assumptions in the data generator
- Ground truth at multiple noise levels enables fine-grained diagnostics
- Orthogonal to the choice of `z` — can be combined with any of the above

**Cons:**
- Doesn't add new structural challenges, only new evaluation signals
- `x_latent` is not the same as `z` (it's `n_cells × n_genes`, not a low-dim latent)

**Good for:** diagnostic richness without complicating the generative model

---

## Recommended combined approach

For a DEQ-prior VAE testbed that is:
- **sample-efficient**: needs structured, low-dimensional `z`
- **anti-mode-collapse**: needs multiple discrete modes
- **handles missing data without imputation**: needs NaN-robust `x`
- **good inductive bias test**: `z` structure should match what the DEQ prior can capture

The recommended extension is **Approach 1 + 2 + 5**:

1. Return `z = cell_state` (the true manifold coordinate) — enables posterior evaluation
2. Generate `cell_state` with `n_clusters > 1` and moderate `concentration` (e.g. 5–20) — enables mode collapse detection
3. Return `x_latent` (pre-NB signal) alongside `x_obs` — enables multi-level evaluation

A concrete signature:

```python
def make_synthetic_data5(
    cell_state:    torch.Tensor,   # (n_cells, n_gene_groups) — z
    ...
) -> tuple[
    torch.Tensor,    # X:               (n_cells, n_genes) — log1p NB, NaN for missing
    np.ndarray,      # group_membership: (n_genes, n_gene_groups)
    torch.Tensor,    # z:               (n_cells, n_gene_groups) = cell_state (normalized)
    torch.Tensor,    # x_latent:        (n_cells, n_genes) — pre-NB linear signal, no NaN
]:
```

The `cell_state` / `z` returned would be `cell_state * corr_scale` (what actually drives the generation), not the raw input.

---

## A note on DEQ-as-prior vs. DEQ-as-encoder

There are two distinct roles a DEQ can play here, and they have different implications for what `z` and `x` should look like:

| Role | Description | What `z` should be |
|------|-------------|---------------------|
| **DEQ as encoder prior** `p(z)` | DEQ defines the prior distribution over latents; encoder `q(z|x)` is standard MLP | `z` on a structured manifold (hypersphere, hierarchical, multi-modal) |
| **DEQ as encoder** `q(z|x)` | DEQ iteratively refines `z` given `x` at test time; handles missing data naturally | `z` as the fixed point of `F(z; x_obs)`, where `x_obs` has NaN masked out |
| **DEQ as decoder** `p(x|z)` | DEQ generates `x` from `z` by fixed-point iteration | `z` low-dimensional, `x` high-dimensional with structured correlations |

Approach 4 is most relevant for testing the DEQ-as-encoder role. Approaches 1–3 are better for testing DEQ-as-prior.

---

Would you like me to:
1. Implement a specific combination of the above approaches?
2. Focus on a particular role (DEQ as encoder, prior, or decoder)?
3. Add specific evaluation metrics alongside the data generator (e.g. posterior quality scores, mode coverage)?

---

## User

Please expand on the two DEQ roles (prior vs. encoder) and their implications for z and x

---

## Assistant (Plan · Claude Sonnet 4.6 · 194.1s)

This is a conceptual design question, not an implementation task — no files need to be read or changed. Let me work through this carefully.

---

## DEQ roles: prior vs. encoder

### Role 1: DEQ as a prior `p(z)`

The DEQ defines the **distribution over latents** before any data is seen. The prior is the fixed-point distribution of a stochastic map:

```
z* = F(z*, ε),   ε ~ N(0, I)
```

where `F` is a learned (or fixed) function and `z*` is the unique fixed point given `ε`. The prior is implicitly defined — you can't write `p(z*)` in closed form, but you can sample from it by running the fixed-point iteration to convergence. This is analogous to how a normalizing flow defines a complex distribution by transforming a simple one, except the transformation is defined implicitly.

The VAE structure is:

```
Prior:    z* = F(z*, ε),  ε ~ N(0, I)     [DEQ — implicit p(z)]
Encoder:  q(z|x) = N(μ_φ(x), σ_φ(x))     [standard MLP amortized inference]
Decoder:  p(x|z) = NB(f_θ(z))             [standard MLP generative model]
```

**Implications for `z` in synthetic data:**

The test for whether the DEQ prior is a good inductive bias is whether its fixed-point distribution `p(z*)` matches the true latent distribution `p(z_true)`. So `z_true` (what `make_synthetic_data5` should return) needs to be on a manifold that the DEQ prior can represent well but a simple Gaussian prior cannot.

Good choices for `z_true`:
- Manifold-constrained: `cell_state` from `random_points_on_hypersphere` — lives on a rotated low-dimensional sphere embedded in high-dim space, which a Gaussian prior misrepresents badly (it assigns mass off the manifold)
- Multi-modal: `cell_state` with `n_clusters > 1` — a Gaussian prior collapses modes; a DEQ prior can learn multi-modal fixed-point attractors
- Hierarchically correlated: the root/intermediate/leaf structure in `groups_def` — a Gaussian prior treats all latent dims independently; a DEQ prior with coupling can capture cross-dim correlations

**Implications for `x` in synthetic data:**

`x` is the observation generated from `z_true`. For the prior role, what matters most is that:
1. `x` is high-dimensional and noisy (NB counts + NaN) — so a simple linear model can't recover `z_true` from `x`
2. There is genuine information loss from `z → x` — the mapping is many-to-one due to dropout and NB noise, so the posterior `q(z|x)` is genuinely uncertain

The current `make_synthetic_data5` pipeline already satisfies this: the NB sampling and NaN dropout both destroy information. You don't need to change `x` for the prior role — but you do need `z_true` to be returned so you can measure whether the learned prior matches it.

**Key evaluation signal for DEQ-as-prior:**

After training, sample `z ~ p_DEQ(z*)` by running the DEQ to convergence on random `ε`, then check whether the distribution of `z*` samples matches the distribution of `z_true`. Metrics: MMD, Wasserstein distance on the marginal, or visual comparison via UMAP. The VAE encoder is still a standard MLP; it's the prior that's being tested.

---

### Role 2: DEQ as an encoder `q(z|x_obs)`

Here the DEQ defines the **inference procedure**: given a (possibly incomplete) observation `x_obs` with NaN entries, the encoder finds a latent `z` as the fixed point of:

```
z* = G(z*, x_obs),   where G is learned
```

The fixed-point structure is what gives this inductive bias over missing data: `G` takes both the current guess of `z` and the available observations, and iterates until consistent. Missing entries in `x_obs` don't need to be imputed before encoding — the DEQ naturally marginalizes over them because `G` operates on the observed subset at each iteration.

The VAE structure is:

```
Prior:    p(z) = N(0, I)                            [standard Gaussian]
Encoder:  z* = G(z*, x_obs),  z* ~ q(z|x_obs)      [DEQ — implicit amortized inference]
Decoder:  p(x|z) = NB(f_θ(z))                       [standard MLP generative model]
```

**Implications for `z` in synthetic data:**

For the encoder role, what matters is that `z_true` is recoverable from `x_obs` — the fixed-point iteration should converge to `z_true` when given the observed subset of `x`. This places constraints on the signal-to-noise ratio and the missing rate. If too many genes are missing, no encoder (DEQ or otherwise) can recover `z_true`.

`z_true` should be:
- Low-dimensional relative to `n_genes` — so the DEQ has a compressed representation to converge to (16-d manifold in 200-d gene space is reasonable)
- Identifiable: different `z_true` values should produce reliably different patterns in `x_obs` even after NaN dropout

Crucially, the DEQ encoder has a specific advantage: it can iteratively refine its estimate of `z*` using progressively more consistent interpretations of the partial observation. This is analogous to message-passing inference. This advantage is most pronounced when:
- Missing entries are **not MCAR** but **structured** (e.g., a whole gene module is missing, which is more realistic biologically) — a DEQ can use the observed modules to refine the estimate and then infer the missing ones implicitly
- The observation model has **within-group correlation** (which `make_synthetic_data5` provides via `group_membership`) — the DEQ can exploit this correlation structure through iteration

**Implications for `x` in synthetic data:**

This is where the extension to `make_synthetic_data5` matters most for the encoder role. The function should return:

1. **`x_latent`** `(n_cells, n_genes)`: the pre-NB linear signal `(group_membership @ z_true.T).T * influence_weights` — this is the noiseless, complete signal that a perfect encoder should recover after projecting back to `z_true`. Needed to measure whether the DEQ is converging to the right fixed point.

2. **`x_clean`** `(n_cells, n_genes)`: log1p NB counts without NaN — lets you separate the effect of NB noise from missing data on encoder performance.

3. **`x_obs`** `(n_cells, n_genes)`: current output — log1p NB counts with NaN.

And optionally a structured missing pattern beyond MCAR:

4. **Module-level missingness**: instead of random MCAR, drop entire gene groups (whole rows of `group_membership`) for some cells — tests whether the DEQ encoder can impute a missing module from correlated observed modules without any preprocessing. This is a much harder test than MCAR.

**Key evaluation signal for DEQ-as-encoder:**

Run the DEQ iteration for `k = 1, 2, 5, 10` steps and measure:
- How does `||z*(k) - z_true||` decrease with iteration count? A good DEQ encoder should show convergence. A standard MLP encoder has `k=1` by definition and cannot improve with more compute at test time.
- How does imputation quality of `x_obs` improve as `k` increases? (More iterations → better `z*` → better decoder output for missing positions)
- Sample efficiency: at what `n_cells` does the DEQ encoder match a standard MLP? If the DEQ prior is correctly specified, it should need fewer examples.

---

## Side-by-side comparison

| Dimension | DEQ as prior `p(z)` | DEQ as encoder `q(z|x_obs)` |
|-----------|--------------------|-----------------------------|
| Where DEQ lives | Prior over `z` | Inference network |
| Encoder | Standard MLP | DEQ fixed-point iteration |
| Prior | DEQ implicit distribution | Standard Gaussian |
| `z_true` should be | On a non-Gaussian manifold (sphere, multi-modal, hierarchical) | Recoverable from partial `x_obs`; low-dim relative to `n_genes` |
| `x` requirements | High-dim, noisy, NaN — just needs to be hard to invert with a linear model | Structured missingness (module-level, not just MCAR) is most discriminative |
| Test of inductive bias | Does the learned prior assign mass to the right manifold? | Does more iteration give better `z*`? Does it scale better with missing rate? |
| Mode collapse test | Sample from `p_DEQ(z*)` and check coverage | Run DEQ on cells from different clusters; check if it converges to distinct fixed points |
| Sample efficiency test | Train on fewer cells; does KL collapse earlier for Gaussian prior? | Train on fewer cells; does DEQ encoder maintain imputation quality longer? |
| Hardest case for baseline | Multi-modal `z_true` — Gaussian prior collapses minority modes | High fraction of structured (whole-module) missingness — MLP encoder sees too little signal |
| New return values needed | `z_true = cell_state * corr_scale`, `labels` (cluster assignments) | `z_true`, `x_latent`, `x_clean`, optionally structured `missing_mask` |

---

## Cross-cutting implication: the role of `influence_weights`

Currently in `make_synthetic_data5`, the per-(gene, cell) influence weights are sampled from `Gamma(10, 10)` — tight around 1, so they are nearly uniform. This means the `z → x` mapping is close to linear (up to NB noise), which makes the DEQ encoder's job easier than it should be.

For a more challenging testbed, the influence weights could be:
- Drawn from a heavier-tailed distribution (e.g. `Gamma(0.5, 0.5)`) — some genes are highly gene-group-specific, others are nearly silent; this creates the kind of sparse, strong-driver structure seen in real scRNA-seq
- Made cell-state-dependent — the influence weight for gene `g` in group `k` depends on `z_true` through a nonlinear function — making the effective observation model nonlinear and harder to invert

Both modifications increase the value of the DEQ's iterative refinement in the encoder role, and make the prior role harder for a standard Gaussian.

---

Does this framing help clarify which role to pursue? The two main follow-up questions that would determine the concrete extension to `make_synthetic_data5` are:

1. **Which role is the primary focus?** (Both can be supported simultaneously, but the structured missingness needed for encoder testing and the non-Gaussian manifold needed for prior testing point in slightly different directions for `z_true`.)

2. **Should the DEQ fixed-point map `G(z, x_obs)` be part of the data generator (Approach 4 from earlier), or should it only be part of the model architecture?** Making it part of the generator creates a "correctly specified" testbed but ties the benchmark to a specific DEQ form; keeping it only in the architecture leaves the generator agnostic and harder for the model.

---

