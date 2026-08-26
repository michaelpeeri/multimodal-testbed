import json
import os
import numpy as np
import torch
from tqdm import tqdm
from synthetic_data import (
    generate_sergio_grn_from_reference, load_sergio_dag, sample_sergio_mr_states,
    sergio_dag_hill_forward, sergio_dag_hill_forward_batched, compute_summary_stats
)


class MRStateGAOptimizer:
    """
                                          choice                          offspring                              fitness
    The process is (n_candidates,n_mrs) ---------->  (n_clusters,n_mrs) -------------> (n_clusters*10,n_genes) -----------> (1,)
    The unit of selection is (n_clusters, n_mrs)
    n_replicates: Independent split/subsampling seeds used to evaluate feature
                   stability. Set ``replicate_grns`` in the config to request
                   separate GRNs instead.
    
    """
    def __init__(
        self,
        base_cfg: str,
        n_population: int = 200,
        n_replicates: int = 1,
        n_clusters: int = 15,
        seed: int = 0,
    ):
        with open(base_cfg, 'rt') as f:
            self.cfg = json.load(f)

        if n_population < 2:
            raise ValueError("n_population must be at least 2")
        if n_replicates < 1:
            raise ValueError("n_replicates must be positive")
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")

        self.n_population = n_population
        self.n_replicates = n_replicates
        self.n_clusters   = n_clusters
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

        base = self.cfg.get('_meta', self.cfg)
        bestcfg = self.cfg.get('best_resolved_params', self.cfg.get('best_params', {}))

        def setting(name, default):
            return bestcfg.get(name, base.get(name, self.cfg.get(name, default)))

        self.grn_seed = int(base.get('grn_seed', self.cfg.get('grn_seed', 42)))
        self.mr_rate_low = float(setting('mr_rate_low', 1.0))
        self.mr_rate_high = float(setting('mr_rate_high', 5.0))
        if self.mr_rate_high <= self.mr_rate_low:
            raise ValueError("mr_rate_high must be greater than mr_rate_low")

        self.decays = setting('decays', 0.8)
        self.stats_n_pca_components = int(
            base.get('stats_n_pca_components', self.cfg.get('stats_n_pca_components', 20))
        )
        self.stats_n_structure_genes = base.get(
            'stats_n_structure_genes', self.cfg.get('stats_n_structure_genes', 500)
        )
        self.stats_percentiles = tuple(
            base.get('stats_percentiles', self.cfg.get('stats_percentiles', [5, 25, 50, 75, 95]))
        )
        self.stats_seed = int(
            base.get('stats_seed', self.cfg.get('stats_seed', 42))
        )
        default_repeats = max(
            1,
            int(round(float(base.get('n_cells', self.n_clusters * 20)) / self.n_clusters)),
        )
        self.surrogate_repeats = int(
            self.cfg.get('surrogate_cells_per_cluster', default_repeats)
        )
        if self.surrogate_repeats < 1:
            raise ValueError("surrogate_cells_per_cluster must be positive")
        self.selection_temperature = float(self.cfg.get('selection_temperature', 0.05))
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        self.replicate_grns = bool(self.cfg.get('replicate_grns', False))


    def generate_tree(self):
        base    = self.cfg.get('_meta', self.cfg)
        bestcfg = self.cfg.get('best_resolved_params', self.cfg.get('best_params', {}))

        def base_setting(name, default):
            return base.get(name, self.cfg.get(name, default))

        def parameter(name, default):
            return bestcfg.get(name, base_setting(name, default))

        grn_tmp_dir = base_setting('grn_tmp_dir', '/tmp') or '/tmp'
        os.makedirs(grn_tmp_dir, exist_ok=True)

        # harness:624
        def gen_grn(seed:int):
            temp_path = os.path.join(
                grn_tmp_dir, f'mr_state_ga_{os.getpid()}_{seed}.csv'
            )
            _, mr_ids, gene_id_to_symbol = generate_sergio_grn_from_reference(
                reference_grn_path=base_setting('reference_grn_path', None),
                n_genes=int(base_setting('n_genes', 800)),
                output_path=temp_path,
                k_dist=(
                    "uniform",
                    parameter('grn_k_low', 1.0),
                    parameter('grn_k_low', 1.0) + parameter('grn_k_span', 1.0),
                ),
                hill_coeff_dist=("constant", parameter('hill_coeff', 2.0)),
                seed=seed,
                coherency_bias=parameter('coherency_bias', 0.0),
                unknown_mode_repressor_prob=parameter('unknown_mode_repressor_prob', 0.5),
                canalization_strength=parameter('canalization_strength', 0.0),
                balancing_strength=parameter('balancing_strength', 0.0),
                path_decay=parameter('path_decay', 0.9),
                delimiter=base_setting('grn_delimiter', '\t'),
                regulator_col=int(base_setting('grn_regulator_col', 0)),
                target_col=int(base_setting('grn_target_col', 1)),
                mode_col=int(base_setting('grn_mode_col', 2)),
                activation_labels=base_setting('grn_activation_labels', ['Activation']),
                repression_labels=base_setting('grn_repression_labels', ['Repression']),
                max_seed_attempts=int(base_setting('grn_max_seed_attempts', 20)),
            )
            return temp_path, mr_ids, gene_id_to_symbol

        grn_seeds = (
            [self.grn_seed + rep for rep in range(self.n_replicates)]
            if self.replicate_grns else [self.grn_seed]
        )
        grns = []
        for seed in grn_seeds:
            temp_path, mr_ids, gene_id_to_symbol = gen_grn(seed=seed)
            grns.append( (seed, temp_path, mr_ids, gene_id_to_symbol) )

        dags = []
        for _grn in grns:
            seed, temp_path, mr_ids, gene_id_to_symbol = _grn
            dag = load_sergio_dag(
                temp_path,
                shared_coop_state=base_setting('shared_coop_state', 0.0),
                mr_gene_ids=mr_ids,
            )
            dags.append(dag)

        if self.replicate_grns:
            reference_mr_ids = tuple(grns[0][2])
            for _, _, mr_ids, _ in grns[1:]:
                if tuple(mr_ids) != reference_mr_ids:
                    raise ValueError(
                        'replicate GRNs do not have identical ordered MR IDs; '
                        'cannot evaluate one candidate matrix against all replicates'
                    )

        # A single fixed DAG avoids changing the MR universe between
        # replicate evaluations. Replicates still differ through the
        # statistics split/subsampling seed; separate DAGs are opt-in and are
        # accepted only when their ordered MR universes match exactly.
        dags = dags * self.n_replicates if not self.replicate_grns else dags
                    
        self.grns = grns
        self.dags = dags

    def initialize_population(self) -> np.ndarray:
        """
        Generate fresh candidates
        """
        if not getattr(self, 'grns', None) or not getattr(self, 'dags', None):
            raise RuntimeError('generate_tree() must be called before initializing the population')
        seed, temp_path, mr_ids, gene_id_to_symbol = self.grns[0]
        candidates = []
        for rep in range(self.n_population):
            new_candidates = sample_sergio_mr_states(
                n_states=self.n_clusters,
                n_mrs=len(mr_ids),
                low=self.mr_rate_low,
                high=self.mr_rate_high,
                design='sobol',
                seed=self.seed + rep,
            )
            candidates.append(new_candidates)
        return np.stack(candidates)


    def eval_candidates(self, candidates:np.ndarray):
        """
        """

        candidates = np.asarray(candidates, dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[0] != self.n_population:
            raise ValueError(
                f'expected candidates with shape ({self.n_population}, n_clusters, n_mrs), '
                f'got {candidates.shape}'
            )
        if candidates.shape[1] != self.n_clusters:
            raise ValueError(
                f'expected {self.n_clusters} cluster states, got {candidates.shape[1]}'
            )
        expected_mrs = len(self.dags[0].mr_ids)
        if candidates.shape[2] != expected_mrs:
            raise ValueError(
                f'expected {expected_mrs} MR columns for the loaded DAG, '
                f'got {candidates.shape[2]}'
            )
        if not np.isfinite(candidates).all():
            raise ValueError('candidate MR states must be finite')

        selected = np.repeat(candidates, repeats=self.surrogate_repeats, axis=1)
        print(f'selected:{selected.shape}')

        f = np.full((self.n_replicates, self.n_population), np.nan)
        for rep in range(self.n_replicates):
            print(f'-- rep {rep+1}/{self.n_replicates} --')

            batch_raw = sergio_dag_hill_forward_batched(
                selected, self.dags[rep], decays=self.decays)

            for i in tqdm(range(self.n_population)):
                raw = np.asarray(batch_raw[i], dtype=np.float64)
                log_expression = torch.tensor(
                    np.log1p(np.maximum(raw, 0.0)),
                    dtype=torch.float32,
                )
                stats = compute_summary_stats(
                    log_expression,
                    n_pca_components=self.stats_n_pca_components,
                    n_structure_genes=self.stats_n_structure_genes,
                    percentiles=self.stats_percentiles,
                    seed=self.stats_seed + rep,
                )
                score = stats['pca_standardized_split_half_subspace_stability']
                if np.isfinite(score):
                    f[rep, i] = score

        return f

    def next_generation(
        self,
        candidates,
        fitness,
        p_crossover=0.25,
        p_mutation=0.02,
        mutation_scale=0.1,
        n_elites=1,
    ):
        candidates = np.asarray(candidates, dtype=np.float32)
        fitness = np.asarray(fitness, dtype=np.float64)
        if candidates.shape[0] != self.n_population:
            raise ValueError('candidate count does not match n_population')
        if fitness.shape != (self.n_replicates, self.n_population):
            raise ValueError(
                f'expected fitness shape {(self.n_replicates, self.n_population)}, '
                f'got {fitness.shape}'
            )
        if not 0.0 <= p_crossover <= 1.0:
            raise ValueError('p_crossover must be in [0, 1]')
        if not 0.0 <= p_mutation <= 1.0:
            raise ValueError('p_mutation must be in [0, 1]')
        if mutation_scale < 0.0:
            raise ValueError('mutation_scale must be non-negative')
        if not 0 <= n_elites < self.n_population:
            raise ValueError('n_elites must be in [0, n_population)')

        scores = np.nanmean(fitness, axis=0)
        finite = np.isfinite(scores)
        if not finite.any():
            raise ValueError('all candidate fitness values are non-finite')
        worst_score = np.min(scores[finite])
        scores = np.where(finite, scores, worst_score)

        # Temperature-scaled softmax keeps selection pressure meaningful for
        # stability scores whose natural range is usually close to [0, 1].
        logits = (scores - np.max(scores)) / self.selection_temperature
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        rng = self.rng
        selected_indices = rng.choice(
            self.n_population,
            size=self.n_population,
            p=probabilities,
        )
        next_candidates = candidates[selected_indices].copy()

        elite_indices = np.argsort(scores)[-n_elites:][::-1]
        if n_elites:
            next_candidates[:n_elites] = candidates[elite_indices]

        # Crossover between distinct child slots. Parent rows are exchanged as
        # units, preserving each complete MR-state vector.
        n_children = self.n_population - n_elites
        n_crossover = min(int(p_crossover * 0.5 * n_children), n_children // 2)
        child_slots = np.arange(n_elites, self.n_population)
        rng.shuffle(child_slots)
        side_a_idx = child_slots[:2 * n_crossover:2]
        side_b_idx = child_slots[1:2 * n_crossover:2]
        a = next_candidates[side_a_idx].copy()
        b = next_candidates[side_b_idx].copy()

        crossover = np.repeat(
            rng.integers(0, 2, size=a.shape[:2] + (1,)),
            repeats=a.shape[2],
            axis=2,
        )
        a_post = a.copy()
        a_post[crossover==0] = b[crossover==0]

        b_post = b.copy()
        b_post[crossover==0] = a[crossover==0]

        next_candidates[side_a_idx] = a_post
        next_candidates[side_b_idx] = b_post

        # Gaussian mutation supplies new continuous MR rates; clipping keeps
        # every state valid for SERGIO and leaves elites untouched.
        if p_mutation and mutation_scale:
            mutation_mask = rng.random(next_candidates[n_elites:].shape) < p_mutation
            mutation = rng.normal(
                0.0,
                mutation_scale * (self.mr_rate_high - self.mr_rate_low),
                size=next_candidates[n_elites:].shape,
            )
            next_candidates[n_elites:] += mutation * mutation_mask
            next_candidates[n_elites:] = np.clip(
                next_candidates[n_elites:], self.mr_rate_low, self.mr_rate_high
            )

        return next_candidates


def evo_loop(config_fn, max_generations=20, **ga_kwargs):
    if max_generations < 1:
        raise ValueError('max_generations must be positive')
    ga = MRStateGAOptimizer(config_fn, **ga_kwargs)
    ga.generate_tree()
    candidates = ga.initialize_population()
    print(candidates.shape)

    evo_log = []
    best_candidate = None
    best_fitness = -np.inf
    for i in range(max_generations):
        print(f'=== gen {i+1}/{max_generations} ===')
        f        = ga.eval_candidates( candidates)
        mean_fitness = np.nanmean(f, axis=0)
        valid = np.isfinite(mean_fitness)
        if not valid.any():
            raise ValueError(f'generation {i} produced no finite fitness values')
        generation_best_idx = int(np.nanargmax(mean_fitness))
        generation_best = float(mean_fitness[generation_best_idx])
        if generation_best > best_fitness:
            best_fitness = generation_best
            best_candidate = candidates[generation_best_idx].copy()

        finite_fitness = f[np.isfinite(f)]
        qs = np.quantile(
            finite_fitness,
            q=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
        )
        evo_log.append({
            'generation': i,
            'mean_fitness': float(np.mean(finite_fitness)),
            'variance_fitness': float(np.var(finite_fitness)),
            'generation_best': generation_best,
            'best_so_far': best_fitness,
            'quantiles': [float(q) for q in qs],
        })
        next_gen = ga.next_generation(candidates, fitness=f, n_elites=4)
        candidates = next_gen.copy()
        print(f'{np.mean(finite_fitness):.4g} q99={qs[6]:.4g} best={best_fitness:.4g}')

    return {
        'best_candidate': best_candidate,
        'best_fitness': best_fitness,
        'population': candidates,
        'evolution_log': evo_log,
    }
