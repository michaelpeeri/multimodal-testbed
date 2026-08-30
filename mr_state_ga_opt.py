import json
import os
import numpy as np
from synthetic_data import (
    generate_sergio_grn_from_reference, load_sergio_dag, sample_sergio_mr_states,
    sergio_dag_hill_forward_batched,
    standardized_split_half_subspace_stability_batched,
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
        self.seed = int(seed)
        seed_sequence = np.random.SeedSequence(seed)
        init_seed, selection_seed, crossover_seed, mutation_seed = seed_sequence.spawn(4)
        self.initialization_rng = np.random.default_rng(init_seed)
        self.selection_rng = np.random.default_rng(selection_seed)
        self.crossover_rng = np.random.default_rng(crossover_seed)
        self.mutation_rng = np.random.default_rng(mutation_seed)
        # Keep DE's random stream separate so adding a DE run does not alter
        # the existing GA streams or their reproducibility.
        self.de_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 0xD1FF3])
        )
        self.last_pre_mutation_fitness = None

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
        self.fitness_batch_size = self.cfg.get('fitness_batch_size')
        if self.fitness_batch_size is not None:
            self.fitness_batch_size = int(self.fitness_batch_size)
            if self.fitness_batch_size < 1:
                raise ValueError('fitness_batch_size must be positive')
        self.last_eval_diagnostics = {}
        self.last_operator_diagnostics = {}
        self.last_crossover_slots = np.array([], dtype=int)
        self.last_crossover_parent_best_scores = np.array([], dtype=np.float64)
        self.last_pre_mutation_candidates = None


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
                seed=int(self.initialization_rng.integers(0, 2**32 - 1)),
            )
            candidates.append(new_candidates)
        return np.stack(candidates)

    def population_diagnostics(self, candidates: np.ndarray) -> dict:
        """Return inexpensive diagnostics for population collapse and bounds."""
        population = np.asarray(candidates, dtype=np.float64)
        if population.ndim != 3:
            raise ValueError(f'expected a 3D population, got {population.shape}')

        flat = population.reshape(population.shape[0], -1)
        state_rows = population.reshape(-1, population.shape[-1])
        finite_mask = np.isfinite(population)
        finite_values = population[finite_mask]
        value_summary = {
            'min': float(np.min(finite_values)) if finite_values.size else float('nan'),
            'max': float(np.max(finite_values)) if finite_values.size else float('nan'),
            'mean': float(np.mean(finite_values)) if finite_values.size else float('nan'),
            'std': float(np.std(finite_values)) if finite_values.size else float('nan'),
        }

        # Exact duplicate counts expose selection takeover and loss of
        # exploratory material. Pairwise distances are measured on a bounded
        # sample to keep logging cheap for larger populations.
        unique_chromosomes = np.unique(flat, axis=0).shape[0]
        unique_states = np.unique(state_rows, axis=0).shape[0]
        sample = flat[:min(flat.shape[0], 100)]
        if sample.shape[0] >= 2:
            differences = sample[:, None, :] - sample[None, :, :]
            distances = np.linalg.norm(differences, axis=2)
            upper = distances[np.triu_indices(sample.shape[0], k=1)]
            mean_chromosome_distance = float(np.mean(upper))
            min_chromosome_distance = float(np.min(upper))
        else:
            mean_chromosome_distance = float('nan')
            min_chromosome_distance = float('nan')

        low = self.mr_rate_low
        high = self.mr_rate_high
        span = high - low
        at_low = np.isclose(population, low, atol=max(span * 1e-6, 1e-7))
        at_high = np.isclose(population, high, atol=max(span * 1e-6, 1e-7))
        return {
            'n_candidates': int(population.shape[0]),
            'n_clusters': int(population.shape[1]),
            'n_mrs': int(population.shape[2]),
            'nonfinite_values': int(np.size(population) - finite_mask.sum()),
            'unique_chromosomes': int(unique_chromosomes),
            'unique_chromosome_fraction': float(unique_chromosomes / population.shape[0]),
            'unique_states': int(unique_states),
            'unique_state_fraction': float(unique_states / state_rows.shape[0]),
            'mean_chromosome_distance': mean_chromosome_distance,
            'min_chromosome_distance': min_chromosome_distance,
            'value_summary': value_summary,
            'fraction_at_low': float(np.mean(at_low)),
            'fraction_at_high': float(np.mean(at_high)),
        }

    @staticmethod
    def fitness_diagnostics(fitness: np.ndarray) -> dict:
        """Summarize finite scores and disagreement across fitness replicates."""
        values = np.asarray(fitness, dtype=np.float64)
        finite = np.isfinite(values)
        finite_values = values[finite]
        counts = finite.sum(axis=0) if values.ndim == 2 else np.array([])
        sums = np.nansum(values, axis=0) if values.ndim == 2 else np.array([])
        means = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=np.float64),
            where=counts > 0,
        ) if values.ndim == 2 else np.array([])
        valid_means = means[np.isfinite(means)]
        replicate_std = np.full(means.shape, np.nan, dtype=np.float64)
        if values.ndim == 2:
            for i, count in enumerate(counts):
                if count:
                    replicate_std[i] = np.std(values[np.isfinite(values[:, i]), i])
        valid_replicate_std = replicate_std[np.isfinite(replicate_std)]
        return {
            'shape': [int(v) for v in values.shape],
            'finite_scores': int(finite.sum()),
            'nonfinite_scores': int(values.size - finite.sum()),
            'finite_score_fraction': float(finite.mean()) if values.size else 0.0,
            'score_summary': {
                'min': float(np.min(finite_values)) if finite_values.size else float('nan'),
                'max': float(np.max(finite_values)) if finite_values.size else float('nan'),
                'mean': float(np.mean(finite_values)) if finite_values.size else float('nan'),
                'std': float(np.std(finite_values)) if finite_values.size else float('nan'),
            },
            'candidate_mean_summary': {
                'min': float(np.min(valid_means)) if valid_means.size else float('nan'),
                'max': float(np.max(valid_means)) if valid_means.size else float('nan'),
                'mean': float(np.mean(valid_means)) if valid_means.size else float('nan'),
                'std': float(np.std(valid_means)) if valid_means.size else float('nan'),
            },
            'replicate_disagreement': {
                'mean_std': float(np.mean(valid_replicate_std)) if valid_replicate_std.size else float('nan'),
                'max_std': float(np.max(valid_replicate_std)) if valid_replicate_std.size else float('nan'),
            },
        }


    def eval_candidates(self, candidates:np.ndarray):
        """
        """

        if not getattr(self, 'dags', None):
            raise RuntimeError('generate_tree() must be called before evaluating candidates')
        candidates = np.asarray(candidates, dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[0] < 1:
            raise ValueError(
                'expected candidates with shape (n_candidates, n_clusters, n_mrs), '
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
        if (candidates < self.mr_rate_low).any() or (candidates > self.mr_rate_high).any():
            raise ValueError(
                f'candidate MR states must be within '
                f'[{self.mr_rate_low}, {self.mr_rate_high}]'
            )

        selected = np.repeat(candidates, repeats=self.surrogate_repeats, axis=1)
        print(f'selected:{selected.shape}')

        n_candidates = candidates.shape[0]
        f = np.full((self.n_replicates, n_candidates), np.nan)
        errors = []
        for rep in range(self.n_replicates):
            print(f'-- rep {rep+1}/{self.n_replicates} --')

            batch_raw = sergio_dag_hill_forward_batched(
                selected, self.dags[rep], decays=self.decays)
            raw = np.asarray(batch_raw, dtype=np.float64)
            finite_candidates = np.isfinite(raw).all(axis=(1, 2))
            for i in np.flatnonzero(~finite_candidates):
                errors.append({
                    'replicate': int(rep),
                    'candidate': int(i),
                    'error': 'surrogate produced non-finite expression values',
                })

            valid_indices = np.flatnonzero(finite_candidates)
            if valid_indices.size:
                chunk_size = self.fitness_batch_size or valid_indices.size
                for start in range(0, valid_indices.size, chunk_size):
                    chunk_indices = valid_indices[start:start + chunk_size]
                    log_expression = np.log1p(
                        np.maximum(raw[chunk_indices], 0.0)
                    )
                    scores = standardized_split_half_subspace_stability_batched(
                        log_expression,
                        n_components=self.stats_n_pca_components,
                        n_structure_genes=self.stats_n_structure_genes,
                        seed=self.stats_seed + rep,
                    )
                    f[rep, chunk_indices] = scores
                    for local_i in np.flatnonzero(~np.isfinite(scores)):
                        errors.append({
                            'replicate': int(rep),
                            'candidate': int(chunk_indices[local_i]),
                            'error': 'non-finite fitness score',
                        })

        self.last_eval_diagnostics = {
            'selected_shape': [int(v) for v in selected.shape],
            'n_replicates': int(self.n_replicates),
            'surrogate_repeats': int(self.surrogate_repeats),
            'fitness_batch_size': int(self.fitness_batch_size or n_candidates),
            'errors': errors,
            'n_errors': int(len(errors)),
            'fitness': self.fitness_diagnostics(f),
        }

        return f

    def evaluate_last_crossover(self, candidates: np.ndarray) -> dict:
        """Optionally evaluate crossover children before and after mutation.

        This performs extra fitness evaluations only for the child slots
        touched by the most recent crossover. It is intended for ablation and
        mechanism studies, not normal production runs.
        """
        slots = self.last_crossover_slots
        if slots.size == 0:
            return {'n_child_slots': 0}
        if self.last_pre_mutation_candidates is None:
            raise RuntimeError('no pre-mutation crossover population is available')

        population = np.asarray(candidates, dtype=np.float32)
        post_mutation_candidates = population[slots]
        pre_mutation_fitness = self.last_pre_mutation_fitness
        if pre_mutation_fitness is None:
            pre_mutation_fitness = self.eval_candidates(self.last_pre_mutation_candidates)
        post_mutation_fitness = self.eval_candidates(post_mutation_candidates)

        def mean_scores(values):
            counts = np.isfinite(values).sum(axis=0)
            return np.divide(
                np.nansum(values, axis=0),
                counts,
                out=np.full(values.shape[1], np.nan),
                where=counts > 0,
            )

        pre_scores = mean_scores(pre_mutation_fitness)
        post_scores = mean_scores(post_mutation_fitness)
        parent_best = self.last_crossover_parent_best_scores
        pre_delta = pre_scores - parent_best
        post_delta = post_scores - parent_best

        pre_valid = np.isfinite(pre_delta)
        post_valid = np.isfinite(post_delta)

        def fraction_positive(delta, valid):
            return float(np.mean(delta[valid] > 0.0)) if valid.any() else float('nan')

        def finite_mean(values):
            return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float('nan')

        diagnostics = {
            'n_child_slots': int(slots.size),
            'child_slots': [int(i) for i in slots],
            'parent_best_scores': [float(v) for v in parent_best],
            'pre_mutation_mean_fitness': [float(v) for v in pre_scores],
            'post_mutation_mean_fitness': [float(v) for v in post_scores],
            'pre_mutation_delta_vs_parent_best': [float(v) for v in pre_delta],
            'post_mutation_delta_vs_parent_best': [float(v) for v in post_delta],
            'pre_mutation_fraction_beating_parent_best': fraction_positive(pre_delta, pre_valid),
            'post_mutation_fraction_beating_parent_best': fraction_positive(post_delta, post_valid),
            'pre_mutation_mean_delta': finite_mean(pre_delta),
            'post_mutation_mean_delta': finite_mean(post_delta),
            'pre_mutation_fitness_by_replicate': pre_mutation_fitness.tolist(),
            'post_mutation_fitness_by_replicate': post_mutation_fitness.tolist(),
        }
        self.last_operator_diagnostics.setdefault('crossover', {})[
            'fitness_comparison'
        ] = diagnostics
        return diagnostics

    def next_generation(
        self,
        candidates,
        fitness,
        p_crossover=0.5,
        p_mutation=0.02,
        mutation_scale=0.1,
        n_elites=1,
        protect_crossover_fraction=0.0,
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
        if not 0.0 <= protect_crossover_fraction <= 1.0:
            raise ValueError('protect_crossover_fraction must be in [0, 1]')
        if not 0.0 <= p_mutation <= 1.0:
            raise ValueError('p_mutation must be in [0, 1]')
        if mutation_scale < 0.0:
            raise ValueError('mutation_scale must be non-negative')
        if not 0 <= n_elites < self.n_population:
            raise ValueError('n_elites must be in [0, n_population)')
        if not np.isfinite(candidates).all():
            raise ValueError('candidate MR states must be finite')
        if (candidates < self.mr_rate_low).any() or (candidates > self.mr_rate_high).any():
            raise ValueError(
                f'candidate MR states must be within '
                f'[{self.mr_rate_low}, {self.mr_rate_high}]'
            )

        finite_counts = np.isfinite(fitness).sum(axis=0)
        scores = np.divide(
            np.nansum(fitness, axis=0),
            finite_counts,
            out=np.full(self.n_population, np.nan),
            where=finite_counts > 0,
        )
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

        selected_indices = self.selection_rng.choice(
            self.n_population,
            size=self.n_population,
            p=probabilities,
        )
        next_candidates = candidates[selected_indices].copy()
        lineage_ids = selected_indices.copy()

        elite_indices = (
            np.argsort(scores)[-n_elites:][::-1] if n_elites else np.array([], dtype=int)
        )
        if n_elites:
            next_candidates[:n_elites] = candidates[elite_indices]
            lineage_ids[:n_elites] = elite_indices

        self.last_pre_mutation_fitness = None

        # Crossover between distinct child slots. Parent rows are exchanged as
        # units, preserving each complete MR-state vector.
        before_crossover = next_candidates.copy()
        n_children = self.n_population - n_elites
        n_crossover = min(int(p_crossover * 0.5 * n_children), n_children // 2)
        child_slots = np.arange(n_elites, self.n_population)
        self.crossover_rng.shuffle(child_slots)
        side_a_idx = child_slots[:2 * n_crossover:2]
        side_b_idx = child_slots[1:2 * n_crossover:2]
        a = next_candidates[side_a_idx].copy()
        b = next_candidates[side_b_idx].copy()

        crossover = np.repeat(
            self.crossover_rng.integers(0, 2, size=a.shape[:2] + (1,)),
            repeats=a.shape[2],
            axis=2,
        )
        a_post = a.copy()
        a_post[crossover==0] = b[crossover==0]

        b_post = b.copy()
        b_post[crossover==0] = a[crossover==0]

        next_candidates[side_a_idx] = a_post
        next_candidates[side_b_idx] = b_post
        crossover_changed_entries = int(np.count_nonzero(
            next_candidates[n_elites:] != before_crossover[n_elites:]
        ))
        crossed_slots = child_slots[:2 * n_crossover].copy()
        parent_a_ids = lineage_ids[side_a_idx].copy()
        parent_b_ids = lineage_ids[side_b_idx].copy()
        parent_a_scores = scores[parent_a_ids]
        parent_b_scores = scores[parent_b_ids]
        parent_best_scores = np.empty(2 * n_crossover, dtype=np.float64)
        parent_best_scores[0::2] = np.maximum(parent_a_scores, parent_b_scores)
        parent_best_scores[1::2] = parent_best_scores[0::2]
        self.last_crossover_slots = crossed_slots
        self.last_crossover_parent_best_scores = parent_best_scores.copy()
        self.last_pre_mutation_candidates = next_candidates[crossed_slots].copy()

        protected_slots = np.array([], dtype=int)
        protected_scores = np.array([], dtype=np.float64)
        if crossed_slots.size and protect_crossover_fraction:
            # Score crossover children before mutation so strong recombinations
            # can be carried forward unchanged.
            pre_mutation_fitness = self.eval_candidates(self.last_pre_mutation_candidates)
            self.last_pre_mutation_fitness = pre_mutation_fitness
            pre_counts = np.isfinite(pre_mutation_fitness).sum(axis=0)
            pre_scores = np.divide(
                np.nansum(pre_mutation_fitness, axis=0),
                pre_counts,
                out=np.full(crossed_slots.size, np.nan),
                where=pre_counts > 0,
            )
            n_protected = min(
                int(np.ceil(protect_crossover_fraction * crossed_slots.size)),
                int(np.isfinite(pre_scores).sum()),
            )
            if n_protected:
                finite_indices = np.flatnonzero(np.isfinite(pre_scores))
                selected = finite_indices[
                    np.argsort(pre_scores[finite_indices])[-n_protected:]
                ]
                protected_slots = crossed_slots[selected]
                protected_scores = pre_scores[selected]

        # Gaussian mutation supplies new continuous MR rates; clipping keeps
        # every state valid for SERGIO and leaves elites/protected children untouched.
        mutated_entries = 0
        clipped_entries = 0
        if p_mutation and mutation_scale:
            child_candidates = next_candidates[n_elites:]
            mutation_mask = self.mutation_rng.random(child_candidates.shape) < p_mutation
            if protected_slots.size:
                mutation_mask[protected_slots - n_elites] = False
            mutation = self.mutation_rng.normal(
                0.0,
                mutation_scale * (self.mr_rate_high - self.mr_rate_low),
                size=child_candidates.shape,
            )
            mutated_entries = int(mutation_mask.sum())
            unbounded = child_candidates + mutation * mutation_mask
            clipped_entries = int(np.count_nonzero(
                (unbounded < self.mr_rate_low) | (unbounded > self.mr_rate_high)
            ))
            child_candidates[...] = np.clip(
                unbounded, self.mr_rate_low, self.mr_rate_high
            )

        log_probabilities = np.log(np.maximum(probabilities, np.finfo(float).tiny))
        protected_slot_set = {int(i) for i in protected_slots}
        crossover_lineage = []
        for pair in range(n_crossover):
            crossover_lineage.append({
                'child_slots': [int(side_a_idx[pair]), int(side_b_idx[pair])],
                'protected': [
                    int(side_a_idx[pair]) in protected_slot_set,
                    int(side_b_idx[pair]) in protected_slot_set,
                ],
                'parent_ids': [int(parent_a_ids[pair]), int(parent_b_ids[pair])],
                'parent_scores': [
                    float(parent_a_scores[pair]), float(parent_b_scores[pair])
                ],
                'swapped_rows': [
                    int(i) for i in np.flatnonzero(crossover[pair, :, 0] == 0)
                ],
                'changed_rows': {
                    'a': [
                        int(i) for i in np.flatnonzero(
                            np.any(a_post[pair] != a[pair], axis=1)
                        )
                    ],
                    'b': [
                        int(i) for i in np.flatnonzero(
                            np.any(b_post[pair] != b[pair], axis=1)
                        )
                    ],
                },
            })
        self.last_operator_diagnostics = {
            'selection': {
                'temperature': float(self.selection_temperature),
                'unique_selected_parents': int(np.unique(selected_indices).size),
                'unique_selected_parent_fraction': float(
                    np.unique(selected_indices).size / self.n_population
                ),
                'probability_entropy_normalized': float(
                    -np.sum(probabilities * log_probabilities) / np.log(self.n_population)
                ),
                'effective_parent_size': float(1.0 / np.sum(probabilities ** 2)),
            },
            'elitism': {
                'n_elites': int(n_elites),
                'elite_indices': [int(i) for i in elite_indices],
                'elite_scores': [float(scores[i]) for i in elite_indices],
            },
            'crossover': {
                'n_pairs': int(n_crossover),
                'n_child_slots': int(2 * n_crossover),
                'changed_entries': crossover_changed_entries,
                'protect_fraction': float(protect_crossover_fraction),
                'n_protected_child_slots': int(protected_slots.size),
                'protected_child_slots': [int(i) for i in protected_slots],
                'protected_child_scores': [float(v) for v in protected_scores],
                'lineage': crossover_lineage,
            },
            'mutation': {
                'entry_probability': float(p_mutation),
                'scale_fraction_of_range': float(mutation_scale),
                'mutated_entries': mutated_entries,
                'mutated_entry_fraction': float(
                    mutated_entries / max(n_children * candidates.shape[1] * candidates.shape[2], 1)
                ),
                'protected_entries': int(
                    protected_slots.size * candidates.shape[1] * candidates.shape[2]
                ),
                'clipped_entries': clipped_entries,
            },
        }

        return next_candidates


    def next_generation_de(
        self,
        candidates,
        differential_weight=0.5,
        crossover_rate=0.8,
    ):
        """Create one DE/rand/1/bin trial for every target chromosome.

        The chromosome remains a complete ``(n_clusters, n_mrs)`` matrix.
        Differential mutation is elementwise, while binomial crossover is
        performed in row blocks so that a cluster's MR profile stays intact.
        Fitness is intentionally not evaluated here; callers can batch-score
        all trials and apply one-to-one greedy replacement.
        """
        candidates = np.asarray(candidates, dtype=np.float32)
        if candidates.shape != (
            self.n_population,
            self.n_clusters,
            candidates.shape[2] if candidates.ndim == 3 else -1,
        ):
            raise ValueError(
                'expected candidates with shape '
                f'({self.n_population}, {self.n_clusters}, n_mrs), '
                f'got {candidates.shape}'
            )
        if not np.isfinite(candidates).all():
            raise ValueError('candidate MR states must be finite')
        if (candidates < self.mr_rate_low).any() or (candidates > self.mr_rate_high).any():
            raise ValueError(
                f'candidate MR states must be within '
                f'[{self.mr_rate_low}, {self.mr_rate_high}]'
            )
        if not np.isfinite(differential_weight) or differential_weight < 0.0:
            raise ValueError('differential_weight must be finite and non-negative')
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError('crossover_rate must be in [0, 1]')
        if self.n_population < 4:
            raise ValueError('DE/rand/1 requires n_population to be at least 4')

        donor_indices = np.empty((self.n_population, 3), dtype=int)
        all_indices = np.arange(self.n_population)
        for target_index in range(self.n_population):
            available = np.delete(all_indices, target_index)
            donor_indices[target_index] = self.de_rng.choice(
                available, size=3, replace=False
            )

        a = candidates[donor_indices[:, 0]]
        b = candidates[donor_indices[:, 1]]
        c = candidates[donor_indices[:, 2]]
        donor = a.astype(np.float64) + differential_weight * (
            b.astype(np.float64) - c.astype(np.float64)
        )

        # One crossover decision applies to every MR in a cluster row. The
        # forced row guarantees that every trial receives donor material,
        # including when crossover_rate is zero.
        row_mask = self.de_rng.random(
            (self.n_population, self.n_clusters)
        ) < crossover_rate
        forced_rows = self.de_rng.integers(self.n_clusters, size=self.n_population)
        row_mask[np.arange(self.n_population), forced_rows] = True
        donor_mask = row_mask[:, :, None]
        unrepaired = np.where(donor_mask, donor, candidates)

        low = self.mr_rate_low
        high = self.mr_rate_high
        span = high - low
        # Reflect repeatedly into the valid interval instead of clipping,
        # which would otherwise accumulate values at the two bounds.
        folded = np.mod(unrepaired - low, 2.0 * span)
        trials = np.where(folded <= span, low + folded, high - (folded - span))
        trials = trials.astype(np.float32)

        changed = trials != candidates
        self.last_de_diagnostics = {
            'strategy': 'rand/1/bin',
            'differential_weight': float(differential_weight),
            'crossover_rate': float(crossover_rate),
            'donor_indices': donor_indices.tolist(),
            'unique_donor_triplets': int(np.unique(donor_indices, axis=0).shape[0]),
            'forced_rows': [int(v) for v in forced_rows],
            'changed_entries': int(np.count_nonzero(changed)),
            'changed_entry_fraction': float(np.mean(changed)),
            'changed_rows': int(np.count_nonzero(np.any(changed, axis=2))),
            'out_of_bounds_entries': int(np.count_nonzero(
                (unrepaired < low) | (unrepaired > high)
            )),
            'repaired_entries': int(np.count_nonzero(
                (unrepaired < low) | (unrepaired > high)
            )),
        }
        return trials


def evo_loop(
    config_fn,
    max_generations=20,
    p_crossover=0.5,
    p_mutation=0.02,
    mutation_scale=0.1,
    n_elites=4,
    diagnose_crossover=False,
    protect_crossover_fraction=0.0,
    **ga_kwargs,
):
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
        population_before = ga.population_diagnostics(candidates)
        f = ga.eval_candidates(candidates)
        finite_counts = np.isfinite(f).sum(axis=0)
        mean_fitness = np.divide(
            np.nansum(f, axis=0),
            finite_counts,
            out=np.full(f.shape[1], np.nan),
            where=finite_counts > 0,
        )
        valid = np.isfinite(mean_fitness)
        if not valid.any():
            raise ValueError(f'generation {i} produced no finite fitness values')
        generation_best_idx = int(np.nanargmax(mean_fitness))
        generation_best = float(mean_fitness[generation_best_idx])
        if generation_best > best_fitness:
            best_fitness = generation_best
            best_candidate = candidates[generation_best_idx].copy()

        finite_fitness = f[np.isfinite(f)]
        evaluation = ga.last_eval_diagnostics
        qs = np.quantile(
            finite_fitness,
            q=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
        )
        evo_log.append({
            'generation': i,
            'mean_fitness': float(np.mean(finite_fitness)),
            'variance_fitness': float(np.var(finite_fitness)),
            'generation_best': generation_best,
            'generation_best_index': generation_best_idx,
            'best_so_far': best_fitness,
            'quantiles': [float(q) for q in qs],
            'population_before': population_before,
            'evaluation': evaluation,
        })
        next_gen = ga.next_generation(
            candidates,
            fitness=f,
            p_crossover=p_crossover,
            protect_crossover_fraction=protect_crossover_fraction,
            p_mutation=p_mutation,
            mutation_scale=mutation_scale,
            n_elites=n_elites,
        )
        if diagnose_crossover:
            ga.evaluate_last_crossover(next_gen)
        evo_log[-1]['operators'] = ga.last_operator_diagnostics
        evo_log[-1]['population_after'] = ga.population_diagnostics(next_gen)
        candidates = next_gen.copy()
        operators = ga.last_operator_diagnostics
        crossover_log = operators['crossover']
        print(
            f'{np.mean(finite_fitness):.4g} q99={qs[6]:.4g} '
            f'best={best_fitness:.4g} '
            f'finite={evaluation["fitness"]["finite_score_fraction"]:.3f} '
            f'errors={evaluation["n_errors"]} '
            f'unique={population_before["unique_chromosome_fraction"]:.3f} '
            f'-> {evo_log[-1]["population_after"]["unique_chromosome_fraction"]:.3f} '
            f'parents={operators["selection"]["unique_selected_parent_fraction"]:.3f} '
            f'cross={crossover_log["n_pairs"]} '
            f'protected={crossover_log["n_protected_child_slots"]} '
            f'changed={crossover_log["changed_entries"]} '
            f'mut={operators["mutation"]["mutated_entry_fraction"]:.3f} '
            f'clip={operators["mutation"]["clipped_entries"]}'
        )
        comparison = crossover_log.get('fitness_comparison')
        if comparison:
            print(
                f'  crossover fitness: '
                f'pre_delta={comparison["pre_mutation_mean_delta"]:.4g} '
                f'post_delta={comparison["post_mutation_mean_delta"]:.4g} '
                f'pre_win={comparison["pre_mutation_fraction_beating_parent_best"]:.3f} '
                f'post_win={comparison["post_mutation_fraction_beating_parent_best"]:.3f}'
            )

    return {
        'best_candidate': best_candidate,
        'best_fitness': best_fitness,
        'population': candidates,
        'evolution_log': evo_log,
    }


def de_loop(
    config_fn,
    max_generations=20,
    differential_weight=0.5,
    crossover_rate=0.8,
    **de_kwargs,
):
    """Optimize MR-state chromosomes with vectorized differential evolution.

    ``max_generations`` DE generations each evaluate one trial population.
    Parent fitness is cached between generations, and target/trial pairs use
    greedy one-to-one replacement, providing implicit elitism without a
    separate elite count. The existing GA loop is intentionally unchanged.
    """
    if max_generations < 1:
        raise ValueError('max_generations must be positive')

    de = MRStateGAOptimizer(config_fn, **de_kwargs)
    de.generate_tree()
    candidates = de.initialize_population()
    print(candidates.shape)

    # Evaluate the initial population once. Thereafter this array is the
    # exact parent-fitness cache aligned with the candidate population.
    fitness = de.eval_candidates(candidates)
    finite_counts = np.isfinite(fitness).sum(axis=0)
    parent_scores = np.divide(
        np.nansum(fitness, axis=0),
        finite_counts,
        out=np.full(de.n_population, np.nan),
        where=finite_counts > 0,
    )
    if not np.isfinite(parent_scores).any():
        raise ValueError('initial population produced no finite fitness values')

    valid_scores = np.isfinite(parent_scores)
    best_index = int(np.flatnonzero(valid_scores)[np.argmax(parent_scores[valid_scores])])
    best_fitness = float(parent_scores[best_index])
    best_candidate = candidates[best_index].copy()
    evo_log = []

    for generation in range(max_generations):
        print(f'=== DE gen {generation + 1}/{max_generations} ===')
        population_before = de.population_diagnostics(candidates)
        parent_scores_before = parent_scores.copy()

        trials = de.next_generation_de(
            candidates,
            differential_weight=differential_weight,
            crossover_rate=crossover_rate,
        )
        trial_fitness = de.eval_candidates(trials)
        trial_counts = np.isfinite(trial_fitness).sum(axis=0)
        trial_scores = np.divide(
            np.nansum(trial_fitness, axis=0),
            trial_counts,
            out=np.full(de.n_population, np.nan),
            where=trial_counts > 0,
        )

        accepted = np.isfinite(trial_scores) & (
            ~np.isfinite(parent_scores) | (trial_scores > parent_scores)
        )
        candidates[accepted] = trials[accepted]
        fitness[:, accepted] = trial_fitness[:, accepted]
        parent_scores[accepted] = trial_scores[accepted]

        finite_scores = np.isfinite(parent_scores)
        if not finite_scores.any():
            raise ValueError(
                f'DE generation {generation} produced no finite population fitness values'
            )
        generation_best_index = int(
            np.flatnonzero(finite_scores)[np.argmax(parent_scores[finite_scores])]
        )
        generation_best = float(parent_scores[generation_best_index])
        if generation_best > best_fitness:
            best_fitness = generation_best
            best_candidate = candidates[generation_best_index].copy()

        trial_finite = trial_scores[np.isfinite(trial_scores)]
        accepted_delta = trial_scores[accepted] - parent_scores_before[accepted]
        accepted_delta = accepted_delta[np.isfinite(accepted_delta)]
        operators = dict(de.last_de_diagnostics)
        operators.update({
            'accepted_trials': int(np.count_nonzero(accepted)),
            'rejected_trials': int(de.n_population - np.count_nonzero(accepted)),
            'acceptance_fraction': float(np.mean(accepted)),
            'nonfinite_trials': int(np.count_nonzero(~np.isfinite(trial_scores))),
            'trial_best': float(np.max(trial_finite)) if trial_finite.size else float('nan'),
            'mean_accepted_delta': float(np.mean(accepted_delta)) if accepted_delta.size else float('nan'),
            'max_accepted_delta': float(np.max(accepted_delta)) if accepted_delta.size else float('nan'),
        })
        population_after = de.population_diagnostics(candidates)
        evaluation = de.last_eval_diagnostics
        population_finite = parent_scores[np.isfinite(parent_scores)]
        evo_log.append({
            'generation': generation,
            'mean_fitness': float(np.mean(population_finite)),
            'variance_fitness': float(np.var(population_finite)),
            'generation_best': generation_best,
            'generation_best_index': generation_best_index,
            'best_so_far': best_fitness,
            'parent_best': float(np.nanmax(parent_scores_before)),
            'trial_best': operators['trial_best'],
            'accepted_trials': int(np.count_nonzero(accepted)),
            'quantiles': [
                float(q) for q in np.quantile(
                    trial_finite,
                    q=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
                )
            ] if trial_finite.size else [float('nan')] * 7,
            'population_before': population_before,
            'population_after': population_after,
            'evaluation': evaluation,
            'operators': operators,
        })

        print(
            f'{np.mean(population_finite):.4g} '
            f'trial_best={operators["trial_best"]:.4g} '
            f'best={best_fitness:.4g} '
            f'accepted={operators["accepted_trials"]}/{de.n_population} '
            f'finite={evaluation["fitness"]["finite_score_fraction"]:.3f} '
            f'errors={evaluation["n_errors"]} '
            f'unique={population_before["unique_chromosome_fraction"]:.3f} '
            f'-> {population_after["unique_chromosome_fraction"]:.3f} '
            f'changed={operators["changed_entries"]} '
            f'out_of_bounds={operators["out_of_bounds_entries"]}'
        )

    return {
        'best_candidate': best_candidate,
        'best_fitness': best_fitness,
        'population': candidates,
        'evolution_log': evo_log,
    }


def exp_test(config_fn:str):
    with_cross = evo_loop(
         config_fn,
         max_generations=110,
         n_population=200,
         seed=0,
         p_crossover=0.25,
         diagnose_crossover=True
     )
    return with_cross

def exp_control(config_fn:str):
    without_cross = evo_loop(
         config_fn,
         max_generations=110,
         n_population=200,
         seed=0,
         p_crossover=0.0,
     )
    return without_cross



def exp_full_xover(config_fn:str):
    with_cross = evo_loop(
         config_fn,
         max_generations=110,
         n_population=200,
         seed=0,
         p_crossover=1.0,
         diagnose_crossover=True
     )
    return with_cross


def exp_protected_xover(config_fn:str):
    with_cross = evo_loop(
         config_fn,
         max_generations=110,
         n_population=200,
         seed=0,
         p_crossover=0.5,
         protect_crossover_fraction=0.1,
         diagnose_crossover=True
     )
    return with_cross

def exp_unprotected_xover05(config_fn:str):
    with_cross = evo_loop(
         config_fn,
         max_generations=110,
         n_population=200,
         seed=0,
         p_crossover=0.5,
         protect_crossover_fraction=0.0,
         diagnose_crossover=True
     )
    return with_cross


def exp_de_F05_CR08(config_fn: str, seed:int):
    return de_loop(
        config_fn,
        max_generations=110,
        n_population=200,
        seed=seed,
        differential_weight=0.5,
        crossover_rate=0.8,
    )


def exp_de_F03_CR05(config_fn: str):
    return de_loop(
        config_fn,
        max_generations=60,
        n_population=200,
        seed=0,
        differential_weight=0.3,
        crossover_rate=0.5,
    )


def exp_de_F03_CR08(config_fn: str):
    return de_loop(
        config_fn,
        max_generations=60,
        n_population=200,
        seed=0,
        differential_weight=0.3,
        crossover_rate=0.8,
    )

def exp_de_F05_CR05(config_fn: str):
    return de_loop(
        config_fn,
        max_generations=60,
        n_population=200,
        seed=0,
        differential_weight=0.5,
        crossover_rate=0.5,
    )

if __name__=='__main__':
    config_fn = '/content/Drive/MyDrive/Colab Notebooks/tuned_synthetic_config.20260821.00.json'

    import sys
    import pickle

    seed=0

    if len(sys.argv)>2:
        if sys.argv[2].startswith('seed='):
            seed = int(sys.argv[2].split('=')[1])
        else:
            assert(False)

    print(f'var={sys.argv[1]} seed={seed}')

    if sys.argv[1] == 'test':
        ret = exp_test(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.test.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'protected_xover':
        ret = exp_protected_xover(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.xover05_protected10.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'unprotected_xover05':
        ret = exp_unprotected_xover05(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.xover05_protected0.pickle', 'wb') as f:
            pickle.dump(ret, f)


    elif sys.argv[1] == 'full_xover':
        ret = exp_full_xover(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.full_xover.pickle', 'wb') as f:
            pickle.dump(ret, f)


    elif sys.argv[1] == 'control':
        ret = exp_control(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.control.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'de_F05_CR08':
        ret = exp_de_F05_CR08(config_fn, seed=seed)
        with open(f'/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.de_F05_CR08.seed{seed}.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'de_F03_CR05':
        ret = exp_de_F03_CR05(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.de_F03_CR05.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'de_F03_CR08':
        ret = exp_de_F03_CR08(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.de_F03_CR08.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'de_F05_CR05':
        ret = exp_de_F05_CR05(config_fn)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.de_F05_CR05.pickle', 'wb') as f:
            pickle.dump(ret, f)


    else:
        assert(False)

    sys.exit(0)
