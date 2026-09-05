import json
import hashlib
import math
import os
import pickle
from pathlib import Path

import numpy as np
from synthetic_data import (
    add_pca_derived_stats,
    generate_sergio_grn_from_reference, load_sergio_dag, sample_sergio_mr_states,
    sergio_dag_hill_forward_batched,
    _sample_cluster_sizes,
)


# OUTSTANDING ISSUES / LIMITATIONS
# - The inner objective is a stochastic Hill surrogate, not full SERGIO. Its
#   cell-level MR perturbation model is an approximation and needs calibration
#   against independent full-SERGIO validation runs.
# - Each run still uses one fixed GRN. Results do not establish robustness to
#   GRN topology or parameter variation.
# - MR columns are positional. Optimizer artifacts must be checked against the
#   exact ordered MR list and GRN fingerprint before reuse.
# - Cluster rows are exchangeable for the expression objective, but the search
#   operators still carry row-position symmetry that can reduce efficiency.
# - DE uses a target-free intrinsic program objective: cluster-level log-Hill
#   response entropy/participation, a PC1-dominance penalty, and split-half
#   stability. These are scale-free structural proxies; program amplitude,
#   target-like expression statistics, modularity, and technical/noise fit are
#   not guaranteed and require separate validation or outer tuning.
# - GA retains the older target-aware PCA surrogate objective. Reference-PCA
#   matching is intentionally excluded from DE because it can make MR states
#   compensate for unresolved GRN/SERGIO or technical-stage behavior.
# - DE/rand/1 mutation is fitness-independent, so DE replacement uses only
#   rotating score replicates; it does not perform a diagnostic-only train
#   evaluation. The final all-replicate score is validation, not a permanently
#   untouched holdout.
# - Generation checkpoints are opt-in and should be enabled for long runs.


class MRStateGAOptimizer:
    """
                                          choice                          offspring                              fitness
    The process is (n_candidates,n_mrs) ---------->  (n_clusters,n_mrs) -------------> (n_clusters*10,n_genes) -----------> (1,)
    The unit of selection is (n_clusters, n_mrs)
    n_replicates: independent stochastic Hill-surrogate realizations used for
                    rotating train/score evaluation in GA and score evaluation
                    in DE. Set ``replicate_grns`` in the config to request
                    separate GRNs instead.
    
    """
    def __init__(
        self,
        base_cfg: str,
        n_population: int = 200,
        n_replicates: int = 4,
        n_clusters: int = 15,
        seed: int = 0,
        objective_mode: str | None = None,
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
        self.objective_mode = str(
            objective_mode
            if objective_mode is not None
            else self.cfg.get('surrogate_objective', 'target_pca')
        )
        if self.objective_mode not in ('target_pca', 'intrinsic'):
            raise ValueError(
                "objective_mode must be 'target_pca' or 'intrinsic'"
            )
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
        self.cluster_conc = float(setting('cluster_conc', 10.0))
        self.min_cells_per_cluster = int(
            base.get('min_cells_per_cluster', self.cfg.get('min_cells_per_cluster', 1))
        )
        if self.cluster_conc <= 0.0 or self.min_cells_per_cluster < 1:
            raise ValueError('cluster_conc must be positive and min_cells_per_cluster must be positive')
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
        self.surrogate_n_cells = int(
            self.cfg.get(
                'surrogate_n_cells',
                base.get('n_cells', self.cfg.get('n_cells', self.n_clusters * 20)),
            )
        )
        if self.surrogate_n_cells < max(4, self.n_clusters):
            raise ValueError(
                'surrogate_n_cells must be at least max(4, n_clusters)'
            )
        self.surrogate_mr_jitter_std = float(
            self.cfg.get('surrogate_mr_jitter_std', 0.05)
        )
        if not math.isfinite(self.surrogate_mr_jitter_std) or self.surrogate_mr_jitter_std < 0.0:
            raise ValueError('surrogate_mr_jitter_std must be finite and non-negative')
        self.surrogate_seed_base = int(self.cfg.get('surrogate_seed_base', 100_000))
        self.surrogate_n_components = int(
            self.cfg.get('surrogate_n_pca_components', self.stats_n_pca_components)
        )
        if self.surrogate_n_components < 1:
            raise ValueError('surrogate_n_pca_components must be positive')
        self.rank_tolerance = float(self.cfg.get('surrogate_rank_tolerance', 1e-6))
        if not math.isfinite(self.rank_tolerance) or self.rank_tolerance <= 0.0:
            raise ValueError('surrogate_rank_tolerance must be finite and positive')
        self.validation_risk_weight = float(
            self.cfg.get('surrogate_validation_risk_weight', 0.25)
        )
        if not math.isfinite(self.validation_risk_weight) or self.validation_risk_weight < 0.0:
            raise ValueError(
                'surrogate_validation_risk_weight must be finite and non-negative'
            )
        self.score_replicates_per_generation = int(
            self.cfg.get('surrogate_score_replicates_per_generation', 2)
        )
        if self.score_replicates_per_generation < 1:
            raise ValueError(
                'surrogate_score_replicates_per_generation must be positive'
            )
        if self.score_replicates_per_generation > self.n_replicates:
            raise ValueError(
                'n_replicates must be at least surrogate_score_replicates_per_generation'
            )
        if (
            self.objective_mode == 'target_pca'
            and self.n_replicates <= self.score_replicates_per_generation
        ):
            raise ValueError(
                'target_pca mode requires n_replicates to exceed '
                'surrogate_score_replicates_per_generation'
            )
        self.canonicalize_cluster_rows = bool(
            self.cfg.get('canonicalize_cluster_rows', True)
        )
        self.target_objective_weights = {
            'shape': float(self.cfg.get('surrogate_shape_weight', 1.0)),
            'tail': float(self.cfg.get('surrogate_tail_weight', 0.5)),
            'stability': float(self.cfg.get('surrogate_stability_weight', 1.0)),
            'rank': float(self.cfg.get('surrogate_rank_weight', 0.5)),
            'degeneracy': float(self.cfg.get('surrogate_degeneracy_weight', 2.0)),
        }
        self.intrinsic_objective_weights = {
            'entropy': float(self.cfg.get('surrogate_program_entropy_weight', 1.0)),
            'participation': float(
                self.cfg.get('surrogate_program_participation_weight', 1.0)
            ),
            'stability': float(
                self.cfg.get('surrogate_program_stability_weight', 1.0)
            ),
            'pc1': float(self.cfg.get('surrogate_program_pc1_weight', 2.0)),
        }
        self.program_max_pc1_fraction = float(
            self.cfg.get('surrogate_program_max_pc1_fraction', 0.25)
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                list(self.target_objective_weights.values())
                + list(self.intrinsic_objective_weights.values())
            )
        ):
            raise ValueError('surrogate objective weights must be finite and non-negative')
        if not 0.0 < self.program_max_pc1_fraction < 1.0:
            raise ValueError(
                'surrogate_program_max_pc1_fraction must be in (0, 1)'
            )
        self.selection_temperature = float(self.cfg.get('selection_temperature', 0.05))
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        self.replicate_grns = bool(self.cfg.get('replicate_grns', False))
        self.fitness_batch_size = self.cfg.get('fitness_batch_size', 16)
        if self.fitness_batch_size is not None:
            self.fitness_batch_size = int(self.fitness_batch_size)
            if self.fitness_batch_size < 1:
                raise ValueError('fitness_batch_size must be positive')
        target_path = base.get('target_stats_path', self.cfg.get('target_stats_path'))
        self.target_stats = None
        if self.objective_mode == 'target_pca' and target_path:
            with open(target_path, 'rb') as f:
                self.target_stats = pickle.load(f)
            add_pca_derived_stats(self.target_stats)
        target_key = (
            self.cfg.get(
                'surrogate_target_pca_key',
                'pca_size_normalized_standardized_explained_variance_ratio',
            )
            if self.objective_mode == 'target_pca'
            else None
        )
        target_ratio = None if self.target_stats is None else self.target_stats.get(target_key)
        if target_ratio is not None:
            target_ratio = np.asarray(target_ratio, dtype=np.float64)
            target_ratio = target_ratio[np.isfinite(target_ratio)]
            if target_ratio.size < 3:
                raise ValueError(
                    f'target PCA key {target_key!r} must contain at least 3 finite values'
                )
        self.target_pca_ratio = target_ratio
        self.target_pca_key = target_key
        self.target_tail_participation = None
        if target_ratio is not None:
            tail = target_ratio[1:]
            if tail.size >= 2 and np.any(tail > 0.0):
                self.target_tail_participation = float(
                    (tail.sum() ** 2) / np.sum(tail ** 2)
                )
        self._surrogate_replicates = []
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
        # surrogate replicate evaluations. Separate DAGs are opt-in and are
        # accepted only when their ordered MR universes match exactly.
        dags = dags * self.n_replicates if not self.replicate_grns else dags
                    
        self.grns = grns
        self.dags = dags
        self.grn_metadata = []
        for seed, temp_path, mr_ids, gene_id_to_symbol in grns:
            digest = hashlib.sha256()
            with open(temp_path, 'rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    digest.update(block)
            self.grn_metadata.append({
                'seed': int(seed),
                'sha256': digest.hexdigest(),
                'mr_ids': list(mr_ids),
                'gene_id_to_symbol': gene_id_to_symbol,
            })
        self._build_surrogate_replicates()

    def _build_surrogate_replicates(self) -> None:
        """Precompute common-random-number stochastic surrogate replicates.

        A replicate is a fixed cell-cluster assignment, cell-level MR jitter
        field, and PCA split.  Every candidate evaluated on that replicate
        sees exactly the same random realization; different replicate IDs are
        independent.  This keeps comparisons paired while preventing the
        optimizer from treating repeated copies of one deterministic cluster
        response as 600 independent cells.
        """
        if not getattr(self, 'grns', None):
            raise RuntimeError('generate_tree() must create a GRN first')
        base = self.cfg.get('_meta', self.cfg)
        n_mrs = len(self.grns[0][2])
        count_rng = np.random.default_rng(self.surrogate_seed_base)
        counts = _sample_cluster_sizes(
            self.surrogate_n_cells,
            self.n_clusters,
            self.cluster_conc,
            self.min_cells_per_cluster,
            count_rng,
        ).astype(np.int64)

        gene_rng = np.random.default_rng(self.stats_seed)
        n_genes = int(base.get('n_genes', self.cfg.get('n_genes', 800)))
        n_structure_genes = self.stats_n_structure_genes
        if n_structure_genes is None or n_genes <= int(n_structure_genes):
            gene_idx = np.arange(n_genes, dtype=np.int64)
        else:
            gene_idx = np.sort(gene_rng.choice(
                n_genes, size=int(n_structure_genes), replace=False
            ))
        half = self.surrogate_n_cells // 2
        k = min(
            self.surrogate_n_components,
            half - 1,
            int(gene_idx.size),
        )
        if k < 1:
            raise ValueError('surrogate dimensions do not permit PCA components')
        self.surrogate_gene_idx = gene_idx
        self.surrogate_k = int(k)
        self._surrogate_replicates = []
        span = self.mr_rate_high - self.mr_rate_low
        for replicate in range(self.n_replicates):
            rng = np.random.default_rng(self.surrogate_seed_base + replicate)
            labels = np.repeat(np.arange(self.n_clusters), counts)
            rng.shuffle(labels)
            order = rng.permutation(self.surrogate_n_cells)[:2 * half]
            self._surrogate_replicates.append({
                'replicate': int(replicate),
                'seed': int(self.surrogate_seed_base + replicate),
                'cluster_labels': labels.astype(np.int64),
                'mr_noise': rng.normal(0.0, 1.0, size=(self.surrogate_n_cells, n_mrs)),
                'split_order': order.astype(np.int64),
                'mr_jitter_scale': float(self.surrogate_mr_jitter_std * span),
            })

    def surrogate_metadata(self) -> dict:
        """Return the stochastic surrogate contract stored with results."""
        objective_weights = (
            self.intrinsic_objective_weights
            if self.objective_mode == 'intrinsic'
            else self.target_objective_weights
        )
        return {
            'objective_mode': self.objective_mode,
            'n_replicates': int(self.n_replicates),
            'seed_base': int(self.surrogate_seed_base),
            'score_replicates_per_generation': int(self.score_replicates_per_generation),
            'n_cells': int(self.surrogate_n_cells),
            'cluster_conc': float(self.cluster_conc),
            'min_cells_per_cluster': int(self.min_cells_per_cluster),
            'n_components': int(self.surrogate_k),
            'n_structure_genes': int(self.surrogate_gene_idx.size),
            'mr_jitter_std': float(self.surrogate_mr_jitter_std),
            'objective_weights': dict(objective_weights),
            'program_max_pc1_fraction': float(self.program_max_pc1_fraction),
            'target_pca_key': self.target_pca_key,
            'target_pca_length': (
                int(self.target_pca_ratio.size)
                if self.target_pca_ratio is not None else None
            ),
        }

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
        population = np.stack(candidates)
        return self.canonicalize_rows(population)

    def canonicalize_rows(self, candidates: np.ndarray) -> np.ndarray:
        """Canonicalize exchangeable cluster rows for permutation-invariant scoring."""
        values = np.asarray(candidates, dtype=np.float32)
        if not self.canonicalize_cluster_rows:
            return values
        output = values.copy()
        for i in range(output.shape[0]):
            keys = tuple(output[i, :, column] for column in range(output.shape[2] - 1, -1, -1))
            output[i] = output[i][np.lexsort(keys)]
        return output

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


    def _evaluate_surrogate_batch(
        self, candidates: np.ndarray, replicate_id: int
    ) -> tuple[np.ndarray, dict]:
        """Evaluate one candidate batch on one stochastic Hill replicate."""
        replicate = self._surrogate_replicates[replicate_id]
        labels = replicate['cluster_labels']
        cell_states = candidates[:, labels, :].astype(np.float64, copy=False)
        if replicate['mr_jitter_scale']:
            cell_states = cell_states + replicate['mr_jitter_scale'] * replicate['mr_noise'][None, :, :]
            cell_states = np.clip(
                cell_states, self.mr_rate_low, self.mr_rate_high
            )

        raw = np.asarray(
            sergio_dag_hill_forward_batched(
                cell_states, self.dags[replicate_id], decays=self.decays
            ),
            dtype=np.float64,
        )
        if not np.isfinite(raw).all():
            raise ValueError('stochastic Hill surrogate produced non-finite values')
        raw = np.maximum(raw, 0.0)

        program_metrics = {}
        if self.objective_mode == 'intrinsic':
            # Measure the programs induced by the cluster states directly.
            # This deliberately does not use reference statistics: a
            # high-entropy, high-participation response that survives
            # replicate perturbations is the intrinsic DE objective, while
            # target matching belongs downstream.
            raw_log = np.log1p(raw)
            program_expression = raw_log[:, :, self.surrogate_gene_idx]
            cluster_programs = np.stack([
                program_expression[:, labels == cluster, :].mean(axis=1)
                for cluster in range(self.n_clusters)
            ], axis=1)
            cluster_programs -= cluster_programs.mean(axis=1, keepdims=True)
            _, program_singular, _ = np.linalg.svd(
                cluster_programs, full_matrices=False, compute_uv=True
            )
            program_k = min(
                self.n_clusters - 1,
                int(self.surrogate_gene_idx.size),
                program_singular.shape[1],
            )
            program_variance = np.square(program_singular[:, :program_k])
            program_total = program_variance.sum(axis=1)
            program_ratios = np.divide(
                program_variance,
                np.maximum(program_total[:, None], 1e-12),
            )
            positive_program_ratios = program_ratios > 0.0
            program_entropy = np.divide(
                -np.sum(
                    np.where(
                        positive_program_ratios,
                        program_ratios * np.log(np.maximum(program_ratios, 1e-12)),
                        0.0,
                    ),
                    axis=1,
                ),
                np.log(program_k),
                out=np.zeros(candidates.shape[0], dtype=np.float64),
                where=program_k > 1,
            )
            program_participation = np.divide(
                1.0,
                np.sum(np.square(program_ratios), axis=1),
                out=np.zeros(candidates.shape[0], dtype=np.float64),
                where=program_total > 0.0,
            )
            program_participation_normalized = program_participation / max(program_k, 1)
            program_pc1_fraction = program_ratios[:, 0] if program_k else np.zeros(
                candidates.shape[0], dtype=np.float64
            )
            program_trace = program_total / max(program_k, 1)
            program_metrics = {
                'program_spectral_entropy': program_entropy,
                'program_participation_ratio': program_participation,
                'program_participation_ratio_normalized': program_participation_normalized,
                'program_pc1_fraction': program_pc1_fraction,
                'program_trace': program_trace,
            }

        library_size = np.maximum(raw.sum(axis=2), 1e-8)
        median_library_size = np.median(library_size, axis=1, keepdims=True)
        size_factor = median_library_size / library_size
        normalized = np.log1p(raw * size_factor[:, :, None])
        structured = normalized[:, :, self.surrogate_gene_idx]
        structured -= structured.mean(axis=1, keepdims=True)
        gene_std = np.clip(structured.std(axis=1, keepdims=True), 1e-6, None)
        standardized = structured / gene_std

        _, singular, _ = np.linalg.svd(
            standardized, full_matrices=False, compute_uv=True
        )
        variance = np.square(singular)
        total = variance.sum(axis=1)
        ratios = variance[:, :self.surrogate_k] / np.maximum(total[:, None], 1e-12)
        rank = np.sum(
            singular > singular[:, :1] * self.rank_tolerance,
            axis=1,
        ).astype(np.float64)

        order = replicate['split_order']
        half = self.surrogate_n_cells // 2
        left = standardized[:, order[:half], :].copy()
        right = standardized[:, order[half:2 * half], :].copy()
        left -= left.mean(axis=1, keepdims=True)
        right -= right.mean(axis=1, keepdims=True)
        _, left_singular, left_basis = np.linalg.svd(
            left, full_matrices=False, compute_uv=True
        )
        _, right_singular, right_basis = np.linalg.svd(
            right, full_matrices=False, compute_uv=True
        )
        left_rank = np.sum(
            left_singular > left_singular[:, :1] * self.rank_tolerance,
            axis=1,
        )
        right_rank = np.sum(
            right_singular > right_singular[:, :1] * self.rank_tolerance,
            axis=1,
        )
        stability = np.zeros(candidates.shape[0], dtype=np.float64)
        for i in range(candidates.shape[0]):
            local_k = min(
                self.surrogate_k,
                int(rank[i]),
                int(left_rank[i]),
                int(right_rank[i]),
            )
            if local_k:
                left_subspace = left_basis[i, :local_k, :].T
                right_subspace = right_basis[i, :local_k, :].T
                cosines = np.linalg.svd(
                    left_subspace.T @ right_subspace,
                    compute_uv=False,
                )
                stability[i] = float(np.mean(np.square(cosines)))

        tail = ratios[:, 1:]
        tail_pr = np.divide(
            np.square(tail.sum(axis=1)),
            np.square(tail).sum(axis=1),
            out=np.zeros(candidates.shape[0], dtype=np.float64),
            where=np.square(tail).sum(axis=1) > 0.0,
        )
        shape_loss = np.zeros(candidates.shape[0], dtype=np.float64)
        tail_loss = np.zeros(candidates.shape[0], dtype=np.float64)
        rank_loss = np.zeros(candidates.shape[0], dtype=np.float64)
        if self.objective_mode == 'target_pca' and self.target_pca_ratio is not None:
            n_shape = min(
                8,
                self.surrogate_k - 1,
                self.target_pca_ratio.size - 1,
            )
            target_shape = self.target_pca_ratio[1:1 + n_shape]
            if n_shape:
                denominator = np.maximum(
                    np.abs(target_shape),
                    max(0.02, 0.05 * float(np.max(np.abs(target_shape)))),
                )
                shape_loss = np.mean(
                    np.square((ratios[:, 1:1 + n_shape] - target_shape) / denominator),
                    axis=1,
                )
            if self.target_tail_participation is not None:
                rank_loss = np.square(
                    np.log((tail_pr + 1e-6) /
                           (self.target_tail_participation + 1e-6))
                )
            n_tail = min(ratios.shape[1] - 1, self.target_pca_ratio.size - 1)
            target_tail = self.target_pca_ratio[1:1 + n_tail]
            if n_tail:
                tail_denominator = max(
                    0.02,
                    0.05 * float(np.max(np.abs(target_tail))),
                )
                tail_loss = np.mean(
                    np.square((ratios[:, 1:1 + n_tail] - target_tail) /
                              (np.abs(target_tail) + tail_denominator)),
                    axis=1,
                )

        null_stability = self.surrogate_k / max(self.surrogate_gene_idx.size, 1)
        stability_score = np.clip(
            (stability - null_stability) / max(1.0 - null_stability, 1e-12),
            0.0,
            1.0,
        )
        degeneracy = np.square(
            np.maximum(0.0, 1.0 - rank / max(self.surrogate_k, 1))
        )
        if self.objective_mode == 'intrinsic':
            weights = self.intrinsic_objective_weights
            pc1_excess = np.maximum(
                0.0,
                program_pc1_fraction - self.program_max_pc1_fraction,
            )
            pc1_loss = np.square(
                pc1_excess / max(1.0 - self.program_max_pc1_fraction, 1e-12)
            )
            loss = (
                weights['entropy'] * (1.0 - program_entropy)
                + weights['participation'] * (
                    1.0 - program_participation_normalized
                )
                + weights['stability'] * (1.0 - stability_score)
                + weights['pc1'] * pc1_loss
            )
        else:
            weights = self.target_objective_weights
            loss = (
                weights['shape'] * shape_loss
                + weights['tail'] * tail_loss
                + weights['rank'] * rank_loss
                + weights['stability'] * (1.0 - stability_score)
                + weights['degeneracy'] * degeneracy
            )
        metrics = {
            'shape_loss': shape_loss,
            'tail_loss': tail_loss,
            'rank_loss': rank_loss,
            'tail_participation_ratio': tail_pr,
            'effective_rank': rank,
            'loading_stability': stability,
            'stability_score': stability_score,
            'degeneracy': degeneracy,
            'loss': loss,
        }
        metrics.update(program_metrics)
        return -loss, metrics

    @staticmethod
    def aggregate_fitness(fitness: np.ndarray, risk_weight: float = 0.0) -> np.ndarray:
        """Aggregate replicate fitness, penalizing instability when requested."""
        values = np.asarray(fitness, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f'expected fitness with shape (n_replicates, n_candidates), got {values.shape}')
        counts = np.isfinite(values).sum(axis=0)
        means = np.divide(
            np.nansum(values, axis=0),
            counts,
            out=np.full(values.shape[1], np.nan),
            where=counts > 0,
        )
        if values.shape[0] == 1 or risk_weight == 0.0:
            return means
        std = np.nanstd(values, axis=0)
        return means - float(risk_weight) * std

    def generation_replicates(self, generation: int) -> tuple[list[int], list[int]]:
        """Return rotating train and scoring replicate IDs for one generation."""
        if self.objective_mode == 'target_pca' and self.n_replicates < 2:
            raise ValueError('at least two surrogate replicates are required for train/score separation')
        train = (
            [] if self.objective_mode == 'intrinsic'
            else [int(generation % self.n_replicates)]
        )
        score = [
            int((generation + 1 + offset) % self.n_replicates)
            for offset in range(self.score_replicates_per_generation)
        ]
        return train, score

    def eval_candidates(
        self, candidates: np.ndarray, replicate_ids=None
    ):
        """Evaluate candidates on explicit stochastic surrogate replicates."""
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
        if replicate_ids is None:
            replicate_ids = list(range(self.n_replicates))
        replicate_ids = [int(rep) for rep in replicate_ids]
        if not replicate_ids or any(
            rep < 0 or rep >= self.n_replicates for rep in replicate_ids
        ):
            raise ValueError(f'invalid surrogate replicate IDs: {replicate_ids!r}')

        n_candidates = candidates.shape[0]
        fitness = np.full((len(replicate_ids), n_candidates), np.nan)
        replicate_diagnostics = []
        errors = []
        chunk_size = self.fitness_batch_size or n_candidates
        for output_rep, replicate_id in enumerate(replicate_ids):
            print(f'-- surrogate replicate {replicate_id + 1}/{self.n_replicates} --')
            metric_chunks = {}
            for start in range(0, n_candidates, chunk_size):
                stop = min(start + chunk_size, n_candidates)
                try:
                    scores, metrics = self._evaluate_surrogate_batch(
                        candidates[start:stop], replicate_id
                    )
                except Exception as exc:
                    errors.append({
                        'replicate': int(replicate_id),
                        'candidate_start': int(start),
                        'candidate_stop': int(stop),
                        'error': repr(exc),
                    })
                    continue
                fitness[output_rep, start:stop] = scores
                for key, values in metrics.items():
                    metric_chunks.setdefault(key, []).append(values)
            replicate_diagnostics.append({
                key: {
                    'mean': float(np.mean(np.concatenate(values))),
                    'std': float(np.std(np.concatenate(values))),
                }
                for key, values in metric_chunks.items()
            })

        self.last_eval_diagnostics = {
            'replicate_ids': replicate_ids,
            'n_replicates': len(replicate_ids),
            'surrogate_n_cells': int(self.surrogate_n_cells),
            'surrogate_mr_jitter_std': float(self.surrogate_mr_jitter_std),
            'surrogate_pca_components': int(self.surrogate_k),
            'fitness_batch_size': int(chunk_size),
            'replicate_metrics': replicate_diagnostics,
            'errors': errors,
            'n_errors': int(len(errors)),
            'fitness': self.fitness_diagnostics(fitness),
        }
        return fitness

    def evaluate_last_crossover(self, candidates: np.ndarray, replicate_ids=None) -> dict:
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
            pre_mutation_fitness = self.eval_candidates(
                self.last_pre_mutation_candidates,
                replicate_ids=replicate_ids,
            )
        post_mutation_fitness = self.eval_candidates(
            post_mutation_candidates,
            replicate_ids=replicate_ids,
        )

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
        replicate_ids=None,
    ):
        candidates = np.asarray(candidates, dtype=np.float32)
        fitness = np.asarray(fitness, dtype=np.float64)
        if candidates.shape[0] != self.n_population:
            raise ValueError('candidate count does not match n_population')
        if fitness.ndim != 2 or fitness.shape[0] < 1 or fitness.shape[1] != self.n_population:
            raise ValueError(
                f'expected fitness shape (n_replicates, {self.n_population}), '
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
            pre_mutation_fitness = self.eval_candidates(
                self.last_pre_mutation_candidates,
                replicate_ids=replicate_ids,
            )
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

        return self.canonicalize_rows(next_candidates)


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
        return self.canonicalize_rows(trials)


def _save_optimizer_checkpoint(path: str | None, payload: dict) -> None:
    """Atomically persist an optimizer checkpoint when a path is configured."""
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with open(temporary, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(temporary, destination)


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
    checkpoint_path = ga.cfg.get('optimizer_checkpoint_path')

    evo_log = []
    best_candidate = None
    best_fitness = -np.inf
    for i in range(max_generations):
        print(f'=== gen {i+1}/{max_generations} ===')
        train_ids, score_ids = ga.generation_replicates(i)
        population_before = ga.population_diagnostics(candidates)
        train_fitness = ga.eval_candidates(candidates, replicate_ids=train_ids)
        train_scores = ga.aggregate_fitness(train_fitness)
        valid = np.isfinite(train_scores)
        if not valid.any():
            raise ValueError(f'generation {i} produced no finite fitness values')
        generation_best = float(np.nanmax(train_scores))
        train_evaluation = dict(ga.last_eval_diagnostics)
        next_gen = ga.next_generation(
            candidates,
            fitness=train_fitness,
            p_crossover=p_crossover,
            protect_crossover_fraction=protect_crossover_fraction,
            p_mutation=p_mutation,
            mutation_scale=mutation_scale,
            n_elites=n_elites,
            replicate_ids=train_ids,
        )
        if diagnose_crossover:
            ga.evaluate_last_crossover(next_gen, replicate_ids=train_ids)

        survival_pool = np.concatenate([candidates, next_gen], axis=0)
        score_fitness = ga.eval_candidates(survival_pool, replicate_ids=score_ids)
        score_scores = ga.aggregate_fitness(
            score_fitness, risk_weight=ga.validation_risk_weight
        )
        score_evaluation = dict(ga.last_eval_diagnostics)
        valid_score = np.isfinite(score_scores)
        if valid_score.sum() < ga.n_population:
            raise ValueError(
                f'generation {i} produced only {valid_score.sum()} valid survival scores'
            )
        selected_indices = np.argsort(
            np.where(valid_score, score_scores, -np.inf)
        )[-ga.n_population:]
        candidates = ga.canonicalize_rows(survival_pool[selected_indices])
        selected_scores = score_scores[selected_indices]
        score_best_index = int(np.argmax(score_scores))
        score_best = float(score_scores[score_best_index])
        if score_best > best_fitness:
            best_fitness = score_best
            best_candidate = survival_pool[score_best_index].copy()

        finite_fitness = train_fitness[np.isfinite(train_fitness)]
        qs = np.quantile(
            finite_fitness,
            q=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
        )
        evo_log.append({
            'generation': i,
            'train_replicate_ids': train_ids,
            'score_replicate_ids': score_ids,
            'mean_fitness': float(np.mean(finite_fitness)),
            'variance_fitness': float(np.var(finite_fitness)),
            'generation_best': generation_best,
            'score_best': score_best,
            'best_so_far': best_fitness,
            'quantiles': [float(q) for q in qs],
            'population_before': population_before,
            'evaluation': train_evaluation,
            'score_evaluation': score_evaluation,
            'score_population_summary': {
                'mean': float(np.mean(selected_scores)),
                'std': float(np.std(selected_scores)),
                'best': float(np.max(selected_scores)),
            },
        })
        evo_log[-1]['operators'] = ga.last_operator_diagnostics
        evo_log[-1]['population_after'] = ga.population_diagnostics(candidates)
        operators = ga.last_operator_diagnostics
        crossover_log = operators['crossover']
        print(
            f'train={np.mean(finite_fitness):.4g} score={score_best:.4g} '
            f'best={best_fitness:.4g} '
            f'finite={train_evaluation["fitness"]["finite_score_fraction"]:.3f} '
            f'errors={train_evaluation["n_errors"]} '
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
        _save_optimizer_checkpoint(checkpoint_path, {
            'algorithm': 'ga',
            'generation': i,
            'candidates': candidates,
            'best_candidate': best_candidate,
            'best_fitness': best_fitness,
            'evolution_log': evo_log,
            'grn': ga.grn_metadata,
            'surrogate': ga.surrogate_metadata(),
        })

    final_replicate_fitness = ga.eval_candidates(
        best_candidate[None, ...], replicate_ids=list(range(ga.n_replicates))
    )
    final_robust_fitness = float(
        ga.aggregate_fitness(
            final_replicate_fitness,
            risk_weight=ga.validation_risk_weight,
        )[0]
    )
    return {
        'best_candidate': best_candidate,
        'best_fitness': best_fitness,
        'final_robust_fitness': final_robust_fitness,
        'final_replicate_fitness': final_replicate_fitness,
        'population': candidates,
        'evolution_log': evo_log,
        'grn': ga.grn_metadata,
        'surrogate': ga.surrogate_metadata(),
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
    Target/trial pairs use greedy one-to-one replacement, providing implicit
    elitism without a separate elite count. DE uses the target-free intrinsic
    program objective; the existing GA loop retains its target-aware mode.
    """
    if max_generations < 1:
        raise ValueError('max_generations must be positive')

    de = MRStateGAOptimizer(
        config_fn,
        objective_mode='intrinsic',
        **de_kwargs,
    )
    de.generate_tree()
    candidates = de.initialize_population()
    print(candidates.shape)
    checkpoint_path = de.cfg.get('optimizer_checkpoint_path')
    best_fitness = -np.inf
    best_candidate = None
    evo_log = []

    for generation in range(max_generations):
        print(f'=== DE gen {generation + 1}/{max_generations} ===')
        _, score_ids = de.generation_replicates(generation)
        population_before = de.population_diagnostics(candidates)

        trials = de.next_generation_de(
            candidates,
            differential_weight=differential_weight,
            crossover_rate=crossover_rate,
        )
        score_pool = np.concatenate([candidates, trials], axis=0)
        score_fitness = de.eval_candidates(score_pool, replicate_ids=score_ids)
        score_pool_scores = de.aggregate_fitness(
            score_fitness, risk_weight=de.validation_risk_weight
        )
        parent_scores_before = score_pool_scores[:de.n_population]
        trial_scores = score_pool_scores[de.n_population:]
        accepted = np.isfinite(trial_scores) & np.isfinite(parent_scores_before) & (
            trial_scores > parent_scores_before
        )
        candidates[accepted] = trials[accepted]
        parent_scores = np.where(accepted, trial_scores, parent_scores_before)

        finite_scores = np.isfinite(parent_scores)
        if not finite_scores.any():
            raise ValueError(
                f'DE generation {generation} produced no finite population fitness values'
            )
        generation_best_index = int(np.flatnonzero(finite_scores)[np.argmax(parent_scores[finite_scores])])
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
            'score_replicate_ids': score_ids,
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
        _save_optimizer_checkpoint(checkpoint_path, {
            'algorithm': 'de',
            'generation': generation,
            'candidates': candidates,
            'best_candidate': best_candidate,
            'best_fitness': best_fitness,
            'evolution_log': evo_log,
            'grn': de.grn_metadata,
            'surrogate': de.surrogate_metadata(),
        })

    final_replicate_fitness = de.eval_candidates(
        best_candidate[None, ...], replicate_ids=list(range(de.n_replicates))
    )
    final_robust_fitness = float(
        de.aggregate_fitness(
            final_replicate_fitness,
            risk_weight=de.validation_risk_weight,
        )[0]
    )
    return {
        'best_candidate': best_candidate,
        'best_fitness': best_fitness,
        'final_robust_fitness': final_robust_fitness,
        'final_replicate_fitness': final_replicate_fitness,
        'population': candidates,
        'evolution_log': evo_log,
        'grn': de.grn_metadata,
        'surrogate': de.surrogate_metadata(),
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
        max_generations=70,
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
        with open(f'/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260901_01.de_F05_CR08.seed{seed}.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'de_F03_CR05':
        ret = exp_de_F03_CR05(config_fn, seed=seed)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.de_F03_CR05.seed{seed}.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'de_F03_CR08':
        ret = exp_de_F03_CR08(config_fn, seed=seed)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260831_01.de_F03_CR08.seed{seed}.pickle', 'wb') as f:
            pickle.dump(ret, f)

    elif sys.argv[1] == 'de_F05_CR05':
        ret = exp_de_F05_CR05(config_fn, seed=seed)
        with open('/content/Drive/MyDrive/Colab Notebooks/ga_opt_log_20260827_02.de_F05_CR05.seed{seed}.pickle', 'wb') as f:
            pickle.dump(ret, f)


    else:
        assert(False)

    sys.exit(0)
