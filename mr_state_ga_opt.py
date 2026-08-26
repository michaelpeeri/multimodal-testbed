import json
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
    n_replicates: Different DAGs/seeds used to evaluate feature stability
    
    """
    def __init__(self, base_cfg:str, n_candidates:int=200, n_replicates:int=10, n_clusters:int=15):
        with open(base_cfg, 'rt') as f:
            self.cfg = json.load(f)

        self.n_candidates = n_candidates
        self.n_replicates = n_replicates
        self.n_clusters   = n_clusters


    def generate_tree(self):
        base    = self.cfg['_meta']
        bestcfg = self.cfg['best_params']

        # harness:624
        def gen_grn(seed:int):
            temp_path = f'./output_{seed}.csv'
            _, mr_ids, gene_id_to_symbol = generate_sergio_grn_from_reference(
                reference_grn_path=base["reference_grn_path"],
                n_genes=base["n_genes"],
                output_path=temp_path,
                k_dist=("uniform", bestcfg["grn_k_low"], bestcfg["grn_k_low"] + bestcfg["grn_k_span"]),
                hill_coeff_dist=("constant", bestcfg["hill_coeff"]),
                seed=seed,
                coherency_bias=bestcfg["coherency_bias"],
                unknown_mode_repressor_prob=bestcfg["unknown_mode_repressor_prob"],
                canalization_strength=bestcfg["canalization_strength"],
                balancing_strength=bestcfg["balancing_strength"],
                path_decay=bestcfg["path_decay"],
            #    delimiter=base["grn_delimiter"],
            #    regulator_col=base["grn_regulator_col"],
            #    target_col=base["grn_target_col"],
            #    mode_col=base["grn_mode_col"],
            #    activation_labels=base["grn_activation_labels"],
            #    repression_labels=base["grn_repression_labels"],
            #    max_seed_attempts=bestcfg["grn_max_seed_attempts"],
            )
            return temp_path, mr_ids, gene_id_to_symbol

        grns = []
        for seed in range(42,42+1):
            temp_path, mr_ids, gene_id_to_symbol = gen_grn(seed=seed)
            grns.append( (seed, temp_path, mr_ids, gene_id_to_symbol) )


        dags = []
        for _grn in grns:
            seed, temp_path, mr_ids, gene_id_to_symbol = _grn
            dag = load_sergio_dag(
                temp_path,
                shared_coop_state=base["shared_coop_state"],
                mr_gene_ids=mr_ids,
            )
        dags.append(dag)
                    
        self.grns = grns
        self.dags = dags

    def generate_candidates(self) -> np.ndarray:
        """
        Generate fresh candidates
        """
        seed, temp_path, mr_ids, gene_id_to_symbol = self.grns[0]  # use first tree (for now)
        candidates = []
        for rep in range(self.n_replicates):
            new_candidates = sample_sergio_mr_states(
                n_states=self.n_candidates,
                n_mrs=len(mr_ids),
                low =1,    # TODO Use config
                high=5,    # TODO Use config
                design='sobol',
                seed=seed+rep,
            )
            candidates.append( new_candidates )        
        return np.stack(candidates)

    def choose_chromosomes( self, candidates ):
        return candidates[:,:self.n_clusters,:]

    def eval_candidates(self, candidates:np.ndarray):
        """
        """

        selected = np.repeat( self.choose_chromosomes(candidates), repeats=20, axis=1)  # use 20 cells/cluster

        print(f'selected:{selected.shape}')

        decays = 0.8
        batch_raw = sergio_dag_hill_forward_batched(
            selected, self.dags[0], decays=decays)

        stats = []
        for i in tqdm(range(self.n_replicates)):
            stats.append(
                compute_summary_stats(
                    torch.tensor(batch_raw[i], requires_grad=False),
                    n_pca_components=20,  # TODO Use config
                    seed=42,              # TODO Use config
                ) )
        objective = [s['pca_standardized_split_half_subspace_stability'] for s in stats]

        return objective, batch_raw, stats
