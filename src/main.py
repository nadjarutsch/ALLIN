import torch
import numpy as np
import wandb

import os
if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')
    
import matplotlib.pyplot as plt
from collections import defaultdict

import networkx as nx
import cdt
#cdt.SETTINGS.rpath = '/usr/local/bin/Rscript' # for macOS
cdt.SETTINGS.rpath = '/sw/arch/Debian10/EB_production/2021/software/R/4.1.0-foss-2021a/lib/R/bin/Rscript'

import data_generation.causal_graphs.graph_generation as graph_gen
import data_generation.causal_graphs.graph_visualization as visual
import data_generation.data_generation as data_gen
import models.multivar_mlp as mmlp
import causal_discovery.prepare_data as cd
import outlier_detection.model_based as ood
import outlier_detection.depcon_kernel as depcon
import metrics

from platform import python_version
#assert python_version() == '3.10.4'


equations = defaultdict(dict)
equations['A']['input_names'] = []
equations['A']['mu_fn'] = lambda inputs : 0
equations['A']['sigma_fn'] = lambda inputs : 0.5 

equations['B']['input_names'] = ['A']
equations['B']['mu_fn'] = lambda inputs : inputs['A']
equations['B']['sigma_fn'] = lambda inputs : 0.5

equations['C']['input_names'] = ['B']
equations['C']['mu_fn'] = lambda inputs : inputs['B']
equations['C']['sigma_fn'] = lambda inputs : 0.5

equations['D']['input_names'] = ['C']
equations['D']['mu_fn'] = lambda inputs : inputs['C']
equations['D']['sigma_fn'] = lambda inputs : 0.5

equations['E']['input_names'] = ['D']
equations['E']['mu_fn'] = lambda inputs : inputs['D']
equations['E']['sigma_fn'] = lambda inputs : 0.5  


N_OBS = 10000
INT_RATIO = 0.01
BATCH_SIZE = 128

lr = 1e-3
loss = mmlp.nll
epochs = 5
fit_epochs = 60
stds = 4
seeds = list(range(50))
variables = list(equations.keys())
NUM_VARS = len(variables)
true_target_indices = np.cumsum([N_OBS] + [INT_RATIO * N_OBS] * NUM_VARS)
alpha_skeleton = 0.00001
alpha = 0.00001


def main():
    
 #   dag = graph_gen.generate_graph_from_equations(equations) 
    
    # wandb config for logging
    config = dict(
        n_obs = N_OBS,
        int_ratio = INT_RATIO,
        batch_size = BATCH_SIZE,
        lr = lr,
        epochs = epochs,
        fit_epochs = fit_epochs,
        threshold = f'{stds} stds',
        num_vars = NUM_VARS,
        graph_structure = 'random',
        edge_prob=0.4
    )
    
#    true_graph = nx.DiGraph()
 #   true_graph.add_nodes_from(variables)
  #  true_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
    
    device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for seed in seeds:
        config['seed'] = seed
        run = wandb.init(project="idiod", entity="nadjarutsch", notes='code setup', group='prototype', tags=['chain', 'model-based', 'known intervention targets'], config=config, reinit=True)
        with run:
            # generate data
            dag = data_gen.generate_dag(num_vars=config['num_vars'], edge_prob=config['edge_prob'], fns='linear gaussian')
            variables = [v.name for v in dag.variables]
            
            true_graph = dag.nx_graph
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(true_graph)
            nx.draw(true_graph, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"true graph": wandb.Image(plt)})
            plt.close()
            
            synth_dataset = data_gen.generate_data(dag=dag, n_obs=N_OBS, int_ratio=INT_RATIO, seed=seed)
    
            # initial causal discovery (skeleton)
            df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)
            
            # logging
            #tbl = wandb.Table(dataframe=df)
            #wandb.log({"initial mixed data": tbl})
            
            model_pc = cdt.causality.graph.PC(alpha=alpha_skeleton, CItest='rcot')    
            skeleton = model_pc.create_graph_from_data(df)
            
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(skeleton)
            nx.draw(skeleton, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"skeleton": wandb.Image(plt)})
            plt.close()
            
            adj_matrix = torch.from_numpy(nx.to_numpy_array(skeleton))
    
            # intervention detection (ood)
            print('Creating model...')
        #    gnmodel = mmlp.GaussianNoiseModel(num_vars=dag.num_vars, hidden_dims=[])
        #    optimizer = torch.optim.Adam(gnmodel.parameters(), lr=lr)
        #    partitions_obs = ood.cluster(synth_dataset, gnmodel, loss, optimizer, epochs, fit_epochs, adj_matrix, stds, BATCH_SIZE)
            synth_dataset.update_partitions([list(range(N_OBS)), list(range(N_OBS, int(N_OBS + config['num_vars'] * N_OBS * INT_RATIO)))])
            partitions_temp = depcon.kernel_k_means(synth_dataset.partitions[1].features[...,0], num_clus=config['num_vars'], device=device)
         #   print(labels)
            partitions_int = []
            for part in partitions_temp:
                partitions_int.append([p + N_OBS for p in part])
         #   partitions_int = [[p + N_OBS for part in partitions_int for p in part]
            partitions = [list(range(N_OBS))]
            partitions.extend(partitions_int)
            synth_dataset.update_partitions(partitions)
            
            # inspect clusters
            borders = true_target_indices.tolist()
            borders.insert(0,0)
    #        borders.remove(N_OBS)
            for p_i, part in enumerate(partitions_int):
                counts = []
                for i in range(len(partitions)):
                    idx_lower = borders[i]
                    idx_upper = borders[i+1]

                    print([(p >= idx_lower) and (p < idx_upper) for p in part])
                    counts.append(sum([(p >= idx_lower) and (p < idx_upper) for p in part]))
                                    
                print(counts)
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_axes([0,0,1,1])
                partitions = list(range(len(partitions)))
                ax.bar(partitions, counts)
                wandb.log({f"partition_{p_i}": wandb.Image(plt)})
                plt.close()
            
        #    synth_dataset.set_true_intervention_targets(true_target_indices)
            
            df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)
            
            # logging
            tbl = wandb.Table(dataframe=df)
            wandb.log({"clustered data": tbl})
    
            for node in list(df.columns.values[config['num_vars']:]):
                skeleton.add_node(node)
                skeleton.add_edge(node, node.replace("I_",""))
    
            model_pc = cdt.causality.graph.PC(CItest="rcot", alpha=alpha)
            created_graph = model_pc.orient_directed_graph(df, skeleton)
            created_graph.remove_nodes_from(list(df.columns.values[config['num_vars']:]))
            
            cc = metrics.causal_correctness(true_graph, created_graph)
            shd = cdt.metrics.SHD(true_graph, created_graph)
            wandb.run.summary["SHD"] = shd
            wandb.run.summary["CC"] = cc
    
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(created_graph)
            nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"discovered graph": wandb.Image(plt)})
            plt.close()
    
    




if __name__ == '__main__':
    
    main()