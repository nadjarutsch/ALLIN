import matplotlib.pyplot as plt
import data_generation.causal_graphs.graph_visualization as visual
import networkx as nx
import wandb

def plot_graph(graph, title):
    visual.set_edge_attributes(graph)
    plt.figure(figsize=(6, 6))
    colors = visual.get_colors(graph)
    nx.draw(graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black', edge_color=colors)
    wandb.log({title: wandb.Image(plt)})
    plt.close()