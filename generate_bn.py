# import networkx as nx
import numpy as np
from pgmpy.models import BayesianNetwork


# Set random seed for reproducibility
np.random.seed(123)

# Set the range of node numbers to generate networks for
start = 10
end = 50
skip = 3
node_counts = range(start, end+1, skip)

# Set the number of networks to generate for each node number
n_networks = 1

# Set the probability of an edge between any two nodes
edge_prob1 = 0.03

# Iterate over the range of node numbers
for n_node in node_counts:

    # Iterate over the number of networks
    for i in range(n_networks):

        G = BayesianNetwork.get_random(n_nodes=n_node, edge_prob=edge_prob1, n_states=2)

        G.save(f'testing/Part_2/network_{n_node}.XMLBIF', filetype='XMLBIF')
