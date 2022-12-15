import networkx as nx
import random
from BayesNet import BayesNet
import itertools
import pandas as pd


class BNGenerator:
    def __init__(self, n_nodes, edge_weight=0.5):
        self.n_nodes = n_nodes
        self.edge_weight = edge_weight
        self.bayes_net = BayesNet()
        self.bayes_net.structure = self.generate_directed_acyclic_graph(self.n_nodes)
        self.generate_cpts()

    def generate_directed_acyclic_graph(self, n_nodes):
        G = nx.gnp_random_graph(n_nodes, self.edge_weight, directed=True)
        DAG = nx.DiGraph([(str(u), str(v), {'weight': random.randint(-10, 10)}) for (u, v) in G.edges() if u < v])
        assert nx.is_directed_acyclic_graph(DAG)
        return DAG

    def generate_cpts(self):
        for node in self.bayes_net.structure.nodes:
            edges = list(self.bayes_net.structure.edges(node))
            columns = [str(e) for _, e in edges] + ([node] if not len(edges) else [str(edges[0][0])]) + ['p']
            cpt = []
            remaining_p = 1
            reversed_list = list(reversed(list(itertools.product([True, False], repeat=len(set(columns) - {'p'})))))
            for truth_values in reversed_list:
                remaining_p = 1 if remaining_p == 0 else remaining_p
                p = random.random() if remaining_p == 1 else remaining_p
                remaining_p = remaining_p - p if remaining_p == 1 else 0
                cpt.append(list(truth_values) + [round(p, 2)])
            pd_cpt = pd.DataFrame(cpt, columns=columns)
            self.bayes_net.structure.add_node(node, cpt=pd_cpt)