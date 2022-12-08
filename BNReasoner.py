import pgmpy
import networkx as nx
from typing import Union
from BayesNet import BayesNet
import pandas as pd
import matplotlib.pyplot as plt


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

net = BNReasoner("C:/Users/Ellis/Documents/VU/Knowledge Representation/KR21_project2-main/testing/dog_problem.BIFXML")

def d_seperated(model, x, y, z):
    if nx.d_separated(model, x, y, z):
        print("X is d-seperated of Y given Z")  #overbodig, maar meer om voor ons duidelijk te hebben als we straks een eigen bn maken
        return True
    else:
        print("X is not d-seperated of Y given Z") # same here
        return False

def independent(model, x, y, z):
    if d_seperated(model, x, y, z) is True:
        print("X is independent of Y given Z") #same here
        return True
    else:
        print("X is dependent of Y given Z") #same here
        return False

def min_degree_heuristic(graph):
    nx.approximation.treewidth_min_degree(graph)
    return graph

def min_fill_heur(graph):
    print(nx.approximation.treewidth_min_fill_in(graph))
    return graph
