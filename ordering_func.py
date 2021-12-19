from BayesNet import BayesNet
from networkx import Graph

import itertools
import random

def sort_min_degree(graph:Graph, node_list:list) -> list:
    node_neighbour = {}
    for node in node_list:
        node_neighbour[node] = len(list(graph.neighbors(node)))

    sorted_node_dict = dict(sorted(node_neighbour.items() ,key = lambda item:item[1]))
    sorted_node_list = list(sorted_node_dict.keys()) 
    return sorted_node_list


def sort_min_fill(graph:Graph, node_list:list) -> list:
	unsorted_ne_dict = {}
	for node in node_list:
		neighbours = list(graph.neighbors(node))
		combi_n = itertools.combinations(neighbours,2)
		new_edges=0
		for combi in combi_n:
			if(not(combi in graph.edges)):
				new_edges+=1
		unsorted_ne_dict[node] = new_edges

	sorted_node_dict = dict(sorted(unsorted_ne_dict.items() ,key = lambda item:item[1]))
	sorted_node_list = list(sorted_node_dict.keys()) 
	return sorted_node_list

def random_sort(node_list:list) -> list:
	return random.sample(node_list, len(node_list))