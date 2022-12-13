from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import XMLBIFReader
import math
import random
import itertools
import pandas as pd
from copy import deepcopy


class BayesNet:

    def __init__(self) -> None:
        # initialize graph structure
        self.structure = nx.DiGraph()

    # LOADING FUNCTIONS ------------------------------------------------------------------------------------------------
    def create_bn(self, variables: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, pd.DataFrame]) -> None:
        """
        Creates the BN according to the python objects passed in.
        
        :param variables: List of names of the variables.
        :param edges: List of the directed edges.
        :param cpts: Dictionary of conditional probability tables.
        """
        # add nodes
        [self.add_var(v, cpt=cpts[v]) for v in variables]

        # add edges
        [self.add_edge(e) for e in edges]

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            raise Exception('The provided graph is not acyclic.')

    def load_from_bifxml(self, file_path: str) -> None:
        """
        Load a BayesNet from a file in BIFXML file format. See description of BIFXML here:
        http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/

        :param file_path: Path to the BIFXML file.
        """
        # Read and parse the bifxml file
        with open(file_path) as f:
            bn_file = f.read()
        bif_reader = XMLBIFReader(string=bn_file)

        # load cpts
        cpts = {}
        # iterating through vars
        for key, values in bif_reader.get_values().items():
            values = values.transpose().flatten()
            n_vars = int(math.log2(len(values)))
            worlds = [list(i) for i in itertools.product([False, True], repeat=n_vars)]
            # create empty array
            cpt = []
            # iterating through worlds within a variable
            for i in range(len(values)):
                # add the probability to each possible world
                worlds[i].append(values[i])
                cpt.append(worlds[i])

            # determine column names
            columns = bif_reader.get_parents()[key]
            columns.reverse()
            columns.append(key)
            columns.append('p')
            cpts[key] = pd.DataFrame(cpt, columns=columns)
        
        # load vars
        variables = bif_reader.get_variables()
        
        # load edges
        edges = bif_reader.get_edges()

        self.create_bn(variables, edges, cpts)

    # METHODS THAT MIGHT ME USEFUL -------------------------------------------------------------------------------------

    def get_children(self, variable: str) -> List[str]:
        """
        Returns the children of the variable in the graph.
        :param variable: Variable to get the children from
        :return: List of children
        """
        return [c for c in self.structure.successors(variable)]

    def get_cpt(self, variable: str) -> pd.DataFrame:
        """
        Returns the conditional probability table of a variable in the BN.
        :param variable: Variable of which the CPT should be returned.
        :return: Conditional probability table of 'variable' as a pandas DataFrame.
        """
        try:
            return self.structure.nodes[variable]['cpt']
        except KeyError:
            raise Exception('Variable not in the BN')

    def get_all_variables(self) -> List[str]:
        """
        Returns a list of all variables in the structure.
        :return: list of all variables.
        """
        return [n for n in self.structure.nodes]

    def get_all_cpts(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of all cps in the network indexed by the variable they belong to.
        :return: Dictionary of all CPTs
        """
        cpts = {}
        for var in self.get_all_variables():
            cpts[var] = self.get_cpt(var)

        return cpts

    def get_interaction_graph(self):
        """
        Returns a networkx.Graph as interaction graph of the current BN.
        :return: The interaction graph based on the factors of the current BN.
        """
        # Create the graph and add all variables
        int_graph = nx.Graph()
        [int_graph.add_node(var) for var in self.get_all_variables()]

        # connect all variables with an edge which are mentioned in a CPT together
        for var in self.get_all_variables():
            involved_vars = list(self.get_cpt(var).columns)[:-1]
            for i in range(len(involved_vars)-1):
                for j in range(i+1, len(involved_vars)):
                    if not int_graph.has_edge(involved_vars[i], involved_vars[j]):
                        int_graph.add_edge(involved_vars[i], involved_vars[j])
        return int_graph

    @staticmethod
    def get_compatible_instantiations_table(instantiation: pd.Series, cpt: pd.DataFrame):
        """
        Get all the entries of a CPT which are compatible with the instantiation.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param cpt: cpt to be filtered
        :return: table with compatible instantiations and their probability value
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        compat_indices = cpt[var_names] == instantiation[var_names].values
        compat_indices = [all(x[1]) for x in compat_indices.iterrows()]
        compat_instances = cpt.loc[compat_indices]
        return compat_instances

    def update_cpt(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Replace the conditional probability table of a variable.
        :param variable: Variable to be modified
        :param cpt: new CPT
        """
        self.structure.nodes[variable]["cpt"] = cpt

    @staticmethod
    def reduce_factor(instantiation: pd.Series, cpt: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and returns a new factor in which all probabilities which are incompatible with the instantiation
        passed to the method to 0.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param cpt: cpt to be reduced
        :return: cpt with their original probability value and zero probability for incompatible instantiations
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        if len(var_names) > 0:  # only reduce the factor if the evidence appears in it
            new_cpt = deepcopy(cpt)
            incompat_indices = cpt[var_names] != instantiation[var_names].values
            incompat_indices = [any(x[1]) for x in incompat_indices.iterrows()]
            new_cpt.loc[incompat_indices, 'p'] = 0.0
            return new_cpt
        else:
            return cpt

    def draw_structure(self, node_size=3000) -> None:
        """
        Visualize structure of the BN.
        """
        nx.draw(self.structure, with_labels=True, node_size=node_size)
        plt.show()

    # BASIC HOUSEKEEPING METHODS ---------------------------------------------------------------------------------------

    def add_var(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Add a variable to the BN.
        :param variable: variable to be added.
        :param cpt: conditional probability table of the variable.
        """
        if variable in self.structure.nodes:
            raise Exception('Variable already exists.')
        else:
            self.structure.add_node(variable, cpt=cpt)

    def add_edge(self, edge: Tuple[str, str]) -> None:
        """
        Add a directed edge to the BN.
        :param edge: Tuple of the directed edge to be added (e.g. ('A', 'B')).
        :raises Exception: If added edge introduces a cycle in the structure.
        """
        if edge in self.structure.edges:
            raise Exception('Edge already exists.')
        else:
            self.structure.add_edge(edge[0], edge[1])

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            self.structure.remove_edge(edge[0], edge[1])
            raise ValueError('Edge would make graph cyclic.')

    def del_var(self, variable: str) -> None:
        """
        Delete a variable from the BN.
        :param variable: Variable to be deleted.
        """
        self.structure.remove_node(variable)

    def del_edge(self, edge: Tuple[str, str]) -> None:
        """
        Delete an edge form the structure of the BN.
        :param edge: Edge to be deleted (e.g. ('A', 'B')).
        """
        self.structure.remove_edge(edge[0], edge[1])

    # CUSTOM METHODS ----------------------------------------------------------
    def get_num_nodes(self) -> int:
        """
        :return: An integer for the number of nodes in the network
        """
        return self.structure.number_of_nodes()

    def get_num_edges(self) -> int:
        """
        :return: An integer for the number of edges in the network
        """
        return self.structure.number_of_edges()


    def is_leaf_node(self, node: str) -> bool:
        """
        Returns True if the given node has no outgoing edges and False otherwise
        :param: The node being queried
        :return: True/False
        """
        return self.structure.out_degree(node) == 0

    def all_leaves(self) -> List[str]:
        """
        return: A list of leaf nodes in the BN
        """
        return [node for node in self.structure.nodes() if self.structure.out_degree(node) == 0]


    def out_edges(self, node: str) -> List[Tuple[str, str]]:
        """
        Returns a list of outgoing edges from the given node
        :param node: A string for the name of the node
        :return: A list of tuples representing the outgoing edges
        """
        return self.structure.out_edges(node)
    
    def all_reachable(self, node: str) -> List[str]:
        """
        Returns a list of nodes reachable from the given node
        :param node: A string for the name of the node
        :return: A list of nodes reachable from the given one
        """
        return list(nx.shortest_path(self.structure.to_undirected(), node).keys())
    
    def rand_Qe(self, q_ratio: float, e_ratio: float) -> Tuple[List[str], List[str]]:
        """
        Generates a random list of nodes for Q and e. The number of nodes is 
        determined by the given ratios of the graph size. If this number is less 
        than zero, a single node is chosen at random
        :param q_ratio: The fraction of nodes to appear in Q (rounded down)
        :param e_ratio: The fraction of nodes to appear in e (rounded down)
        :return: A tuple containing a list of nodes in Q and e respectively
        :raises ValueError: If a ratio is less than 0
        :raises ValueError: If q_ratio + e_ratio > 1
        """

        # Input checks
        if (q_ratio < 0) or (e_ratio < 0):
            raise ValueError('Ratios must be in the range [0, 1]')
        
        if q_ratio + e_ratio > 1:
            raise ValueError('The sum of ratios must be less than 1')
        
        # Determine the number of nodes to sample
        nodes = list(self.structure.nodes())
        size = len(nodes)

        num_Q = max(int(q_ratio * size), 1)
        num_e = max(int(e_ratio * size), 1)

        # Randomize sampling order
        q_first = random.randint(0, 1)

        # Sample
        if q_first:
            Q = random.sample(nodes, num_Q)
            remaining_nodes = list(set(nodes) - set(Q))
            e_vars = random.sample(remaining_nodes, num_e)
        else: # e first
            e_vars = random.sample(nodes, num_e)
            remaining_nodes = list(set(nodes) - set(e))
            Q = random.sample(remaining_nodes, num_Q)

        # Randomly assign truth values to variables in e
        e = {node: random.choice([True, False]) for node in e_vars}
        e = pd.Series(e)
        
        return (Q, e)

        

        
