from itertools import combinations, product
from typing import Union, List, List
import pandas as pd

from BayesNet import BayesNet
import pandas as pd


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
        
        self.variables = self.bn.get_all_variables()
        self.extended_factor = {}

    def prune(self, Q: List[str], e: pd.Series) -> None:
        """
        Given a set of query variables Q and evidence e, performs node and edge pruning on
        the BN such that queries of the form P(Q|e) can still be correctly calculated.

        :param Q: List of query variables. E.g.: ["X", "Y", "Z"]
        :param e: Evidence as a series of assignments. E.g.: pd.Series({"A": True, "B": False})
        """
        # Edge pruning - remove outgoing edges of every node in evidence e
        for e_node, e_value in e.items():
            for edge in list(self.bn.out_edges(e_node)):
                # Delete the edge
                self.bn.del_edge(edge)

                # Update the cpts of the nodes on the receiving side of the edge
                recv_node = edge[1]
                new_cpt = self.bn.reduce_factor(pd.Series({e_node: e_value}), self.bn.get_cpt(recv_node))
                new_cpt = self.marginalization(e_node, new_cpt)
                self.bn.update_cpt(recv_node, new_cpt)
        
        # Node pruning - iteratively delete any leaf nodes that do not appear in Q or e
        node_deleted = True
        while node_deleted:
            node_deleted = False
            for node in self.bn.get_all_variables():
                if self.bn.is_leaf_node(node) and (node not in [*Q, *list(e.keys())]):
                    self.bn.del_var(node)
                    node_deleted = True
                    break
    
    def is_dsep(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z.
        """
        # Delete all outgoing edges from nodes in Z
        for node in Z:
            for edge in list(self.bn.out_edges(node)):
                self.bn.del_edge(edge)
            
        # Iteratively delete any leaf nodes that are not in X, Y or Z
        node_deleted = True
        while node_deleted:
            node_deleted = False
            for node in self.bn.get_all_variables():
                if self.bn.is_leaf_node(node) and (node not in [*X, *Y, *Z]):
                    self.bn.del_var(node)
                    node_deleted = True
                    break
        
        # If X and Y are disconnected, then they are d-separated by Z
        for x in X:
            reachable_nodes = self.bn.all_reachable(x)
            if any(node in Y for node in reachable_nodes):
                return False
        
        return True
    
    def is_independent(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z
        """

        return self.is_dsep(X, Y, Z)

    def marginalization(self, X, cpt):
        """
        This function computes the CPT in which the variable X is summed-out 
        """

        variables_left = [variable for variable in cpt.columns if variable != X and variable != 'p']

        # Take the sum of the factors
        new_cpt  = pd.DataFrame(cpt.groupby(variables_left, as_index=False).agg({'p': 'sum'}))
        
        return new_cpt
        
    def maxing_out(self, X, cpt):
        """
        This function computes the CPT in which the variable X is maxed-out
        """

        # Compute the CPT with the maximum probabilty when X is maxed-out 
        variables_left = [variable for variable in cpt.columns if variable != X and variable != 'p']
        new_cpt = pd.DataFrame(cpt.groupby(variables_left).agg({"p": "max"}))
        new_cpt.reset_index(inplace=True)
        
        # Check if there are any previous factors in the table 
        previous_factors = [column for column in cpt.columns.tolist() if "extended factor" in column]

        # Compute the new CPT with the extended factor added
        extended_factor = pd.merge(cpt, new_cpt, on=["p"], how="inner").rename(columns= {X: "extended factor " + X})[f'extended factor {X}']
        
        if previous_factors:
            return new_cpt.assign(**dict(cpt[previous_factors]), **{f'extended factor {X}': extended_factor}) 
        else:
            return new_cpt.assign(**{f"extended factor {X}": extended_factor})

    def factor_multiplication(self, cpt1, cpt2):
        """
        This function computes the multiplied factor of two factors for two cpt's
        """

        cpt1_variables = list(cpt1.columns)
        cpt2_variables = list(cpt2.columns)
        common_variables = [variable for variable in cpt1_variables if variable in cpt2_variables and variable != 'p']

        if not common_variables:
            return 'ERROR: no common variables in CPTs, no multiplication possible'

        cpt_combined = cpt1.merge(cpt2, left_on=common_variables ,right_on=common_variables, suffixes=('_1', '_2'))
        cpt_combined['p'] = cpt_combined['p_1'] * cpt_combined['p_2']
        cpt_combined = cpt_combined.drop(['p_1','p_2'], axis=1)

        return cpt_combined

    
    def min_degree_ordering(self, X):
        """
        Given a set of variables X in the Bayesian network, 
        compute a good ordering for the elimination of X based on the min-degree heuristics.
        """

        graph = self.bn.get_interaction_graph()

        degrees = dict(graph.degree)
        for node, _ in graph.degree:
            if node not in X:
                del degrees[node]
        degrees = sorted(degrees.items(), key=lambda item: item[1])

        order = []
        while len(degrees):
            node = degrees[0][0]
            print(node)

            # connect neighbours with each other
            neighbours = list(graph.neighbors(node)) # get all the neighbours of the node, participating in order
            for ind, _ in enumerate(neighbours):
                neighbors_i = list(graph.neighbors(neighbours[ind])) # get all the neighbours from the current neibour
                if ind < len(neighbours):
                    for jnd in range(ind + 1, len(neighbours)): # check residual neighbours
                        if neighbours[jnd] not in neighbors_i:
                            graph.add_edge(neighbours[ind], neighbours[jnd])

            # remove node from interaction graph
            graph.remove_node(node)
            order.append(node)

            # recalculate degrees
            degrees = dict(graph.degree)
            for node, _ in graph.degree:
                if node not in X:
                    del degrees[node]
            degrees = sorted(degrees.items(), key=lambda item: item[1])

        return order

    @staticmethod
    def calculate_additional_edges(graph, node):
        #TODO: rewrite with combinations
        amount = 0
        neighbours = list(graph.neighbors(node)) # get all the neighbours of the node
        for ind, _ in enumerate(neighbours):
            neighbors_i = list(graph.neighbors(neighbours[ind])) # get all the neighbours from the current neibour
            if ind < len(neighbours):
                for jnd in range(ind + 1, len(neighbours)): # check residual neighbours
                    if neighbours[jnd] not in neighbors_i:
                        amount += 1
        
        return amount

    def min_fill_ordering(self, X):
        """Given a set of variables X in the Bayesian network, 
        compute a good ordering for the elimination of X based on the min-fill heuristics.
        """
        graph = self.bn.get_interaction_graph()

        amounts = dict()
        for node in X:
            amounts[node] = self.calculate_additional_edges(graph, node)
        amounts = sorted(amounts.items(), key=lambda item: item[1])

        order = []
        while len(amounts):
            node = amounts[0][0]
            print(node)

            # connect neighbours with each other
            neighbours = list(graph.neighbors(node)) # get all the neighbours of the node, participating in order
            for ind, _ in enumerate(neighbours):
                neighbors_i = list(graph.neighbors(neighbours[ind])) # get all the neighbours from the current neibour
                if ind < len(neighbours):
                    for jnd in range(ind + 1, len(neighbours)): # check residual neighbours
                        if neighbours[jnd] not in neighbors_i:
                            graph.add_edge(neighbours[ind], neighbours[jnd])

            # remove node from interaction graph
            graph.remove_node(node)
            order.append(node)

            # recalculate degrees
            amounts = dict()
            for node in (X - set(order)):
                amounts[node] = self.calculate_additional_edges(graph, node)
            amounts = sorted(amounts.items(), key=lambda item: item[1])
        
        return order

    def elimination_order(self, X, heuristic=None):
        if heuristic is None:
            order = X
        elif heuristic == 'min_deg':
            order = self.min_degree_ordering(X)
        elif heuristic == 'min_fill':
            order = self.min_fill_ordering(X)
        else:
            raise ValueError('Unknown ordering heuristic')

        return order

    def variable_elimination(self, cpt, X, heuristic=None):
        """
        Sum out a set of variables by using variable elimination. 
        """
        new_cpt = cpt.copy()
        order = self.elimination_order(X, heuristic)

        for node in order:
            if node in new_cpt and len(new_cpt.columns) != 1:
                new_cpt = self.marginalization(node, cpt)
        
        return new_cpt


    def marginal_distributions(self, Q, e=None):
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). 
        Note that Q is a subset of the variables 
        in the Bayesian network X with Q ⊂ X but can also be Q = X. 
        """

        
        
        # reduce e
        # compute probability Q and e - P(Q, e)
        # compute probability of e
        # compute P(Q, e)/P(e)

        pass


if __name__ == "__main__":
    bayes = BNReasoner('testing/lecture_example.BIFXML')
    bayes.print()
    