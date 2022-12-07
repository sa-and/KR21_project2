from itertools import combinations, product
import pandas as pd
from typing import Union, List
from BayesNet import BayesNet


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

    def prune(self, Q, e):
        """
        Given a set of query variables Q and evidence e, performs node and edge pruning on
        the BN such that queries of the form P(Q|e) can still be correctly calculated.

        :param Q: List of query variables
        :param e: List of query variables
        """
        # Edge pruning - remove outgoing edges of every node in evidence e
        for node in e:
            for edge in list(self.bn.out_edges(node)):
                self.bn.del_edge(edge)
        
        # Node pruning - iteratively delete any leaf nodes that do not appear in Q or e
        node_deleted = True
        while node_deleted:
            node_deleted = False
            for node in self.bn.get_all_variables():
                if self.bn.is_leaf_node(node) and (node not in [*Q, *e]):
                    self.bn.del_var(node)
                    node_deleted = True
                    break
    
    def is_dsep(self, X, Y, Z):
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
    
    def is_independent(self, X, Y, Z):
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z
        """
        return self.is_dsep(X, Y, Z)

    def compute_factor(self):
        pass
    
    def marginalization(self, X, cpt):
        """
        This function computes the CPT in which the variable X is summed-out 
        """

        # Delete node X
        new_cpt = cpt.drop([X], axis=1)
        variables_left = [variable for variable in new_cpt.columns if variable != X and variable != 'p']

        # Take the sum of the factors
        new_cpt = new_cpt.groupby(variables_left).agg({'p': 'sum'})
        cpt.reset_index(inplace=True)

        return new_cpt

    def maxing_out(self, X, cpt):
        """
        This function computes the CPT in which the variable X is maxed-out
        """
        
        # Delete node X
        new_cpt = cpt.drop([X], axis=1)
        variables_left = [variable for variable in new_cpt.columns if variable != X and variable != 'p']

        # Take the max of the factors
        new_cpt = new_cpt.groupby(variables_left).agg({'p': 'max'})
        cpt.reset_index(inplace=True)

        return new_cpt

    def factor_multiplication(self, cpt1, cpt2):
        """
        This function computes the multiplied factor of two factors for two cpt's
        """

        # Add an edge between every neighbour of 𝑋 that is not already connected by an edge
        
        pass

    
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
            # print(node)

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
            order = list(X)
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


    def marginal_distribution(self, Q, e, heuristic='min_deg'):
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). 
        Note that Q is a subset of the variables 
        in the Bayesian network X with Q ⊂ X but can also be Q = X. 
        """

        cpts = self.bn.get_all_cpts()

        # reduce all factors w.r.t. e
        upd_cpts = {}
        for var, cpt in cpts.items():
            upd_cpt = self.bn.get_compatible_instantiations_table(e, cpt)
            upd_cpts[var] = upd_cpt
        # pprint(upd_cpts)

        # get all the variables that are not in Q or e
        X = set(self.bn.get_all_variables()) - Q - set(e.keys())

        # get order of variables summation
        order = self.elimination_order(X, heuristic=heuristic)

        # get joint probability Q and e - P(Q, e)
        p_Q_e = pd.DataFrame()
        visited = []
        for var in order:
            for child in self.bn.get_children(var):
                if child not in visited:
                    if p_Q_e.size == 0:
                        p_Q_e = self.factor_multiplication(upd_cpts[var], upd_cpts[child])
                        visited.extend([var, child])
                    else:
                        p_Q_e = self.factor_multiplication(p_Q_e, upd_cpts[child])
                        visited.append(child)

            p_Q_e = self.marginalization(var, p_Q_e)

        # compute probability of e
        p_e = p_Q_e.copy()
        for var in Q:
            p_e = self.marginalization(var, p_e)
        p_e = p_e['p'][0]

        # divide joint probability on probability of evidence
        p_Q_e['p'] = p_Q_e['p'].apply(lambda x: x/p_e['p'][0])

        return p_Q_e

    def marginal_distribution_brutto(self, Q, e):
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). 
        Note that Q is a subset of the variables 
        in the Bayesian network X with Q ⊂ X but can also be Q = X. 
        """

        cpts = self.bn.get_all_cpts()

        # reduce all factors w.r.t. e
        upd_cpts = {}
        for var, cpt in cpts.items():
            upd_cpt = self.bn.get_compatible_instantiations_table(e, cpt)
            upd_cpts[var] = upd_cpt

        # get joint probability Q and e - P(Q, e)
        p_Q_e = pd.DataFrame()
        visited = []
        for var in self.bn.get_all_variables():
            for child in self.bn.get_children(var):
                if child not in visited:
                    if p_Q_e.size == 0:
                        p_Q_e = self.factor_multiplication(upd_cpts[var], upd_cpts[child])
                        visited.extend([var, child])
                    else:
                        p_Q_e = self.factor_multiplication(p_Q_e, upd_cpts[child])
                        visited.append(child)
        
        # get all the variables that are not in Q or e and sum-out them
        X = set(self.bn.get_all_variables()) - Q - set(e.keys())
        for var in X:
            p_Q_e = self.marginalization(var, p_Q_e)
        
        # compute probability of e
        p_e = p_Q_e.copy()
        for var in Q:
            p_e = bayes.marginalization(var, p_e)
        p_e = p_e['p'][0]

        # divide joint probability on probability of evidence
        p_Q_e['p'] = p_Q_e['p'].apply(lambda x: x/p_e['p'][0])

        return p_Q_e
        

if __name__ == "__main__":
    bayes = BNReasoner('testing/lecture_example.BIFXML')
    