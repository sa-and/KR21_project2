from typing import Union, List
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

    def compute_factor(self):
        pass
    
    def marginalization(self, X, cpt):
        """
        This function computes the CPT in which the variable X is summed-out 
        """

        # Delete node X
        new_cpt = cpt.drop([X], axis=1)
        variables_left = [variable for variable in new_cpt.columns if variable != X and variable != 'p']

        # Make case if there is only one variable

        # Take the sum of the factors
        new_cpt = new_cpt.groupby(variables_left).agg({'p': 'sum'})
        cpt.reset_index(inplace=True)

        return new_cpt

    def maxing_out(self, X, cpt):
        """
        This function computes the CPT in which the variable X is maxed-out
        """

        variables_left = [variable for variable in cpt.columns if variable != X and variable != 'p']
        new_cpt = cpt.groupby(variables_left).agg({"p": "max"})
        new_cpt.reset_index(inplace=True)

        # Keep track of instatiation of X that led to maximized value
        # change this 
        extended_factor = cpt.groupby(X).agg({"p": "max"})
        extended_factor = extended_factor.drop(["p"], axis=1)
        extended_factor.reset_index(inplace=True)

        #self.extended_factor[] = cpt.groupby.agg({"p": "max"})
        

        return new_cpt, extended_factor

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

if __name__ == "__main__":
    bayes = BNReasoner('testing/lecture_example.BIFXML')
    
