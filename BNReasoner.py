from typing import Union
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

    def marginalization(self, X, cpt):
        """
        This function computes the CPT in which the variable X is summed-out 
        """

        variables_left = [variable for variable in new_cpt.columns if variable != X and variable != 'p']

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

        # Get instantiation of X where variable X is maxed-out
        combined_cpt = pd.concat([cpt, new_cpt], axis=1)
        reduced_cpt = combined_cpt.dropna(axis=0, how='any')
        reduced_cpt[X] = reduced_cpt[X].map({True: f'{X} = True', False: f'{X} = No'}) 
        reduced_cpt = reduced_cpt.iloc[:, :-2:]
        reduced_cpt["p"] = reduced_cpt["p"].astype(str)
        reduced_cpt['factor'] = reduced_cpt[["p", X]].agg(': '.join, axis=1)
        extended_factor = reduced_cpt.drop([X, "p"], axis=1)
        
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
    print(bayes.print())

    
