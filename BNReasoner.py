from typing import Union
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
        
if __name__ == "__main__":
    bayes = BNReasoner('testing/lecture_example.BIFXML')
    