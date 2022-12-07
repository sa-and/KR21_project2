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

    def print(self):
        """
        for testing purpose
        """

        all_ctps = self.bn.get_all_cpts()
        
        for ctp in all_ctps.keys():
            test_cpt = all_ctps[ctp]
        
        #print(test_cpt)
        new_cpt, extended_factor = self.maxing_out('Rain?', test_cpt)
        print(new_cpt)

        print(f'extended {extended_factor}')
  

           
    
        
    
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
    bayes.print()
    

    