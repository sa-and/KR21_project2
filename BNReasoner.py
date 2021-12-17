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
        
    def dSeperator(self, network, x,z,y):
        ## here we check for d-seperation
        return
    
    def ordering(self, network, x):
        ## order x based on min-fill and min-degree heuristics
        return
    
    def networkPruning(self, network, q, e): ## set of variables Q and evidence E
        ##
        return
    
    def marginalDistributions(self, network, q, e):
        return
    
    def MAP(self, network, q, e):
        return
    
    def MPE(self, network, q, e):
        return

    # TODO: This is where your methods should go
