from typing import Union
from BayesNet import BayesNet
import pandas as pd

# I guess dat dit werkt, alleen weet ik niet helemaal hoe ik met bNRReasoner moet werken dus de structuur klopt niet helemaal...
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

    # TODO: This is where your methods should go
    # print("hi")
    def Network_Pruning(self, Q, evidence):

        #Edge Pruning
        children = dict()
        e = list()

        #Get edges between nodes by asking for children, which are save in a list (children)
        #Adds the evidence to list e such that later on, these nodes will not be removed from the BN
        for u, v in evidence.items():
            children[u] = (BayesNet.get_children(self.bn, u))
            e.append(u)    

        #Remove edges between evidence and children
        #Replaces the factors/cpt to the reduced factors/cpt 
        for key in children:
            for value in children[key]:
                BayesNet.del_edge(self.bn,(key,value))
                BayesNet.update_cpt(self.bn, value, BayesNet.reduce_factor(evidence, BayesNet.get_cpt(self.bn, value)))
    
        #Node Pruning
        #Need to keep removing leafnodes untill all leafnodes that can be removed are removed
        i = 1
        while i > 0:
            i = 0
            var = BayesNet.get_all_variables (self.bn)
            for v in var:
                child = BayesNet.get_children(self.bn, v)
                #If node is a leaf node and not in the Q or e, remove from bn
                if len(child) == 0 and v not in Q and v not in e:
                    BayesNet.del_var(self.bn, v)                
                    i += 1  


bnreasoner = BNReasoner("testing/lecture_example.BIFXML")
# BayesNet.draw_structure(bnreasoner.bn)
# print(BayesNet.get_all_cpts(bnreasoner.bn))
# print(BayesNet.get_all_variables(bnreasoner.bn))

Queri, evidence = ["Rain?"], {"Winter?": True}
#Is needed for pd.Series
e = []
for k in evidence:
    e.append(k)
bnreasoner.Network_Pruning(Queri, pd.Series(data= evidence, index = e))
# BayesNet.draw_structure(bnreasoner.bn)
# print(BayesNet.get_all_cpts(bnreasoner.bn))
# print(BayesNet.get_all_variables(bnreasoner.bn))


    
