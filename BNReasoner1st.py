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



    def  NetworkPrune(self,query:list, evidence: pd.Series,) -> BayesNet:
        """ 
        Edge-prunes and iteratively Node-prunes the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated.
        :param query:    a list of variables (str) containing the query
        :param evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :returns:        The pruned version of the network w.r.t the query and evidence given.
        """
        var_names = evidence.index.values
        
        # Performs edge pruning
        for var in var_names:
            childs = self.bn.get_children(var)
            for child in childs:
                # Removes edges
                self.bn.del_edge((var,child))
                # Updates CPTs
                self.bn.update_cpt(child,self.bn.get_compatible_instantiations_table(evidence,self.bn.get_cpt(child)))

        # Performs node pruning
        union = list(var_names) + query
        options = self.bn.get_all_variables()

        for var in union:       # We only consider nodes that are neither in Q nor in e
            options.remove(var)

        done = False
        while not done:
            done = True
            for var in options:
                childs = self.bn.get_children(var)
                if childs == []:        # If there are still leaf nodes, we delete them and iter one more time
                    self.bn.del_var(var)
                    options.remove(var)
                    done = False

a = BNReasoner('testing/dog_problem.bifxml')
print(a.bn.get_children("light-on"))
a.NetworkPrune(["dog-out","family-out"],pd.Series({"bowel-problem":False}))
a.bn.draw_structure()
print(a.bn.get_cpt("dog-out"))

# a = BNReasoner('testing/lecture_example2.bifxml').bn
# a.draw_structure()
# a.del_var("X")
# a.draw_structure()