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

    def  𝑝𝑟𝑢𝑛𝑒Edges(self, 𝑒vidence: pd.Series,):
        """
        -param evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        """

        var_names = evidence.index.values
        for var in var_names:
            childs = self.bn.get_children(var)
            for child in childs:
                # Removes edges
                self.bn.del_edge((var,child))
                # Updates CPTs
                self.bn.update_cpt(child,self.bn.get_compatible_instantiations_table(evidence,self.bn.get_cpt(child)))

a = BNReasoner('testing/dog_problem.bifxml')
# a = BNReasoner('testing/lecture_example2.bifxml').bn
a.pruneEdges(pd.Series({"bowel-problem":False}))
a.bn.draw_structure()
print(a.bn.get_cpt("dog-out"))