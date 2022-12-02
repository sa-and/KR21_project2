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

    def pruneNetwork(self, evidence=dict()):
        print(evidence)
        cpts = self.bn.get_all_cpts()
        initiations = pd.Series(evidence)
        # self.bn.draw_structure()

        # First reduce factors
        for node in cpts.keys():
            if sum([1 if ev in cpts[node].keys() else 0 for ev in evidence]) >= 1:
                newCPT = self.bn.reduce_factor(initiations,cpts[node])
                newCPT = newCPT[newCPT.p != 0]
                self.bn.update_cpt(node, newCPT)

        # Then prune the edges and nodes
        for ev in evidence.keys():
            for child in self.bn.get_children(ev):
                self.bn.del_edge((ev,child))
                newCPT = self.bn.get_cpt(child).drop(ev,axis=1)
                self.bn.update_cpt(child, newCPT)
                if newCPT.drop('p',axis=1).columns.shape[0] <= 1:
                    self.bn.del_var(child)
            self.bn.del_var(ev)
        self.bn.draw_structure()

reasoner = BNReasoner("./testing/dog_problem.BIFXML")
cpts = reasoner.bn.get_all_cpts()
#reasoner.bn.draw_structure()
variable = 'dog-out'
#cpts['dog-out'].columns.drop(['dog-out','p'])
df = cpts[variable]
res = pd.DataFrame(columns = df.columns.drop([variable]))

for i in range(len(df['family-out'])):
    if i % 2 == 0:
        max = df.loc[i,'p']
    else:
        if df.loc[i,'p'] > max:
            max = df.loc[i,'p']
        print(max)
        maxres = df.drop([variable,'p'], axis = 1).loc[i,:]
        maxres['p'] = max
        print(maxres)
        res = pd.concat([res,pd.DataFrame(maxres)], axis = 0)
