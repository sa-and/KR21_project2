from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx

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
        cpts = self.bn.get_all_cpts()

        # remove edges
        for node in cpts.keys():
            for ev in [ev for ev in evidence if ev in cpts[node].keys() and ev != list(cpts[node].keys())[-2]]: # Make sure the evidence is in the node and that the node is not the evidence itself
                new_cpt = cpts[node][cpts[node][ev] == evidence[ev]]
                new_cpt = new_cpt.drop(ev,axis=1)
                self.bn.update_cpt(node, new_cpt)
                # Delete the edge itself
                self.bn.del_edge((ev,list(cpts[node].keys())[-2]))
                # Now also remove the node
                if len(self.bn.get_children(node)) == 0 and len(new_cpt[node].keys()) == 2 and len(self.bn.get_parents(node)) == 0:
                    self.bn.del_var(node)
                if len(self.bn.get_children(ev)) == 0 and len(self.bn.get_parents(ev)) == 0:
                    self.bn.del_var(ev)
        self.bn.draw_structure()

    def reduceNet(self, evidence=dict()):
        cpts = self.bn.get_all_cpts()
        for node in cpts.keys():
            if sum([1 if ev in cpts[node].keys() else 0 for ev in evidence]) >= 1:
                newCPT = self.bn.reduce_factor(pd.Series(evidence),cpts[node])
                newCPT = newCPT[newCPT.p != 0]
                self.bn.update_cpt(node, newCPT)

    def dSeperation(self, x=list(), y=list(), z=list()):
        structure = self.bn.structure
        edges = structure.edges
        nodes = structure.nodes
        filter_ = "bowel-problem"
        print([node+":"+str(len(self.bn.get_children(node))+len(self.bn.get_parents(node))) for node in self.bn.structure.nodes])
        print(edges)
        for node in lst:
            for cpt in self.bn.get_all_cpts():
                self.bn.update_cpt(node, maxingout(node))
        # print(self.bn.get_parents(x[0]))
        # for node in x:
        #     pass
            # print(list(self.bn.get_parent(node)))

        
        # print(list(nx.all_simple_paths(self.bn.get_interaction_graph(),source=x[0],target=y[0])))
        # print(edges,nodes)
        # active_paths = False
        # possible_edges = [edge for edge in edges if sum([1 if e in x or e in y else 0 for e in edge]) >= 1]
        # for edge in possible_edges:

        # print(possible_paths)           

reasoner = BNReasoner("./testing/dog_problem.BIFXML")
reasoner.pruneNetwork(evidence={"family-out": False})
# reasoner.dSeperation(["hear-bark"],["family-out"],["light-on"])