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
    # TODO: This is where your methods should go

def prune(net, q, e):
    node_prune(net, q, e)
    edge_prune(net, q, e)
    BayesNet.draw_structure(net)
    return net

def edge_prune(net, q, e): #TODO Update Factors see Bayes 3 slides page 28
    for node in e:
        edges = net.get_children(node)
        for edge in edges:
            net.del_edge([node, edge])
    return net

    
def node_prune(net, q, e): #Performs Node Pruning given query q and evidence e
    for node in BayesNet.get_all_variables(net):
        if node not in q and node not in e:
            net.del_var(node)
    return net


#def d-blocked(x,y,z):
    
#    for w in x:





net = BNReasoner("C:/Users/Bart/Documents/GitHub/KR21_project2/testing/dog_problem.BIFXML")
prune(net.bn, [], ['bowel-problem'])
