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
        
        # Node pruning - delete any leaf node that does not appear in Q or e
        node_deleted = True
        while node_deleted:
            node_deleted = False
            for node in self.bn.get_all_variables():
                if len(self.bn.out_edges(node)) == 0 and (node not in [*Q, *e]):
                    self.bn.del_var(node)
                    node_deleted = True
                    break
