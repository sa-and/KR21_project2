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
    def pruning(self, values: Dict[str, bool]) -> BayesNet:
        """ Prune the network, will drop variables that are not needed anymore

        Args:
            values (Dict): The given evedince to a node, must be True or False. Structure: {Node:Value}

        Returns:
            p: BayesNet.object, returns the new pruned network
        """
        print("pruning")
        p = deepcopy(self.bn)
        finished = True # Continue checking if the pruning can continue
        all_cpts = p.get_all_cpts() # A dictionary of all cps in the network indexed by the variable they belong to
        while finished: # Stop pruning when the network cannot be pruned further
            finished = False
            for a, b in values.items(): # Get node name and value
                for variable in p.get_all_variables(): # Get all variables
                    cpt = all_cpts[variable]
                    if a in cpt.columns: # If variable from input matches a variable from a cpt column:
                        new_cpt = cpt.drop(cpt[cpt[a] != b].index) # If the same node exist with the opposite value, delete it
                        p.update_cpt(variable, new_cpt) # Update the cpt table without the opposite value
                    else:
                        continue
                for child in p.get_children(a): # Check if the removed node has children, if yes delete edge
                    p.del_edge((a, child)) # Delete edge if input variable has child
                    finished = True
        print(p.get_all_cpts()," dit is p")
        return p
