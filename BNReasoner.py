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
    print("hi")

def run():
    # bayesnet = BNReasoner("testing/lecture_example.BIFXML") 
    bn = BayesNet()
    bn.load_from_bifxml("testing/lecture_example.BIFXML")

    # Obtain children
    children = bn.get_children('Rain?')

    for child in children:
        # Obtain child node
        cpt = bn.get_cpt(child)

        # Delete edge
        bn.del_edge(('Rain?',child))

        # Get new CPT, given the evidence
        new_cpt = bn.get_compatible_instantiations_table(pd.Series({'Rain?': False}),cpt)

        # Delete the column with evidence

        print(new_cpt)
    


    # print(bn.get_cpt(children[0]))
    # print(bn.get_cpt(children[1]))






    # bn.del_edge(self, edge):
    #     """
    #     Delete an edge form the structure of the BN.
    #     :param edge: Edge to be deleted (e.g. ('A', 'B')).
    #     """

if __name__ == "__main__":
    print("Start run ------------------------------------")
    run()
    print("End run --------------------------------------")