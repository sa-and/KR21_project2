from BNReasoner import BNReasoner
from BayesNet import BayesNet

if __name__ == "__main__":
    print("begin")
    net = "testing/dog_problem.BIFXML"
    bn = BayesNet()
    bn.load_from_bifxml(net)
    bnr = BNReasoner(bn)
    bn.draw_structure()
