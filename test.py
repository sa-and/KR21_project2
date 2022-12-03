from BNReasoner import BNReasoner
from BayesNet import BayesNet

if __name__ == "__main__":
    print("begin")
    net = 'testing/lecture_example.BIFXML'
    bn = BayesNet()
    bn.load_from_bifxml(net)
    #bn.draw_structure()
    bnr = BNReasoner(bn)
    outcome = bnr.pruning(["Wet Grass?"],{'Rain?': False,'Winter?':True})
    #outcome.draw_structure()


    #python3 test.py
