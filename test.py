from BNReasoner import BNReasoner
from BayesNet import BayesNet

if __name__ == "__main__":
    print("begin")
    net = 'testing/lecture_example.BIFXML'
    bn = BayesNet()
    bn.load_from_bifxml(net)
    # bn.draw_structure()
    bnr = BNReasoner(bn)
    #outcome = bnr.pruning(["Wet Grass?"],{'Rain?': False,'Winter?':True})


    ### D-seperation
    #outcome_d_not = bnr.d_separation(["family-out"], ["hear-bark"], ["light-on"]) # Not seperated
    #outcome_d = bnr.d_separation(["family-out"], ["bowel-problem"], ["light-on"]) # Seperated


    ### Order min-degree
    #outcome_min_degree = bnr.min_degree()
    #print(outcome_min_degree)

    ### Order min-degree
    # outcome_min_fill = bnr.min_fill()
    # print(outcome_min_fill)

    #outcome.draw_structure()

    X = "Wet Grass?"
    Y = BayesNet.get_cpt(bnr.bn,X)
    # f = bnr.multip_factors(Y)
    print(Y)
    outcome = bnr.sum_out_factors(Y,X) #bnr.maximise_out(Y,X)
    print(outcome)
    #python3 test.py
