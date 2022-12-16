from BNReasoner import BNReasoner
from BayesNet import BayesNet
import pandas as pd

if __name__ == "__main__":
    print("begin")

    net = 'testing/lecture_example.BIFXML'

    #net = 'testing/lecture_example.BIFXML'

    bn = BayesNet()
    bn.load_from_bifxml(net)
    # bn.draw_structure()
    bnr = BNReasoner(bn)

    ### Test pruning
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


    ### Factor Multiplication
    # net = 'testing/lecture_example.BIFXML'
    # outcome_factor = bnr.factor_multiplication(['Rain?', 'Wet Grass?'])
    # print(outcome_factor)

    ### MPE
    # net = 'testing/lecture_example2.BIFXML'
    # mpe = bnr.MPE(pd.Series({'J': True, 'O': False}))
    # print(mpe)


    ### MAP
    # map = bnr.MAP(['I', 'J'], pd.Series({'O': True}))
    # print(map)



    # outcome.draw_structure()

    # X = "Wet Grass?"
    # Y = BayesNet.get_cpt(bnr.bn,X)
    # # f = bnr.multip_factors(Y)
    # print(Y)
    # outcome = bnr.sum_out_factors(Y,X) #bnr.maximise_out(Y,X)
    # print(outcome)
    
    ### Elimination
    # outcome_elim = bnr.elimination(bn.get_cpt('Wet Grass?'), ['Rain?'])
    #outcome_elim = bnr.elimination(bn.get_cpt('Winter?'), ['Winter?']) # Empty Dataframe
    # print(outcome_elim)

    ### Independence
    Y = {"Winter?"}
    X = {"Slippery Road?"}
    Z = {} 
    if bnr.independence(bnr.bn, X,Y,Z):
        print(X, "is independent from ", Y, "given ", Z)
    else:
        print(X, "is not independent from ", Y, "given ", Z)

    #python3 test.py