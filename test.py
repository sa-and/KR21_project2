from BNReasoner import BNReasoner
from BayesNet import BayesNet
import pandas as pd

if __name__ == "__main__":
    print("begin")

    # net = 'testing/lecture_example2.BIFXML'

    net = 'testing/lecture_example.BIFXML'
    bn = BayesNet()
    bn.load_from_bifxml(net)
    bnr = BNReasoner(bn)

    Pruning = True
    check_d_separation = False 
    Independence = False 
    Marginalization = False
    MaxingOut = False
    FactorMultiplication = False
    Ordering_min_degree = False
    Ordering_min_fill = False
    Elimination = False
    Marginal_distribution = False
    checkMAP = False
    checkMEP = False

    ### Test pruning
    if Pruning:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome = bnr.pruning(["Wet Grass?"],{'Rain?': False,'Winter?':True})
        print(outcome)

    ### D-seperation
    if check_d_separation:
        net = 'testing/dog_problem.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        #outcome_d_not = bnr.d_separation(["family-out"], ["hear-bark"], ["light-on"]) # Not seperated
        outcome_d = bnr.d_separation(["family-out"], ["bowel-problem"], ["light-on"]) # Seperated
        print(outcome_d)
   
    ### Test independence
    if Independence:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        Y = {"Winter?"}
        X = {"Slippery Road?"}
        Z = {} 
        if bnr.independence(bnr.bn, X,Y,Z):
            print(X, "is independent from ", Y, "given ", Z)
        else:
            print(X, "is not independent from ", Y, "given ", Z)

    ### Test marginalization
    if Marginalization:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        X = "Wet Grass?"
        Y = BayesNet.get_cpt(bnr.bn,X)
        outcome_marg = bnr.sum_out_factors(Y,X)
        print(outcome_marg)

    ### Test maxing out
    if MaxingOut:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        X = "Wet Grass?"
        Y = BayesNet.get_cpt(bnr.bn,X)
        outcome_max = bnr.maximise_out(Y,X)
        print(outcome_max)
        
    ### Test factor multiplication
    if FactorMultiplication:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()     
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_factor = bnr.factor_multiplication(['Rain?', 'Wet Grass?'])
        print(outcome_factor)

    ### Test ordering min   
    if Ordering_min_degree:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_min_degree = bnr.min_degree()
        print(outcome_min_degree)

    ### Test ordering min fill
    if Ordering_min_fill:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_min_fill = bnr.min_fill()
        print(outcome_min_fill)

    ### Elimination
    if Elimination:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_elim = bnr.elimination(bn.get_cpt('Wet Grass?'), ['Rain?'])
        print(outcome_elim)

    ### Marginal distribution
    if Marginal_distribution:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_md = bnr.compute_marginal(['Wet Grass?', 'Slippery Road?'], order=bnr.min_degree())
        print(outcome_md)

    ### MAP
    if checkMAP:
        net = 'testing/lecture_example2.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_map = bnr.MAP(['I', 'J'], pd.Series({'O': True}))
        print(outcome_map)

    ### MPE
    if checkMEP:
        net = 'testing/lecture_example2.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_mpe = bnr.MPE(pd.Series({'J': True, 'O': False}))
        print(outcome_mpe)

    # print(outcome)

    #python3 test.py
