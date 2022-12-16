from BNReasoner import BNReasoner
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    Pruning = True
    check_d_separation = True 
    Independence = True 
    Marginalization = True
    MaxingOut = True
    FactorMultiplication = True
    Ordering_min_degree = True
    Ordering_min_fill = True
    Elimination = True
    Marginal_distribution = True
    checkMAP = True
    checkMEP = True

    ### Test pruning
    if Pruning:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        print("#----------------------------------------Pruning---------------------------------------#")
        outcome = bnr.pruning(["Wet Grass?"],{'Rain?': False,'Winter?':True})
        print(outcome)

    ### D-seperation
    if check_d_separation:
        net = 'testing/dog_problem.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        #outcome_d_not = bnr.d_separation(["family-out"], ["hear-bark"], ["light-on"]) # Not seperated
        print("#-----------------------------------------D-seperation---------------------------------#")
        outcome_d = bnr.d_separation(["family-out"], ["bowel-problem"], ["light-on"]) # Seperated

        print(outcome_d)
   
    ## Test independence
    if Independence:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        Y = {"Winter?"}
        X = {"Slippery Road?"}
        Z = {} 
        if bnr.check_independence(bnr.bn, X,Y,Z):
            print("#--------------------------------Independence--------------------------------------#")
            print(X, "is independent from ", Y, "given ", Z)
        else:
            print("#--------------------------------Independence--------------------------------------#")
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
        print("#-------------------------------Marginalization--------------------------------------#")
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
        print("#-------------------------------Maxing out-------------------------------------------#")
        print(outcome_max)
        
    ### Test factor multiplication
    if FactorMultiplication:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()     
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_factor = bnr.factor_multiplication(['Rain?', 'Wet Grass?'])
        print("#-------------------------------Factor multiplication-------------------------------#")
        print(outcome_factor)

    ### Test ordering min   
    if Ordering_min_degree:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_min_degree = bnr.min_degree()
        print("#-------------------------------Ordering min degree----------------------------------#")
        print(outcome_min_degree)


    ### Test ordering min fill
    if Ordering_min_fill:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_min_fill = bnr.min_fill()
        print("#--------------------------------Ordering min fill----------------------------------#")
        print(outcome_min_fill)

    ### Elimination
    if Elimination:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_elim = bnr.elimination(bn.get_cpt('Wet Grass?'), ['Rain?'])
        print("#-------------------------------Elimination---------------------------------------#")
        print(outcome_elim)

    ### Marginal distribution
    if Marginal_distribution:
        net = 'testing/lecture_example.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_md = bnr.compute_marginal(['Wet Grass?', 'Slippery Road?'], order=bnr.min_degree())
        print("#-------------------------------Marginal distribution-----------------------------#")
        print(outcome_md)

    ### MAP
    if checkMAP:
        net = 'testing/lecture_example2.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_map = bnr.MAP(['I', 'J'], pd.Series({'O': True}))
        print("#-------------------------------------MAP-----------------------------------------#")
        print(outcome_map)


    ## MPE
    if checkMEP:
        net = 'testing/lecture_example2.BIFXML'
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(net)
        outcome_mpe = bnr.MPE(pd.Series({'J': True, 'O': False}))
        print("#-------------------------------------MPE-----------------------------------------#")
        print(outcome_mpe)

