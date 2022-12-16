from BNReasoner import BNReasoner
from BayesNet import BayesNet
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("begin")

    net = 'testing/use_case.BIFXML'
    bn = BayesNet()
    bn.load_from_bifxml(net)
    bnr = BNReasoner(bn)


    Marginal_distribution = True
    checkMAP = False
    checkMEP = False

    
    ### Marginal distribution, a-priori for stroke or bleed
    if Marginal_distribution:
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        #outcome_md = bnr.compute_marginal(['Stroke'], order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Stroke history': True, "High blood pressure": True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Anticoagulant': True, "High blood pressure": True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Diabetes': True, "70+": True, "Liver/kidney disease":True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Diabetes': True, "70+": True, "Anticoagulant":True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Smoking': True, "70+": True, "Hyperlipidemia":True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Smoking': True, "70+": True, "Diabetes":True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Smoking': True, "Blood thick": True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Stroke history': True, "Hyperlipidemia": True}), order=bnr.min_degree())
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Liver/kidney disease': True, "Stroke history": True}), order=bnr.min_degree())
        outcome_md = bnr.compute_marginal(['Internal Bleeding'], order=bnr.min_degree())
        print(outcome_md)
    '''

    ### Marginal distribution, apostpriori for stroke or bleed
    if Marginal_distribution:
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        #outcome_md = bnr.compute_marginal(['Stroke'], pd.Series({'Stroke history': True, "High blood pressure": True}), order=bnr.min_degree())
        outcome_md = bnr.compute_marginal(['Internal Bleeding'], pd.Series({'Anticoagulant':True, "Diabetes":True}), order=bnr.min_degree())
        print(outcome_md)
'''

    ### MAP
    if checkMAP:
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_map = bnr.MAP(["70+", "Smoker"], pd.Series({'Stroke history': True, "High blood pressure": True, "Hyperlipidemie":True}))
        print(outcome_map)


    ### MPE
    if checkMEP:
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        outcome_mpe = bnr.MPE(pd.Series({'Stroke': True, "High blood pressure":True}))
        print(outcome_mpe.to_string())
