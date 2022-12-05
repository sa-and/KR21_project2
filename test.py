from BNReasoner import BNReasoner
from BayesNet import BayesNet

if __name__ == "__main__":
    print("begin")
    net = 'testing/lecture_example.BIFXML' #'testing/dog_problem.BIFXML'
    bn = BayesNet()
    bn.load_from_bifxml(net)
    #bn.draw_structure()
    bnr = BNReasoner(bn)
    outcome = bnr.pruning(["Wet Grass?"],{'Rain?': False,'Winter?':True})
    
    ### D-seperation, use dog example
    #outcome_d_not = bnr.d_separation(["family-out"], ["hear-bark"], ["light-on"]) # Not seperated
    #outcome_d = bnr.d_separation(["family-out"], ["bowel-problem"], ["light-on"]) # Seperated  

    #outcome.draw_structure()


    #python3 test.py
