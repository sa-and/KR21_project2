from typing import Union
from BayesNet import BayesNet
import pandas as pd


class BNReasoner_:
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
    # print("hi")
    def Network_Pruning(self, bn, Q, e, evidence=pd.Series(dtype=object)):
        '''
        :param bn: Bayesnetwork
        :param Q: (query) list of variables, in case of d-separation the var of which you want to know whether they are d-separated
        :param e: the set of evidence, is list of (evidence) variables or when used to check whether X and Y are d-separation by Z, is this Z
        :param evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False}), containing the assignments of the evidence. 
                It is not obligated, therefor this function is usable for d-separation, without knowing the assignment of the Z
        :return: pruned network
        '''

        children = dict()

        #Get edges between nodes by asking for children, which are saved in a list (children)
        for u in e:
            children[u] = (BayesNet.get_children(bn, u)) 
        
        #Edge Pruning  
        #Remove edges between evidence and children
        #if evidence has an assignment: Replace the factors/cpt to the reduced factors/cpt 
        for key in children:
            for value in children[key]:
                BayesNet.del_edge(bn,(key,value))
                if not evidence.empty:
                    BayesNet.update_cpt(bn, value, BayesNet.reduce_factor(evidence, BayesNet.get_cpt(bn, value)))
        
        #Node Pruning
        #Need to keep removing leafnodes untill all leafnodes that can be removed are removed
        i = 1
        while i > 0:
            i = 0
            var = BayesNet.get_all_variables (bn)
            for v in var:
                child = BayesNet.get_children(bn, v)
                #If node is a leafnode and not in the Q or e, remove from bn
                if len(child) == 0 and v not in Q and v not in e:
                    BayesNet.del_var(bn, v)                
                    i += 1    
        
        
    def d_separation(self, bn, X, Y, Z):
        '''
        :param bn: Bayes network
        :param X, Y, Z: sets of var (lists) of which you want to know whether they are d-separated by Z
        :returns: True if X and Y are d-separated, False if they are not d-separated by Z
        '''        
        q = X.union(Y)
        self.Network_Pruning(bn, q, Z)
        
        #If there is no path between X and Y (order does not matter) means X and Y are d-separated
        for x in X:
            for y in Y:
                if self.loop_over_children(bn, y, x):
                    if self.loop_over_children(bn, x, y):
                        return True
        return False
     
    def loop_over_children(self,bn, y, parent):
        children = BayesNet.get_children(bn, parent)
        if len(children) == 0:
            return True
        else:
            for child in children:
                if child == y:
                    return False
                else:
                    return self.loop_over_children(bn, y, child)    


Pruning = False
check_d_separation = False

#We need to make new BN for d-separation and pruning as the bn is adjusted in these functions

if Pruning:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Queri, evidence = {"Winter?"}, {"Rain?": True}
    #Is needed for pd.Series
    e = set()
    for k in evidence:
        e.add(k)
    bnreasoner.Network_Pruning(bnreasoner.bn, Queri, e, pd.Series(data= evidence, index = e))

#determine whether X is d-seperated from Y by Z
if check_d_separation:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Y = {"Winter?"}
    X = {"Wet Grass?"}
    Z = {"Rain?"} 
    if bnreasoner.d_separation(bnreasoner.bn, X,Y,Z):
        print(X, "is d-separated from ", Y, "by ", Z)
    else:
        print(X, "is not d-separated from ", Y, "by ", Z)

BayesNet.draw_structure(bnreasoner.bn)



    
