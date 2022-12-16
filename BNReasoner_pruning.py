from typing import Union
from BayesNet import BayesNet
import pandas as pd
from copy import deepcopy
import random
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("Finished importing")

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
    def Network_Pruning(self, Q, e):
        '''
        Given a set of query variables Q and evidence e, node- and edge-prunes the Bayesian network s.t. queries of the form P(Q|E) can
            still be correctly calculated

        :param Q: list of (query) variables; in case of d-separation it contains the variables of which you want to know whether they are d-separated
        :param e: the set of evidence variables, which is a dictionary of assignments as {"A": True, "B": False}, 
                containing the assignments of the evidence; or when used to check whether X and Y are d-separation by Z, is this a list containing all variables in Z
        :return: pruned network
        '''       
        if type(e) is dict():
            evidence_set = e.keys()
        else:
            evidence_set = e
        new_bn = deepcopy(self.bn)
        children = dict()

        #Get edges between nodes by asking for children, which are saved in a list (children)
        for u in e:
            children[u] = BayesNet.get_children(new_bn, u)
        
        #Edge Pruning  
        #Remove edges between evidence and children
        #if evidence has an assignment: Replace the factors/cpt to the reduced factors/cpt 
        for key in children:
            for value in children[key]:
                BayesNet.del_edge(new_bn,(key,value))
                if type(e) is dict:
                    BayesNet.update_cpt(new_bn, value, BayesNet.get_compatible_instantiations_table(pd.Series(e), BayesNet.get_cpt(new_bn, value)))
        
        #Node Pruning
        #Need to keep removing leafnodes untill all leafnodes that can be removed are removed
        i = 1
        while i > 0:
            i = 0
            var = BayesNet.get_all_variables (new_bn)
            for v in var:
                child = BayesNet.get_children(new_bn, v)
                #If node is a leafnode and not in the Q or e, remove from bn
                if len(child) == 0 and v not in Q and v not in evidence_set:
                    BayesNet.del_var(new_bn, v)                
                    i += 1    
        return new_bn  

    def d_separation(self, X, Y, Z):
        '''
         Given three sets of variables X, Y, and Z, determines
            whether X is d-separated of Y given Z

        :param X, Y, Z: lists of var of which you want to know whether X and Y are d-separated by Z
        :returns: True if X and Y are d-separated, False if they are not d-separated by Z
        '''        
        q = X + Y
        bn = self.Network_Pruning(q, Z)
        
        #If there is no path between X and Y means X and Y are d-separated
        for x in X:
            for y in Y:
                if self.loop_over_children(bn, y, x):
                    if self.loop_over_children(bn, x, y):
                        return True
        return False
     
    def loop_over_children(self,bn, y, parent):
        '''
        Checks if y is a descendant of parent

        :param bn: a BN
        :param y: a variable of the bn
        :param parent: a variable in the bn
        :returns: True, if there is no path between parent and y, False, otherwise
        '''

        children = BayesNet.get_children(bn, parent)
        if len(children) == 0:
            return True
        else:
            for child in children:
                if child == y:
                    return False
                else:
                    if not self.loop_over_children(bn, y, child):
                        return False
            return True    

    def independence(self, X, Y, Z):
        '''
        Given three sets of variables X, Y, and Z, determines
            whether X is independent of Y given Z

        Implementation of Markov Property and Symmetry in DAGS to determine independence
        :param X, Y, Z: lists of variable of which to decide whether X is independent of Y given Z
        :returns: Bool, True if X and Y are independent given Z, False if X and Y are not independent given Z 
        '''
        Not_all_parents_of_X = False        
        for x in X: 
            for parent in BayesNet.get_all_variables(self.bn):
                if x in BayesNet.get_children(self.bn, parent):
                    if parent not in Z:
                        Not_all_parents_of_X = True
                        break
            if Not_all_parents_of_X:
                break

        if Not_all_parents_of_X:
            for y in Y: 
                for parent in BayesNet.get_all_variables(self.bn):
                    if y in BayesNet.get_children(self.bn, parent):
                        if parent not in Z:
                            return False 
            for y in Y:
                for x in X:
                    if not self.loop_over_children(self.bn, x, y):
                        return False
            return True

        for x in X:
            for y in Y:
                if not self.loop_over_children(self.bn, y, x):
                    return False
                     
        return True 

    def marginalization(self,cpt, X):
        '''
        Given a factor and a variable X, computes the CPT in which X is summed-out
        :param cpt: a factor
        :param X: variable X
        :returns: the CPT in which X is summed-out
        '''
        newcpt = deepcopy(cpt)
        if X in list(newcpt.columns):
            newcpt = cpt.drop([X],axis=1)
        
        remaining_vars = [x for x in newcpt.columns if x != X and x != 'p'] 
          
        if len(list(newcpt.columns)) == 1:
            p = newcpt["p"].sum()
            newcpt = pd.DataFrame({" ": ["T"], "p":[p]})
        else:
            newcpt = newcpt.groupby(remaining_vars).agg({'p': 'sum'})        
            newcpt.reset_index(inplace=True)

        return newcpt

    def maxing_out(self,cpt,X): 
        '''
        Given a factor and a variable X, computes the CPT in which X is maxed-out
        :param cpt: a factor
        :param X: variable X
        :returns: the CPT in which X is maxed-out, including the extended factor
        '''
        
        newcpt = deepcopy(cpt)
        
        remaining_vars = [x for x in newcpt.columns if x != X and x != 'p' and x != 'extended factor']

        if len(remaining_vars) != 0:
            newcpt = newcpt.loc[newcpt.groupby(remaining_vars)['p'].idxmax()]
        else:
            newcpt = newcpt.loc[newcpt["p"].round(10) == max(newcpt['p'].round(10))]
        
        if 'extended factor' not in newcpt.columns:
            newcpt['extended factor'] = X + ': ' + newcpt[X].astype(str)
        else:            
            newcpt['extended factor'] = newcpt['extended factor'].astype(str) + ', ' + X + ': ' + newcpt[X].astype(str)
            

        if X in list(newcpt.columns):
            newcpt = newcpt.drop([X],axis=1)

        return newcpt

    def multiply_factors(self,f,g):
        '''
        Given two factors f and g, computes the multiplied factor h=fg
        :param f: Factor f
        :param g: Factor g
        :returns: Multiplied factor h=f*g
        '''
        # check what the overlapping var(s) is
        vars_f = [x for x in f.columns]
        vars_g = [x for x in g.columns]
        join_var = list()

        for var in vars_f:
            if var in vars_g and var != 'p':
                join_var.append(var)

        if len(join_var) != 0:
            merged = pd.merge(f,g, how="outer", on=join_var)
            # multiply probabilities
            merged['p'] = merged['p_x']*merged['p_y']
            # drop individual probability columns
            h = merged.drop(['p_x','p_y'],axis=1)
        else:
            merged = pd.merge(f,g, how="cross")
            # multiply probabilities
            merged['p'] = merged['p_x']*merged['p_y']
            # drop individual probability columns
            h = merged.drop(['p_x','p_y'],axis=1)
        return h

    def minimum_fill_ordering(self, bnreasoner, set): 
        '''
        Given a set of variables X in the Bayesian network, computes a
            good ordering for the elimination of X based on the min-fill heuristics 

        :param set: list of variables X in the Bayesian network of which an ordering needs to be returned based on minimum fill
        :returns: a list containing an ordering based on minumum fill
        '''
        order = list()
    
        #dictionary of all edges in the interaction graph
        _, edges = self.least_edges(set)
            
        #saves number of added edges for each variable when that node would be removed
        dict_added_edges = dict()
        for var in set:
            vars_edges = deepcopy(edges)
            _, added_edges = self.connect_nodes(var, vars_edges, set)
            dict_added_edges[var] = added_edges
        #picks the node with the least number of added edges
        var_to_del = sorted(dict_added_edges.items(), key=lambda item: item[1])[0][0]
        order.append(var_to_del)
            
        # remove node and return new edges 
        edges, _ = self.connect_nodes(var_to_del, edges, set)

        #for all remaining nodes pick the one that adds the least number of edges, add this to the ordering and 
        # remove this from the node from the variables that still need to be added to the ordering, and 
        # add edges if necessary 
        for i in range(len(set) - 1):
            variables = list(edges.keys())
            to_del_var = self.min_fill(variables, edges, set)
            order.append(to_del_var)
            edges, _ = self.connect_nodes(to_del_var, edges, set)
        return order

    def min_fill(self, variables, edges, set):
        '''
        :param variables: a list with variables that are not yet in the ordering, that do need to be in it
        :param edges: a dictionary containing all the edges of those variables not yet in the ordering/ 
                    still in the interaction graph
        :param set: list of all variable that need to be in the ordering
        :returns: the variable which would, when removed add the least number of edges
        '''
        dict_fill = dict()
        for var in variables: 
            vars_edges = deepcopy(edges)
            _, added_edges = self.connect_nodes(var, vars_edges, set)
            dict_fill[var] = added_edges   
        return sorted(dict_fill.items(), key=lambda item: item[1])[0][0]        
        
    def minimum_degree_ordering(self, bnreasoner, set):
        '''
        Given a set of variables X in the Bayesian network, computes a
            good ordering for the elimination of X based on the min-degree heuristics 

        :param set: list of variables X in the Bayesian network of which an ordering needs to be returned based on minimum degree
        :returns: a list containing an ordering based on minumum degree
        '''
        order = list()

        #get the var that has the least edges and a dict containing for each variable the edges in the interaction graph
        to_del_var, edges = self.least_edges(set)
        order.append(to_del_var)

        #remove var and add edges if necessary
        edges, _ = self.connect_nodes(to_del_var, edges, set)
        #while not all var are added to the ordering, pick var that has the least edges, 
        # add this to the ordering and remove the var form the to added variables, add edges where necessary
        for i in range(len(set)-1):#BayesNet.get_all_variables(bn)) - 1):
            variables = list(edges.keys())
            to_del_var = self.min_deg(variables, edges)
            order.append(to_del_var)
            edges, _ = self.connect_nodes(to_del_var, edges, set)
        return order
        
    def min_deg(self, variables, edges):
        '''
        :param variables: a list with all variables that are not yet in the ordering, but do need to be in it
        :param edges: a dictionary containing all the edges of those variables not yet in the ordering/ 
                    still in the interaction graph
        :returns: the variable with the least number of edges
        '''
        dict_degrees = dict()
        for var in variables:
            dict_degrees[var] = len(edges[var]) 
        return sorted(dict_degrees.items(), key=lambda item: item[1])[0][0]

    def least_edges(self, set):
        '''
        :param set: a list with variables that still need to be added to the ordering
        :returns: the variable with the least number of edges and a dictionary containing the edges for each variable in the set
        '''
        dict_nr_edges = dict()
        dict_edges = dict()
        for var in set:
            dict_nr_edges[var] = 0
            dict_edges[var] = list()
 
        all_var = BayesNet.get_all_variables(self.bn)
        for var in all_var:
            children = BayesNet.get_children(self.bn, var)
            if var in set:
                dict_nr_edges[var] += len(children)
            for child in children:
                if var in dict_edges:
                    dict_edges[var].append((var,child))  
                if child in set:              
                    dict_edges[child].append((child, var))
                    dict_nr_edges[child] += 1  
        return sorted(dict_nr_edges.items(), key=lambda item: item[1])[0][0], dict_edges

    def connect_nodes(self, var, edges, set):  
        '''
        adds edges between nodes if necessary when a variable is removed and removes the variable
        and count the number of added edges
        :param var: string containing the variable to be deleted
        :param edges: dict containing all edges of each variable in the parameter "set"
        :param set: a list containing all variables in the bn that need to be in the ordering, but have not been added yet
        :returns: a dict containing for all variables that are not yet in the ordering , the edges
                  in the graph, including the edges added when var is removed
                : an integer that counts the number of edges that are added when var is removed
        '''
        edge_added = 0 
                  
        for edge_1 in edges[var]:
            if edge_1[1] in set:
                edges[edge_1[1]].remove((edge_1[1], var))
            for edge_2 in edges[var]: 
                added = False                              
                if edge_1 != edge_2: 
                    if edge_2[1] in set:
                        if (edge_2[1], edge_1[1]) not in edges[edge_2[1]]:                        
                            edges[edge_2[1]].append((edge_2[1],edge_1[1]))
                            added = True
                            edge_added += 1   
                    if edge_1[1] in set:                 
                        if (edge_1[1], edge_2[1]) not in edges[edge_1[1]]:                        
                            edges[edge_1[1]].append((edge_1[1],edge_2[1]))  
                            if added == False:
                                edge_added += 1                     
        edges.pop(var)
        return edges, edge_added

    def variable_elimination(self, cpts, to_sum_out_vars, order =None):
        '''
        : Sum out a set of variables by using variable elimination.

        :param cpts: a dictionary containing the cpts 
        :param to_sum_out_vars: List of variables that need to be summed out
        :param e: a series of assignments as tuples, containing the evidence
        :param order: a string containing the type of ordering, if no order is given a random order is picked
        :return: the factor with all var in the to_sum_out_vars summed out, and when evidence is present, factors reduced. It also returns a list with the cpts that have been used
        '''
        all_vars = list(BayesNet.get_all_variables(self.bn))

        q = list()
        for var in all_vars:
            if var not in to_sum_out_vars:
                q.append(var)      

        f = deepcopy(cpts)           
        
        #If no order is assigned, the order in which the to sum out variables are given is used as ordering --> misschien random maken?
        if order is None:
            order = list()
            r = range(len(to_sum_out_vars))
            for i in r:
                in_order = random.choice(to_sum_out_vars)
                order.append(in_order)
                to_sum_out_vars.remove(in_order)
        elif order == "minimum_degree_ordering":
            order = self.minimum_degree_ordering(self.bn, to_sum_out_vars)
        else:
            order = self.minimum_fill_ordering(self.bn, to_sum_out_vars)
        # print(order)
        #do the variable elimination
        done = list()
        n = None
        for s in order:
            to_multiply = list()
            for var in f:   
                if var not in done:             
                    if s in list(f[var].columns):                    
                        done.append(var)
                        to_multiply.append(f[var])

            if n is None:             
                n = to_multiply[0]
                if len(to_multiply) > 1:
                    for factor in to_multiply[1:]:
                        n = self.multiply_factors(n, factor)

                    
            else:      
                for factor in to_multiply:
                    n = self.multiply_factors(n, factor)  

            n = self.marginalization(n, s)
        
        #Multiply by all factors
        for var in q:
            if var not in done:
                n = self.multiply_factors(n, self.bn.get_cpt(var))

        return n   

    def joint_prob(self):
        '''
        Calculares the joint probability of a BN
        :returns: a CPT containing the joint probability of the BN
        '''
        cpts = self.bn.get_all_cpts()
        i = 0
        for var in (cpts):
            if i == 0:
                n = cpts[var]
                i+=1
            else:
                n = self.multiply_factors(n, cpts[var])
        return n

    def marginal_distribution(self, Q, e={}, order = None):  
        '''
        Given query variables Q and possibly empty
        evidence e, computes the marginal distribution P(Q|e). Q is a
        subset of the variables in the Bayesian network X with Q ⊂ X, but can also
        be Q = X

        :param Q: list of query variables
        :param e: a dictionary of assignments, containing the evidence
        :param order: possibly a string containing an ordering type
        :returns: marginal distribution P(Q|e)
        '''
        
        to_sum_out = self.bn.get_all_variables()
        for var in Q:            
            to_sum_out.remove(var) 

        if len(to_sum_out) == 0:
            if len(e) == 0:          
                return self.joint_prob()
        else:
            cpt = self.bn.get_all_cpts()
            #reduce factors by evidence 
            if len(e) != 0:                
                for var in cpt:
                    cpt[var] = BayesNet.get_compatible_instantiations_table(pd.Series(e), cpt[var])
                    
            # calculate the prior marginal of Q:
            cpt = self.variable_elimination(cpt, to_sum_out, order)            
            
            if len(e) != 0: 
                Pr_e = cpt["p"].sum()
                cpt['p'] = cpt['p']/Pr_e
                #cpt is now the posterior marginal of Q

            return cpt         
    
    def map(self, q,e={}, order=None):
        '''
        Computes the maximum a-posteriory instantiation and the value of query variables Q, given a possibly empty evidence e.

        :param q: list of query variables
        :param e: a dictionary of assignments, containing the evidence
        :param order: possibly a string containing an ordering type
        :return: maximum a-posteriory instantiation, including the extended factor
        '''
        all_var = self.bn.get_all_variables()
        to_sum_out = list()
        for var in all_var:
            if var not in q:
                to_sum_out.append(var)
        cpts = self.bn.get_all_cpts()
        if len(e) != 0:                
            for var in cpts:
                cpts[var] = BayesNet.get_compatible_instantiations_table(pd.Series(e), cpts[var])

        p_Q_E = self.variable_elimination(cpts, to_sum_out, order)
        for var in q:
            p_Q_E = self.maxing_out(p_Q_E, var)
   
        return p_Q_E    

    def mpe(self, e, ordering = None):
        '''
        Computes the most probable explanation given an evidence e.

        :param e: dictionary containing the evidence and its truth assignments
        :param order: possibly a string containing an ordering type
        :return: most probable explanation, including the extended factor
        
        '''
        
        pruned_bn = self.Network_Pruning(Q=[], e=e)
        f = deepcopy(BayesNet.get_all_cpts(pruned_bn))
        
        all_var = deepcopy(BayesNet.get_all_variables(pruned_bn))
        q = list()
        for var in all_var:
            if var not in e:
                q.append(var)   

        #reduce factors by evidence 
        if len(e) != 0:
            for var in e.keys():
                f[var] = BayesNet.get_compatible_instantiations_table(pd.Series(e), f[var])   

        if ordering is None:
            order = list()
            r = range(len(all_var))
            for i in r:
                in_order = random.choice(all_var)
                order.append(in_order)
                all_var.remove(in_order)
        
        elif ordering == "minimum_degree_ordering":
            order = self.minimum_degree_ordering(pruned_bn, all_var)
        else:
            order = self.minimum_fill_ordering(pruned_bn, all_var)

        done = list()
        n = None
        for s in order:
            to_multiply = list()
            for var in f:   
                if var not in done:             
                    if s in list(f[var].columns):                    
                        done.append(var)
                        to_multiply.append(f[var])

            if n is None:            
                n = to_multiply[0]
                if len(to_multiply) > 1:
                    for factor in to_multiply[1:]:
                        n = self.multiply_factors(n, factor)
                    
            else:      
                for factor in to_multiply:
                    n = self.multiply_factors(n, factor)             
            n = self.maxing_out(n, s)
       
        n[" "] = 'T'
        p = n["p"].max()        
        n = n[n['p'] == p]

        list_order_cpt = [' ', 'p', 'extended factor'] 
        n = n[list_order_cpt]

        return n  

Pruning = False
check_d_separation = False
Independence = False
Marginalization = False
MaxingOut =  False
MultiplyFactor = False
Ordering = False
Variable_Elimination = False
Marginal_distribution = True
Map = True
Mpe = True

#file = "testing/lecture_example.BIFXML"
file = "testing/usecase.BIFXML"
#Queri, evidence = ["Wet Grass?"], {"Rain?": False, "Winter?": True}
#Queri, evidence = ['liver-biopsy'], {"hepatitis":True, "cirrhosis":False, 'metastases':True}

trials = 1000
qs = [
    ['hepatitis'],
    ['hepatitis'],
    ['genetic-predisposition-cancer'],
    ['genetic-predisposition-cancer']
]
es = [
    {"liver-cancer":True},
    {"liver-cancer":True, "liver-biopsy":False, "cirrhosis":True, "jaundice":True, "excessive-alcohol-use":True},
    {"cirrhosis":False},
    {"hepatitis":True, "jaundice":False, "excessive-alcohol-use":True, "liver-biopsy":False}
]
text_to_plot = [
    "1 related evidence",
    "all related evidences",
    "1 unrelated evidence",
    "all unrelated evidence"
]

for iter in range(len(qs)):
    Queri, evidence = qs[iter], es[iter]

    result_marginal = {'data_fill':[], 'data_degree':[]}
    result_map = {'data_fill':[], 'data_degree':[]}
    result_mpe = {'data_fill':[], 'data_degree':[]}

    for trial_number in range(trials):
        if trial_number % 10 == 0:
            print(f"Trail: {trial_number}")
        if Pruning:
            #Test case, show pruning is working as it should by showing that it returns the same (pruned) BN and CPTs as slide 31 of the third Bayes' lecture
            bnreasoner = BNReasoner_(file)
            Queri, evidence = ["Wet Grass?"], {"Rain?": False, "Winter?": True}
            return_bn = bnreasoner.Network_Pruning(Queri, evidence)
            print(return_bn.get_all_cpts())
            return_bn.draw_structure()
        if check_d_separation:
            #Test case, show d-separation is working as it should by showing that it returns the same (pruned) BN and CPTs as were calculated by hand and described in the report
            #The answer should be True/Yes/is d-separated, and the node "Wet Grass?" and the edges between "Winter?" and "Rain?" and between "Winter?" and "Sprinkler" should be deleted
            bnreasoner = BNReasoner_(file)    
            X = ["Sprinkler?"]
            Y = ["Slippery Road?"]
            Z = ["Winter?"]
            d_sep = bnreasoner.d_separation(X,Y,Z)
            if d_sep:
                print(d_sep,", ", X, "is d-separated from ", Y, "by ", Z)
            else:
                print(d_sep,", ", X, "is not d-separated from ", Y, "by ", Z)

        if Independence:
            
            bnreasoner = BNReasoner_(file)
            #Test whether “Wet Grass?” is independent from “Slippery Road?” given “Rain?". We computed by 
            #hand that this query should result in True, “Wet Grass” is independent form “Slippery Road?” given “Rain?”
            X = ["Wet Grass?"]
            Y = ["Slippery Road?"]
            Z = ["Rain?"] 
            if bnreasoner.independence(X,Y,Z):
                print(X, "is independent from ", Y, "given ", Z)
            else:
                print(X, "is not independent from ", Y, "given ", Z)

        if Marginalization:
            #Test case, the resulting CPT for "Rain?" summed-out of the CPT of "Wet Grass?" should be equal to: 
            # Sprinkler?  Wet Grass?     P
            # True        	True  	0.9+0.95 = 1.85
            # True      	False  	0.05+0.1=0.15
            # False        	True	0.8+0 = 0.80
            # False       	False  	0.2+ 1 = 1.20
            #

            bnreasoner = BNReasoner_(file)
            X = "Rain?"
            cpt = BayesNet.get_cpt(bnreasoner.bn,"Wet Grass?")
            print(bnreasoner.marginalization(cpt, X))

        if MaxingOut:
            #Test case, the resulting CPT for "Rain?" maxed-out of the CPT of "Rain?" should be equal to: 
            # Winter?	P	
            # True		0.8	Rain? = True
            # False		0.9	Rain? = False
            bnreasoner = BNReasoner_(file)
            X = "Rain?"
            cpt = BayesNet.get_cpt(bnreasoner.bn,X)
            print(bnreasoner.maxing_out(cpt,X))
        
        if MultiplyFactor:
            #To show the multiplication function works we calculate the multiplication of the factors of “Sprinkler?” and “Rain?” by hand, which resulted in: 
            # Winter?	Sprinkler? 	Rain?		P
            # True		True		True		0.16
            # True		True		False		0.04
            # True		False		True		0.64
            # True		False		False		0.16
            # False		True		True		0.075
            # False		True		False		0.675
            # False		False		True		0.025
            # False		False		False		0.225
            #This should be/is equal to the result printed, when the following you run the following:

            bnreasoner = BNReasoner_(file)
            X = 'Sprinkler?'
            Y = 'Rain?'
            cpt_1 = BayesNet.get_cpt(bnreasoner.bn, X)
            cpt_2 = BayesNet.get_cpt(bnreasoner.bn,Y)
            print(bnreasoner.multiply_factors(cpt_1,cpt_2)) 
            
        if Ordering:
            #Test case: For  both orderings the ordering should always start with “Slippery Road?” the order of the other variables 
            #does not matter, are all valid, as after “Slippery Road?” is removed, they all have 2 edges and would all add one edge when removed.  
            bnreasoner = BNReasoner_(file)
            Set = bnreasoner.bn.get_all_variables()
            print(bnreasoner.minimum_degree_ordering( Set))
            print(bnreasoner.minimum_fill_ordering(Set))

        if Variable_Elimination:
            #Test case: The returned CPT should be equivalent to the by hand calculated CPT on which variable elimination is applied. Eliminating the variables:
            #"Winter?", "Rain?", "Wet Grass?" and "Sprinkler?”. This resulted in the hand calculated CPT:
            #“Slippery Road?”	p
            # True			0.364
            # False			0.636
            bnreasoner = BNReasoner_(file)
            Set = ["Winter?", "Rain?", "Wet Grass?","Sprinkler?"]
            print(bnreasoner.variable_elimination(bnreasoner.bn.get_all_cpts(), to_sum_out_vars=Set))
            
        if Marginal_distribution:
            #Test case: With Q = “Winter?” and “Rain?”, and with evidence e = “Sprinkler?” : True, the hand calculated CPT that needs to be returned is for the prior marginal: 
            # Rain?  Winter?     P
            # True     True  0.48
            # False     True  0.12
            # True    False  0.04
            # False    False  0.36 
            # and for the posterior marginal:
            # Winter?  Rain?         P
            # True   	True 	0.228571
            # True 	False  	0.057143
            # False   	True  	0.071429
            # False  	False  	0.642857
            #Which should be equal to the CPTs that are returned below

            bnreasoner = BNReasoner_(file)
            #q = ["Winter?", "Rain?"]
            #evidence = {"Sprinkler?": True}
            current_time = time.time()
            marg_evi = bnreasoner.marginal_distribution(Queri, evidence, order='minimum_degree_ordering')
            elapsed_time = time.time()
            result_marginal['data_degree'].append(elapsed_time - current_time)
            
            current_time = time.time()
            marg_evi = bnreasoner.marginal_distribution(Queri, evidence, order='minimum_fill_ordering')
            elapsed_time = time.time()
            result_marginal['data_fill'].append(elapsed_time - current_time)
            
            #marg_q = bnreasoner.marginal_distribution(Queri)

            #print("Posterior Marginal:")
            #print(bnreasoner.marginal_distribution(q,evidence))
            #print("Prior Marginal:")
            #print(bnreasoner.marginal_distribution(q))
            
        if Map:
            # Test case: Given the BN, which can also be found in the provided lecture_example2.BIFXML and the query 
            # given in example 2 on slides 20-21 of the fourth Bayes’ lecture. Our implementation of MAP 
            # returns the values for J and I, given O = True, namely, J = False, I = False, but as the 
            # values for J = False, I = True and J = False, I = False are the same both instantiation
            # give a correct result.
            #bnreasoner = BNReasoner_("testing/lecture_example2.BIFXML")
            #q = ["I", "J"]
            #evidence = {"O":True}    
            #print(bnreasoner.map(q,evidence))

            bnreasoner = BNReasoner_(file)

            current_time = time.time()
            map_evi = bnreasoner.map(Queri, evidence, order='minimum_degree_ordering')
            elapsed_time = time.time()
            result_map['data_degree'].append(elapsed_time - current_time)

            current_time = time.time()
            map_evi = bnreasoner.map(Queri, evidence, order='minimum_fill_ordering')
            elapsed_time = time.time()
            result_map['data_fill'].append(elapsed_time - current_time)

        if Mpe:
            # Test case: Given the BN, which can also be found in the provided lecture_example2.BIFXML and 
            # the query given in example 1 on slides 18-19 of the fourth Bayes’ lecture. Our implementation 
            # of MPE should return the same probability and extended factor (and it does)
            #
            #bnreasoner = BNReasoner_("testing/lecture_example2.BIFXML")
            bnreasoner = BNReasoner_(file)
            #evidence = {"O": False, "J": True}
            
            elapsed_time = time.time()
            mpe_evi = bnreasoner.mpe(evidence, ordering='minimum_degree_ordering')
            elapsed_time = time.time()
            result_mpe['data_degree'].append(elapsed_time - current_time)

            elapsed_time = time.time()
            mpe_evi = bnreasoner.mpe(evidence, ordering='minimum_fill_ordering')
            elapsed_time = time.time()
            result_mpe['data_fill'].append(elapsed_time - current_time)


    #print(data_to_plot)

    '''
    result_marginal = {'data_fill':[], 'data_degree':[]}
    result_map = {'data_fill':[], 'data_degree':[]}
    result_mpe = {'data_fill':[], 'data_degree':[]}

    result_marginal['data_fill'] = [0,0]
    result_marginal['data_degree'] = [1,0]
    result_map['data_fill'] = [0,0]
    result_map['data_degree'] = [0,2]
    result_mpe['data_fill'] = [0,0]
    result_mpe['data_degree'] = [3,0]
    '''


    total = pd.DataFrame()
    data = [result_marginal, result_map, result_mpe]
    names = ['marginal', 'MAP', 'MPE']
    for method in range(len(data)):
        #print(data[method]["data_fill"])
        df_fill = pd.DataFrame(data[method]["data_fill"], columns=['values'])
        df_fill['type']='fill'
        df_degree = pd.DataFrame(data[method]["data_degree"], columns=['values'])
        df_degree['type']='degree'
        df_new = pd.concat([df_fill, df_degree])
        df_new['label'] = names[method]
        total = pd.concat([total, df_new])

        '''

        result_values = pd.DataFrame(data[method], columns='values')
        result_values['type'] = names[method]
        total = pd.concat([total, result_values])

        result = pd.DataFrame(value['data_degree'], columns=['values'])
        result['type'] = 'Minimum degree'
        df_x = pd.DataFrame(value['data_fill'], columns=['values'])
        df_x['type']='Minimum fill'
        df = pd.concat([result, df_x])
        df['label'] = value["label"]
        total = pd.concat([total, df])
        '''
    #print(total)
    fig = plt.figure(figsize=(6,5))    
    fig.set_dpi(1200)
    sns.set()
    sns.boxplot(y="values", x="label", data=total, hue='type')

    query_text = ""
    if len(Queri)>1:
        query_text = "Big query with"
    else:
        query_text = "Small query with"

    evidence_text = ""
    if len(evidence)>1:
        evidence_text = "a lot of evidence"
    else:
        evidence_text = "a small amount of evidence"

    plt.title(text_to_plot[iter])
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Type")
    #plt.legend()

    # Create the output folder if it doesn't exist
    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #plt.show()
    output_file = output_folder + f"boxplot_{text_to_plot[iter]}_.png"
    print(output_file)
    #plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()