from typing import Union
from BayesNet import BayesNet
import pandas as pd
from copy import deepcopy
import random


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
        :param bn: Bayesnetwork
        :param Q: (query) set of variables, in case of d-separation the var of which you want to know whether they are d-separated
        :param e: the set of evidence variables or when used to check whether X and Y are d-separation by Z, is this Z
        :param evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False}), containing the assignments of the evidence. 
                It is not obligated, therefor this function is usable for d-separation, without knowing the assignment of the Z
        :return: pruned network
        '''       
        # print(type(e))
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
        #print(BayesNet.get_all_cpts(new_bn))
        return new_bn  

    def d_separation(self, X, Y, Z):
        '''
        :param X, Y, Z: sets of var of which you want to know whether they are d-separated by Z
        :returns: True if X and Y are d-separated, False if they are not d-separated by Z
        '''        
        q = X + Y
        bn = self.Network_Pruning(q, Z)
        
        #If there is no path between X and Y (order does not matter) means X and Y are d-separated
        for x in X:
            for y in Y:
                if self.loop_over_children(bn, y, x):
                    if self.loop_over_children(bn, x, y):
                        return True
        return False
     
    def loop_over_children(self,bn, y, parent):
        '''Checks for if y is a descendant of parent'''

        # print("parent: ", parent)
        children = BayesNet.get_children(bn, parent)
        # print("children: ", children)
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
        Implementation of Markov Property and Symmetry in DAGS to determine independence
        :param X, Y, Z: sets of variable of which to decide whether X is independent of Y given Z
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
        Given a factor and a variable X, compute the CPT in which X is summed-out
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
        Given a factor and a variable X, compute the CPT in which X is maxed-out
        :param cpt: a factor
        :param X: variable X
        :returns: the CPT in which X is maxed-out
        '''
        
        newcpt = deepcopy(cpt)
        remaining_vars = [x for x in newcpt.columns if x != X and x != 'p']
        
        # if len(list(newcpt.columns)) == 1:
        #     p = newcpt["p"].max()
        #     newcpt = pd.DataFrame({" ": ["T"], "p":[p]})
        # else:
        if len(remaining_vars) != 0:
            newcpt = newcpt.loc[newcpt.groupby(remaining_vars)['p'].idxmax()]
        cpt_with_extended_factor = list()
        for x in newcpt.columns:
            if x != X:
                cpt_with_extended_factor.append(x)
        cpt_with_extended_factor.append(X)
        newcpt = newcpt[cpt_with_extended_factor]


        return newcpt

    def multiply_factors(self,f,g):
        '''
        Given two factors f and g, compute the multiplied factor h=fg
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

    def minimum_fill_ordering(self, bn, set):
        '''
        :param bn: Bayesnet of which an ordering needs to be returned based on minimum fill
        :returns: a string containing an ordering based on minumum fill
        '''
        order = list()
    
        #dictionary of all edges in the interaction graph
        _, edges = self.least_edges(bn, set)
            
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
        :param variables: a list with all variables that are not yet in the ordering
        :param edges: a dictionary containing all the edges of the variables not yet in the ordering/ 
                    stillin the interaction graph
        :returns: the variable which would, when removed add the least number of edges
        '''
        dict_fill = dict()
        for var in variables: 
            vars_edges = deepcopy(edges)
            _, added_edges = self.connect_nodes(var, vars_edges, set)
            dict_fill[var] = added_edges   
        return sorted(dict_fill.items(), key=lambda item: item[1])[0][0]        
        
    def minimum_degree_ordering(self, bn, set):
        '''
        :param bn: Bayesian network for which an ordering needs to be returned based on minimum degree
        :returns: a string containing an ordering based on minumum fill
        '''
        order = list()

        #get the var that has the least edges and a dict containing for each variable the edges in the interaction graph
        to_del_var, edges = self.least_edges(bn, set)
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
        :param variables: a list with all variables that are not yet in the ordering
        :param edges: a dictionary containing all the edges of the variables not yet in the ordering/ 
                    still in the interaction graph
        :returns: the variable with the least number of edges
        '''
        dict_degrees = dict()
        for var in variables:
            dict_degrees[var] = len(edges[var]) 
        return sorted(dict_degrees.items(), key=lambda item: item[1])[0][0]

    def least_edges(self, bn, set):
        dict_nr_edges = dict()
        dict_edges = dict()
        for var in set:
            dict_nr_edges[var] = 0
            dict_edges[var] = list()

        for var in set:
            children = BayesNet.get_children(bn, var)
            dict_nr_edges[var] += len(children)
            for child in children:
                    dict_edges[var].append((var,child))  
                    if child in set:              
                        dict_edges[child].append((child, var))
                        dict_nr_edges[child] += 1      
        return sorted(dict_nr_edges.items(), key=lambda item: item[1])[0][0], dict_edges

    def connect_nodes(self, var, edges, set):  
        '''
        adds edges between nodes if necessary when var is removed and removes var
        and count the number of added edges
        :param var: string containing the variable to be deleted
        :param edges: dict containing all edges of each variable
        :returns: a dict containing for all variables that are not yet in the ordering , the 
                  in the graph, including the edges added when var is removed
                : an integer that count the number of edges that are added when var is removed
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
        cpts = self.bn.get_all_cpts()
       # print(cpts)
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
        evidence e, compute the marginal distribution P(Q|e). Note that Q is a
        subset of the variables in the Bayesian network X with Q ⊂ X but can also
        be Q = X
        :param Q: list of queri variables
        :param e: a series of assignments as tuples, containing the evidence
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
        p_Q_E = self.marginal_distribution(q,e, order) 
        p_Q_E = p_Q_E.loc[p_Q_E["p"].round(10) == max(p_Q_E['p'].round(10))]
        p_Q_E[" "] = 'T'
        list_order_cpt = list()
        for item in list(p_Q_E.columns):
            if item == 'p':
                list_order_cpt = [' ', 'p'] + list_order_cpt
            elif item != ' ':
                list_order_cpt.append(item)
        p_Q_E = p_Q_E[list_order_cpt]
        return p_Q_E    

    def mpe(self, e, ordering = None):
        
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

        list_order_cpt = list()
        for item in list(n.columns):
            if item == 'p':
                list_order_cpt = [' ', 'p'] + list_order_cpt
            elif item != ' ':
                list_order_cpt.append(item)
        n = n[list_order_cpt]
        return n  

Pruning = False
check_d_separation = False#True
Independence = False #True
Marginalization = False
MaxingOut = False
MultiplyFactor = False #True
Ordering = False #True
Variable_Elimination = False#True
Marginal_distribution = False#True
Map = True
Mpe = False#True

if Pruning:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Queri, evidence = ["Wet Grass?"], {"Winter?": True,"Rain?": False}
    return_bn = bnreasoner.Network_Pruning(Queri, evidence)
    print("----------------------")
    print(return_bn.get_all_cpts())
    return_bn.draw_structure()
    
#determine whether X is d-seperated from Y by Z
if check_d_separation:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Y = ["Slippery Road?"]
    X = ["Sprinkler?"]
    Z = ["Winter?"]
    if bnreasoner.d_separation(X,Y,Z):
        print(X, "is d-separated from ", Y, "by ", Z)
    else:
        print(X, "is not d-separated from ", Y, "by ", Z)

if Independence:
    #Ik weet niet zeker of de implementatie van independence compleet of efficient is,
    #maar I guess dat het werkt, het is gebaseerd op DAGs en de Markov Property en Symmetry
    #zijn denk ik geimplementeerd.
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Y = ["Slippery Road?"]
    X = ["Wet Grass?"]
    Z = ["Rain?"] 
    if bnreasoner.independence(X,Y,Z):
        print(X, "is independent from ", Y, "given ", Z)
    else:
        print(X, "is not independent from ", Y, "given ", Z)

if Marginalization:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    X = "Rain?"
    cpt = BayesNet.get_cpt(bnreasoner.bn,"Wet Grass?")
    # print(cpt)
    print(bnreasoner.marginalization(cpt, X))

if MaxingOut:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    X = "Rain?"
    cpt = BayesNet.get_cpt(bnreasoner.bn,X)
    print(bnreasoner.maxing_out(cpt,X))
   
if MultiplyFactor:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    X = 'Sprinkler?'
    Y = 'Rain?'
    cpt_1 = BayesNet.get_cpt(bnreasoner.bn, X)
    cpt_2 = BayesNet.get_cpt(bnreasoner.bn,Y)
    print(bnreasoner.multiply_factors(cpt_1,cpt_2)) 
    
if Ordering:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Set = ["Winter?","Rain?", "Wet Grass?", "Sprinkler?"]#BayesNet.get_all_variables(bnreasoner.bn)
    print(bnreasoner.minimum_degree_ordering(bnreasoner.bn, Set))
    print(bnreasoner.minimum_fill_ordering(bnreasoner.bn, Set))

if Variable_Elimination:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Set = ["Winter?", "Rain?", "Wet Grass?","Sprinkler?"]
    # print(Set)
    print(type(bnreasoner.bn.get_all_cpts(),))
    Pr_set = bnreasoner.variable_elimination(bnreasoner.bn.get_all_cpts(), to_sum_out_vars=Set, order = "minimum_degree_ordering")
    print("---------")
    print(Pr_set)

if Marginal_distribution:
    #Ik weet niet zeker of deze klopt
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    q = ["Winter?", "Rain?"]
    evidence = {"Sprinkler?":True}
    
    print("Pr(Q,e)")
    print(bnreasoner.marginal_distribution(q,evidence))
    print("Pr(Q)")
    print(bnreasoner.marginal_distribution(q))
    
if Map:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    q = ["Winter?", "Rain?", "Slippery Road?"]
    evidence = {"Sprinkler?":True}
    
    print(bnreasoner.map(q,evidence))
    print(bnreasoner.map(q))
    q = ["Slippery Road?", "Wet Grass?"]
    evidence = {"Rain?":True, "Winter?":False}
    print(bnreasoner.map(q))

if Mpe:
    #Returns only the assignments of the var that have been maxed out, want rest is irrelevant
    #Vaker runnen als je een random extra row heb
    bnreasoner = BNReasoner_("testing/usecase.BIFXML")
    evidence = {"hepatitis": False}
    print(bnreasoner.mpe(evidence, "minimum_degree_ordering"))