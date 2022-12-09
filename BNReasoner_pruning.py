from typing import Union
from BayesNet import BayesNet
import pandas as pd
from copy import deepcopy


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
        :param Q: (query) set of variables, in case of d-separation the var of which you want to know whether they are d-separated
        :param e: the set of evidence variables or when used to check whether X and Y are d-separation by Z, is this Z
        :param evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False}), containing the assignments of the evidence. 
                It is not obligated, therefor this function is usable for d-separation, without knowing the assignment of the Z
        :return: pruned network
        '''
        new_bn = deepcopy(bn)
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
                if not evidence.empty:
                    BayesNet.update_cpt(new_bn, value, BayesNet.reduce_factor(evidence, BayesNet.get_cpt(new_bn, value)))
        
        #Node Pruning
        #Need to keep removing leafnodes untill all leafnodes that can be removed are removed
        i = 1
        while i > 0:
            i = 0
            var = BayesNet.get_all_variables (new_bn)
            for v in var:
                child = BayesNet.get_children(new_bn, v)
                #If node is a leafnode and not in the Q or e, remove from bn
                if len(child) == 0 and v not in Q and v not in e:
                    BayesNet.del_var(new_bn, v)                
                    i += 1    
        return new_bn  #Niet zeker of dit nodig is
        
        
    def d_separation(self, bn, X, Y, Z):
        '''
        :param bn: Bayes network
        :param X, Y, Z: sets of var of which you want to know whether they are d-separated by Z
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

    def independence(self, bn, X, Y, Z):
        '''
        Implementation of Markov Property and Symmetry in DAGS to determine independence
        :param bn: Bayesian Network
        :param X, Y, Z: sets of variable of which to decide whether X is independent of Y given Z
        :returns: Bool, True if X and Y are independent given Z, False if X and Y are not independent given Z 
        '''
        Not_all_parents_of_X = False        
        for x in X: 
            for parent in BayesNet.get_all_variables(bn):
                if x in BayesNet.get_children(bn, parent):
                    if parent not in Z:
                        Not_all_parents_of_X = True
                        break
            if Not_all_parents_of_X:
                break

        if Not_all_parents_of_X:
            for y in Y: 
                for parent in BayesNet.get_all_variables(bn):
                    if y in BayesNet.get_children(bn, parent):
                        print(parent)
                        if parent not in Z:
                            return False 
            for y in Y:
                # print(y)
                for x in X:
                    # print(self.loop_over_children(bn, x, y))
                    if not self.loop_over_children(bn, x, y):
                        # print(x, "is not a descendent of ", y)
                        return False
            return True

        for x in X:
            for y in Y:
                if not self.loop_over_children(bn, y, x):
                    return False
                     
        return True 

    def marginalization(self,cpt, X):#bn,X):
        '''
        Given a factor and a variable X, compute the CPT in which X is summed-out
        :param cpt: a factor
        :param X: variable X
        :returns: the CPT in which X is summed-out
        '''

        #cpt = BayesNet.get_cpt(bn,X)
                
        newcpt = cpt.drop([X],axis=1)
        
        remaining_vars = [x for x in newcpt.columns if x != X and x != 'p'] 
          
        newcpt = newcpt.groupby(remaining_vars).agg({'p': 'sum'})
        
        newcpt.reset_index(inplace=True)
        
        return newcpt

    def maxing_out(self,bn,X):#cpt,X):
        '''
        Given a factor and a variable X, compute the CPT in which X is maxed-out
        :param cpt: a factor
        :param X: variable X
        :returns: the CPT in which X is maxed-out
        '''

        cpt = BayesNet.get_cpt(bn,X)

        newcpt = cpt.drop([X],axis=1)

        remaining_vars = [x for x in newcpt.columns if x != X and x != 'p']

        newcpt = newcpt.groupby(remaining_vars).agg({'p': 'max'})
        newcpt.reset_index(inplace=True)


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

        # merge two dataframes
        #merged = f.merge(g,left_on=join_var,right_on=join_var)
        merged = pd.merge(f,g, how="outer", on=join_var)
        # multiply probabilities
        merged['p'] = merged['p_x']*merged['p_y']

        # drop individual probability columns
        h = merged.drop(['p_x','p_y'],axis=1)

        # return h
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
            # print(to_del_var)
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
        # print(dict_degrees)      
        return sorted(dict_degrees.items(), key=lambda item: item[1])[0][0]

    def least_edges(self, bn, set):
        # All_var  = BayesNet.get_all_variables(bn)
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
        # print(dict_edges)
        # print()
        # print(dict_nr_edges)
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

    def variable_elimination(self, to_sum_out_vars, order):
        '''
        :param to_sum_out_vars: Set of variables that need to be summed out
        :param order: a string containing the type of ordering --> misschien gewoon de ordering mee geven (dus al eerder berekenen)
        :return: the factor with all var in the to_sum_out_vars summed out
        '''
        f = deepcopy(BayesNet.get_all_cpts(self.bn))
        if order == "minimum_degree_ordering":
            order = self.minimum_degree_ordering(self.bn, to_sum_out_vars)
        else:
            order = self.minimum_fill_ordering(self.bn, to_sum_out_vars)

        # order = ["Winter?", "Wet Grass?", "Sprinkler?", "Rain?"]

        done = list()
        n = None
        # print(to_sum_out_vars)
        # print(order)
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
                        # print(n)
                        n = self.multiply_factors(n, factor)
                    # print(n)
                    n = self.marginalization(n, s)
                    
            else:      
                for factor in to_multiply:
                    # print(n)
                    n = self.multiply_factors(n, factor)
                # print("1")
                # print(n)
                # print("2")
                n = self.marginalization(n, s)
        print(n)
        return n               

Pruning = False
check_d_separation = False #True
Independence = False #True
Marginalization = False
MaxingOut = False
MultiplyFactor = False#True
Ordering = False# True
Variable_Elimination = False#True

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
    Y = {"Slippery Road?"}
    X = {"Wet Grass?"}
    Z = {"Rain?"} 
    if bnreasoner.d_separation(bnreasoner.bn, X,Y,Z):
        print(X, "is d-separated from ", Y, "by ", Z)
    else:
        print(X, "is not d-separated from ", Y, "by ", Z)

if Independence:
    #Ik weet niet zeker of de implementatie van independence compleet of efficient is,
    #maar I guess dat het werkt, het is gebaseerd op DAGs en de Markov Property en Symmetry
    #zijn denk ik geimplementeerd.
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    Y = {"Winter?"}
    X = {"Slippery Road?"}
    Z = {} 
    if bnreasoner.independence(bnreasoner.bn, X,Y,Z):
        print(X, "is independent from ", Y, "given ", Z)
    else:
        print(X, "is not independent from ", Y, "given ", Z)

if Marginalization:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    X = "Wet Grass?"
    cpt = BayesNet.get_cpt(bnreasoner.bn,X)
    print(bnreasoner.marginalization(cpt, X))

if MaxingOut:
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    X = "Wet Grass?"
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
    Set = {"Sprinkler?", "Rain?"}#BayesNet.get_all_variables(bnreasoner.bn)
    print(bnreasoner.minimum_degree_ordering(bnreasoner.bn, Set))
    print(bnreasoner.minimum_fill_ordering(bnreasoner.bn, Set))

if Variable_Elimination:
    Set = {"Rain?", "Winter?", "Sprinkler?", "Wet Grass?"}#, "Winter"}
    bnreasoner = BNReasoner_("testing/lecture_example.BIFXML")
    print(bnreasoner.variable_elimination(Set, "minimum_degree_ordering"))
