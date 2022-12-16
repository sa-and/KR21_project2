from typing import Union, Dict, List
from xmlrpc.client import Boolean
from BayesNet import BayesNet
from copy import deepcopy

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from itertools import product, combinations

class BNReasoner:
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
    def pruning(self,query: List[str], values: Dict[str, bool]) -> BayesNet:
        """ Prune the network, will drop variables that are not needed anymore

        Args:
            query (List): Given query, used to check if nodes are part of it
            values (Dict): The given evedince to a node, must be True or False. Structure: {Node:Value}
            
        Returns:
            p: BayesNet.object, returns the new pruned network
        """
        print("pruning")
        p = deepcopy(self.bn)
        finished = True # Continue checking if the pruning can continue
        all_cpts = p.get_all_cpts() # A dictionary of all cps in the network indexed by the variable they belong to
        while finished: # Stop pruning when the network cannot be pruned further
            finished = False
            for var in p.get_all_variables(): 
                if p.get_children(var) == [] and var not in query and var not in values: # Check if node is part of query
                    p.del_var(var) # Delete node if not part of query
                    finished = True
            for a, b in values.items(): # Get node name and value
                for variable in p.get_all_variables(): # Get all variables
                    cpt = all_cpts[variable]
                    if a in cpt.columns: # If variable from input matches a variable from a cpt column:
                        new_cpt = cpt.drop(cpt[cpt[a] != b].index) # If the same node exist with the opposite value, delete it
                        p.update_cpt(variable, new_cpt) # Update the cpt table without the opposite value
                    else:
                        continue
                for child in p.get_children(a): # Check if the removed node has children, if yes delete edge
                    p.del_edge((a, child)) # Delete edge if input variable has child
                    finished = True
        # print(p.get_all_cpts()," dit is p")
        return p

    def d_separation(self, x: List[str], y: List[str], z: List[str]) -> bool:
        """ Check if variable x is d-seperated from y given z

        Args:
            x (List): List of variables to check
            y (list): List of variables to check if x is seperated from
            z (list): List of given variables
            
        Returns:
            bool: True is d-seperated, False if not
        """
        p = deepcopy(self.bn)
        reach_single = []
        iterated = []
        
        for var in p.get_all_variables():
            if p.get_children(var) == [] and var not in x and var not in y and var not in z:
                p.del_var(var)
        for var in z:
            for child in p.get_children(var):
                p.del_edge([var, child])
        for var in x:
            iterated.append(var) # Append variables to check
            reach_single.extend(p.get_children(var)) # Append children
            while list(set(reach_single) - set(iterated)) !=[]: # While loop makes sure children of children are checked
                for a in list(set(reach_single) - set(iterated)): # Make sure all children are checked
                    reach_single.extend(p.get_children(a)) # Append children of children if there are any
                    iterated.append(a) # Append variable to prevent loop from checking again
        for var_2 in y:
            if var_2 in reach_single: # If y in x, not d-seperated
                print(x, 'and', y, 'are not d-separated by', z)
                return False
        print(x, 'and', y, 'are d-separated by', z) # Else d-seperated
        return True

    def random_order(self, network: BayesNet = None) -> List[str]:
        """
        :return: a random ordering of all variables in self.bn
        """
        if network is None:
            return list(np.random.permutation(self.bn.get_all_variables()))
        else:
            return list(np.random.permutation(network.get_all_variables()))
    
    def min_degree(self) -> List[str]:
        """Sort the nodes by looking at the degree

        Returns:
            List of variables sorted from small to big (degree in the interaction graph to the ordering)
        """
        p = deepcopy(self.bn)
        return [x[0] for x in sorted(p.get_interaction_graph().degree(), key = lambda x: x[1])] # Sort list and only return the variable name

    def min_fill(self) -> List[str]:
        """ Minimum fill ordering
            
        Returns:
            List of variables sorted from small to big (whose deletion would add the fewest new interaction to the ordering)
        """
        p = deepcopy(self.bn)
        i_graph = p.get_interaction_graph()
        order_edges = []
        for node in i_graph:
            i = 0
            neighbors = i_graph.neighbors(node)
            for a in neighbors:
                neigh_a = i_graph.neighbors(a)
                for b in neighbors:
                    if a == b: # Do nothing
                        continue
                    if b not in neigh_a: # Count if b is not in neigh_a
                        i += 1
            order_edges.append((node, i)) 
        return [x[0] for x in sorted(order_edges, key=lambda x: x[1])] # Sort the list and return only the variable name

    def init_factor(self,variables: List[str], value=0) -> pd.DataFrame:
        """
        Generate a default CPT.
        :param variables:   Column names
        :param value:       Which the default p-value should be
        :return:            A CPT
        """
        truth_table = product([True, False], repeat=len(variables))
        factor = pd.DataFrame(truth_table, columns=variables)
        factor['p'] = value
        return factor

    def sum_out_factors(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
        """
        Sum out some variable(s) in subset from a factor.
        :param factor:  factor over variables X
        :param subset:  a subset of variables X
        :return:        a factor corresponding to the factor with the subset summed out
        """
        if isinstance(factor, str):
            factor = self.bn.get_cpt(factor)
        if isinstance(subset, str):
            subset = [subset]

        new_factor = factor.drop(subset + ['p'], axis=1).drop_duplicates()
        new_factor['p'] = 0
        subset_factor = self.init_factor(subset)

        for i, y in new_factor.iterrows():
            for _, z in subset_factor.iterrows():
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] + self.bn.get_compatible_instantiations_table(
                    y[:-1].append(z[:-1]), factor)['p'].sum()
                # sum() instead of float() here, since the compatible table can be empty at times, this works around it

        return new_factor.reset_index(drop=True)

    def maximise_out(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
        """
        :param factor:
        :param subset:
        :return:
        """
        if isinstance(factor, str):
            factor = self.bn.get_cpt(factor)
        if isinstance(subset, str):
            subset = [subset]

        # Copy the factor, drop the variable(s) to be maximized, the extensions and 'p' and drop all duplicates to get
        # each possible instantiation.
        ext = [c for c in factor.columns if c[:3] == 'ext']
        instantiations = deepcopy(factor).drop(subset + ext + ['p'], axis=1).drop_duplicates()
        res_factor = pd.DataFrame(columns=factor.columns)

        if len(instantiations.columns) == 0:
            try:
                res_factor = res_factor.append(factor.iloc[factor['p'].idxmax()])
            except IndexError:
                print('w')
        else:
            for _, instantiation in instantiations.iterrows():
                cpt = self.bn.get_compatible_instantiations_table(instantiation, factor)
                res_factor = res_factor.append(factor.iloc[cpt['p'].idxmax()])

        # For each maximized-out variable(s), rename them to ext(variable)
        for v in subset:
            x = res_factor.pop(v)
            res_factor[f'ext({v})'] = x

        return res_factor.reset_index(drop=True)

    def factor_multiplication(self,f,g):
        '''
        Given two factors f and g, compute the multiplied factor h=fg
        :param f: Factor f
        :param g: Factor g
        :returns: Multiplied factor h=f*g
        '''

        # check what the overlapping var(s) is
        vars_f = [x for x in f.columns]
        vars_g = [x for x in g.columns]

        for var in vars_f:
            if var in vars_g and var != 'p':
                join_var = var

        # merge two dataframes
        merged = f.merge(g,left_on=join_var,right_on=join_var)

        # multiply probabilities
        merged['p'] = merged['p_x']*merged['p_y']

        # drop individual probability columns
        h = merged.drop(['p_x','p_y'],axis=1)

        # return h
        return h

    def factor_multiplication(self, factors: List[Union[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Multiply multiple factors with each other.
        :param factors: a list of factors
        :return:        a factor corresponding to the product of all given factors
        """
        # If there are strings in the input-list of factors, replace them with the corresponding cpt
        for x, y in enumerate(factors):
            if isinstance(y, str):
                factors[x] = self.bn.get_cpt(y)

        new_factor = factors[0].drop('p', axis=1)
        for i, factor in enumerate(factors[1:]):
            try:
                new_factor = new_factor.merge(factor.drop('p', axis=1), how='outer')
            except pd.errors.MergeError:
                new_factor = new_factor.join(factor.drop('p', axis=1), how='outer')
        new_factor['p'] = 1

        for i, z in new_factor.iterrows():
            for _, f in enumerate(factors):
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] * self.bn.get_compatible_instantiations_table(
                    z[:-1], f)['p'].sum()
                # sum() instead of float() here, since the compatible table can be empty at times, this works around it

        # Reordering new_factor, putting the extensions to the back
        cols = new_factor.columns
        ext = [c for c in cols if c[:3] == 'ext']
        if len(ext) > 0:
            rest = list(np.setdiff1d(cols, ext))
            new_factor = new_factor[rest + ext]

        return new_factor.reset_index(drop=True)


    def MPE(self, evidence: pd.Series, order_func: str = None) -> pd.DataFrame:
        """
        Compute the MPE instantiation for some given evidence.
        :param evidence:
        :param order_func:  String describing which order function to use
        :return:            Dataframe describing the MPE instantiation
        """
        N = deepcopy(self.bn)
        # Prune Edges
        for var in evidence.keys():
            for child in N.get_children(var):
                N.del_edge((var, child))

                new = N.get_compatible_instantiations_table(evidence, N.get_cpt(child)).reset_index(drop=True)
                new = new.drop([var], axis=1)
                N.update_cpt(child, new)
            u = N.get_compatible_instantiations_table(evidence, N.get_cpt(var)).reset_index(drop=True)
            N.update_cpt(var, u)

        if order_func is None or order_func == "random":
            order = self.random_order(N)
        elif order_func == "min_degree":
            order = self.min_degree(N)
        elif order_func == "min_fill":
            order = self.min_fill(N)
        else:
            raise Exception("Wrong order argument")

        S = N.get_all_cpts()
        for var_pi in order:
            # Pop all functions from S, which mention var_pi...
            func_k = [S.pop(key) for key, cpt in deepcopy(S).items() if var_pi in cpt]

            new_factor = self.multiply_factors(func_k) if len(func_k) > 1 else func_k[0]
            new_factor = self.maximise_out(new_factor, var_pi)

            # And replace them with the new factor
            S[var_pi] = new_factor

        res_factor = self.multiply_factors(list(S.values())) if len(S) > 1 else S.popitem()[1]
        return res_factor

    def MAP(self, M: List[str], evidence: pd.Series, order_func: str = None) -> pd.DataFrame:
        """
        Compute the MAP instantiation for some variables and some given evidence.
        :param M:           The MAP variables
        :param evidence:
        :param order_func:  String describing which order function to use
        :return:            Dataframe describing the MAP instantiation
        """
        if len(np.intersect1d(list(evidence.keys()), M)) > 0:
            raise Exception("Evidence cannot intersect with M")

        N = deepcopy(self.bn)
        N = self.pruning(M, evidence)
        for e in evidence.items():
            N.update_cpt(e[0], self.bn.get_compatible_instantiations_table(evidence,
                                                                           N.get_cpt(e[0])).reset_index(drop=True))

        if order_func is None or order_func == "random":
            order = self.random_order(N)
        elif order_func == "min_degree":
            order = self.min_degree(N)
        elif order_func == "min_fill":
            order = self.min_fill(N)
        else:
            raise Exception("Wrong order argument")

        S = N.get_all_cpts()
        for var_pi in order:
            # Pop all functions from S, which mention var_pi...
            func_k = [S.pop(key) for key, cpt in deepcopy(S).items() if var_pi in cpt]

            new_factor = self.factor_multiplication(func_k) if len(func_k) > 1 else func_k[0]
            if var_pi in M:
                new_factor = self.maximise_out(new_factor, var_pi)
            else:
                new_factor = self.sum_out_factors(new_factor, var_pi)
            # And replace them with the new factor
            # print(new_factor)
            S[var_pi] = new_factor


        res_factor = self.factor_multiplication(list(S.values())) if len(S) > 1 else S.popitem()[1]
        return res_factor

    # def maximise_out(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
    #     """
    #     :param factor:
    #     :param subset:
    #     :return:
    #     """
    #     if isinstance(factor, str):
    #         factor = self.bn.get_cpt(factor)
    #     if isinstance(subset, str):
    #         subset = [subset]

    #     # Copy the factor, drop the variable(s) to be maximized, the extensions and 'p' and drop all duplicates to get
    #     # each possible instantiation.
    #     ext = [c for c in factor.columns if c[:3] == 'ext']
    #     instantiations = deepcopy(factor).drop(subset + ext + ['p'], axis=1).drop_duplicates()
    #     res_factor = pd.DataFrame(columns=factor.columns)

    #     if len(instantiations.columns) == 0:
    #         try:
    #             res_factor = res_factor.append(factor.iloc[factor['p'].idxmax()])
    #         except IndexError:
    #             print('w')
    #     else:
    #         for _, instantiation in instantiations.iterrows():
    #             cpt = self.bn.get_compatible_instantiations_table(instantiation, factor)
    #             res_factor = res_factor.append(factor.iloc[cpt['p'].idxmax()])

    #     # For each maximized-out variable(s), rename them to ext(variable)
    #     for v in subset:
    #         x = res_factor.pop(v)
    #         res_factor[f'ext({v})'] = x

    #     return res_factor.reset_index(drop=True)

    # def init_factor(self, variables: List[str], value=0) -> pd.DataFrame:
    #     """
    #     Generate a default CPT.
    #     :param variables:   Column names
    #     :param value:       Which the default p-value should be
    #     :return:            A CPT
    #     """
    #     print(variables)
    #     truth_table = product([True, False], repeat=len(variables))
    #     factor = pd.DataFrame(truth_table, columns=variables)
    #     factor['p'] = value
    #     return factor

    # def sum_out_factors(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
    #     """
    #     Sum out some variable(s) in subset from a factor.
    #     :param factor:  factor over variables X
    #     :param subset:  a subset of variables X
    #     :return:        a factor corresponding to the factor with the subset summed out
    #     """
    
    #     if isinstance(factor, str):
    #         factor = self.bn.get_cpt(factor)
    #     if isinstance(subset, str):
    #         subset = [subset]

    #     new_factor = factor.drop(subset + ['p'], axis=1).drop_duplicates()
    #     new_factor['p'] = 0
    #     subset_factor = self.init_factor(subset)

    #     for i, y in new_factor.iterrows():
    #         for _, z in subset_factor.iterrows():
    #             new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] + self.bn.get_compatible_instantiations_table(
    #                 y[:-1].append(z[:-1]), factor)['p'].sum()
    #             # sum() instead of float() here, since the compatible table can be empty at times, this works around it

    #     return new_factor.reset_index(drop=True)
    
    def elimination(self, data: pd.DataFrame, var: List[str]) -> pd.DataFrame:
        """ Sum out a set of variables by using variable elimination

        Args:
            data (pd.Dataframe): Dataframe of where elimination should take place
            var (List): Variable to be summed out
            
        Returns:
            pd.Dataframe: Dataframe after elimination
        """
        print(data, "initial dataframe")
        remaining = data.drop(columns=var) # Drop column thats need to be summed out
        rem_list = list(remaining.columns.values)[:-1] # Get remaining dataframe
        if len(rem_list) == 0: # If empty return empty frame
            print("This was the only variable, thus no data could be returned")
            return pd.DataFrame()
        eliminated = remaining.groupby( # Else sum of matching values of variables
            rem_list).aggregate({'p': 'sum'})
        eliminated.reset_index(inplace=True)
        return eliminated

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

