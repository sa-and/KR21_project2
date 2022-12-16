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
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def pruning(self,query: List[str], values: Dict[str, bool]) -> BayesNet:

        bayes_net = deepcopy(self.bn)
        finished = True # Continue checking if the pruning can continue
        all_cpts = bayes_net.get_all_cpts() # A dictionary of all cps in the network indexed by the variable they belong to
        while finished: # Stop pruning when the network cannot be pruned further
            finished = False
            for var in bayes_net.get_all_variables(): 
                if bayes_net.get_children(var) == [] and var not in query and var not in values: # Check if node is part of query
                    bayes_net.del_var(var) # Delete node if not part of query
                    finished = True
            for a, b in values.items(): # Get node name and value
                for variable in bayes_net.get_all_variables(): # Get all variables
                    cpt = all_cpts[variable]
                    if a in cpt.columns: # If variable from input matches a variable from a cpt column:
                        new_cpt = cpt.drop(cpt[cpt[a] != b].index) # If the same node exist with the opposite value, delete it
                        bayes_net.update_cpt(variable, new_cpt) # Update the cpt table without the opposite value
                    else:
                        continue
                for child in bayes_net.get_children(a): # Check if the removed node has children, if yes delete edge
                    bayes_net.del_edge((a, child)) # Delete edge if input variable has child
                    finished = True
        print(bayes_net.get_all_cpts()," dit is p")
        return bayes_net

    def d_separation(self, x: List[str], y: List[str], z: List[str]) -> bool:
        bayes_net = deepcopy(self.bn)
        reach_single = []
        iterated = []
        
        for var in bayes_net.get_all_variables():
            if bayes_net.get_children(var) == [] and var not in x and var not in y and var not in z:
                bayes_net.del_var(var)
        for var in z:
            for child in bayes_net.get_children(var):
                bayes_net.del_edge([var, child])
        for var in x:
            iterated.append(var) # Append variables to check
            reach_single.extend(bayes_net.get_children(var)) # Append children
            while list(set(reach_single) - set(iterated)) !=[]: # While loop makes sure children of children are checked
                for a in list(set(reach_single) - set(iterated)): # Make sure all children are checked
                    reach_single.extend(bayes_net.get_children(a)) # Append children of children if there are any
                    iterated.append(a) # Append variable to prevent loop from checking again
        for var_2 in y:
            if var_2 in reach_single: # If y in x, not d-seperated
                print(x, 'and', y, 'are not d-separated by', z)
                return False
        print(x, 'and', y, 'are d-separated by', z) # Else d-seperated
        return True

    def random_order(self, network: BayesNet = None) -> List[str]:
        if network is None:
            return list(np.random.permutation(self.bn.get_all_variables()))
        else:
            return list(np.random.permutation(network.get_all_variables()))
    
    def min_degree(self, network: BayesNet = None) -> List[str]:
        if network is None:
            bayes_net = deepcopy(self.bn)
        else:
            bayes_net = deepcopy(network)
            
        return [x[0] for x in sorted(bayes_net.get_interaction_graph().degree(), key = lambda x: x[1])] # Sort list and only return the variable name

    def min_fill(self, network: BayesNet = None) -> List[str]:
        if network is None:
            bayes_net = deepcopy(self.bn)
        else:
            bayes_net = deepcopy(network)

        i_graph = bayes_net.get_interaction_graph()
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

    def init_factor(self, variables: List[str], value=0) -> pd.DataFrame:
        truth_table = product([True, False], repeat=len(variables))
        factor = pd.DataFrame(truth_table, columns=variables)
        factor['p'] = value
        return factor

    def sum_out_factors(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
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


    def factor_multiplication(self, factors: List[Union[str, pd.DataFrame]]) -> pd.DataFrame:
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

            new_factor = self.factor_multiplication(func_k) if len(func_k) > 1 else func_k[0]
            new_factor = self.maximise_out(new_factor, var_pi)

            # And replace them with the new factor
            S[var_pi] = new_factor

        res_factor = self.factor_multiplication(list(S.values())) if len(S) > 1 else S.popitem()[1]
        return res_factor

    def MAP(self, M: List[str], evidence: pd.Series, order_func: str = None) -> pd.DataFrame:
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

    def compute_marginal(self, query: List[str], evidence: pd.Series = None, order: List[str] = None) -> pd.DataFrame:
        if order is None:
            order = self.random_order()

        S = self.bn.get_all_cpts()

        if evidence is not None:  # If there's evidence, reduce all CPTs using the evidence
            for var in self.bn.get_all_variables():
                var_cpt = self.bn.get_cpt(var)
                if any(evidence.keys().intersection(var_cpt.columns)):  # If the evidence occurs in the cpt
                    new_cpt = self.bn.get_compatible_instantiations_table(evidence, var_cpt)
                    S[var] = new_cpt

        pi = [nv for nv in order if nv not in query]
        for var_pi in pi:
            # Pop all functions from S, which mention var_pi...
            func_k = [S.pop(key) for key, cpt in deepcopy(S).items() if var_pi in cpt]

            new_factor = self.factor_multiplication(func_k)
            new_factor = self.sum_out_factors(new_factor, var_pi)
            # And replace them with the new factor
            S[var_pi] = new_factor

        res_factor = self.factor_multiplication(list(S.values())) if len(S) > 1 else S.popitem()[1]

        if evidence is not None:  # Normalizing over pr_evidence
            cpt_e = self.compute_marginal(list(evidence.keys()), order=order)
            pr_evidence = float(self.bn.get_compatible_instantiations_table(evidence, cpt_e)['p'])
            res_factor['p'] = res_factor['p'] / pr_evidence

        return res_factor

    def elimination(self, data: pd.DataFrame, var: List[str]) -> pd.DataFrame:
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

    def check_independence(self, bn, X, Y, Z):
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
