from typing import Union, Dict, List
from xmlrpc.client import Boolean
from BayesNet import BayesNet
from copy import deepcopy
import pandas as pd
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
        print(p.get_all_cpts()," dit is p")
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

    def init_factor(self, variables: List[str], value=0) -> pd.DataFrame:
        """
        Generate a default CPT.
        :param variables:   Column names
        :param value:       Which the default p-value should be
        :return:            A CPT
        """
        print(variables)
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