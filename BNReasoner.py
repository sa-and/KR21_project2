import pandas as pd
from typing import Union, Literal
from BayesNet import BayesNet
import random
import networkx as nx
from copy import deepcopy
import cProfile

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

    @staticmethod
    def leaves(bn: BayesNet) -> set[str]:
        vars = bn.get_all_variables()
        ls = set()
        for v in vars:
            if len(bn.get_children(v)) == 0:
                ls.add(v)
        return ls

    @staticmethod
    def has_undirected_path(bn: BayesNet, x: str, Y: set[str]):
        visited = [x]
        seen = set()
        while len(visited) > 0:
            node = visited.pop(0)
            seen.add(node)
            ancs = BNReasoner.parents(bn, node) - seen
            children = set(bn.get_children(node)) - seen
            if len(Y.intersection(children.union(ancs))) > 0:
                return True
            visited.extend(ancs)
            visited.extend(children)
        return False

    @staticmethod
    def parents(bn: BayesNet, x: str) -> set[str]:
        rents = set()
        for var in bn.get_all_variables():
            if x in bn.get_children(var):
                rents.add(var)
        return rents

    @staticmethod
    def disconnected(bn: BayesNet, X: set[str], Y: set[str]) -> bool:
        """
        Determines whether two sets X and Y of variables are disconnected
        in a bayes net
        """
        for x in X:
            if BNReasoner.has_undirected_path(bn, x, Y):
                return False
        for y in Y:
            if BNReasoner.has_undirected_path(bn, y, X):
                return False
        return True

    @staticmethod
    def marginalize_pr(cpt: pd.DataFrame, var: str, pr: float) -> pd.DataFrame:
        cpt.loc[cpt[var] == True, 'p'] = cpt[cpt[var] == True].p.multiply(pr)
        cpt.loc[cpt[var] == False, 'p'] = cpt[cpt[var] == False].p.multiply(1-pr)
        all_other_vars = [col for col in cpt.columns if col not in [var, 'p']]
        cpt = cpt.groupby(all_other_vars, as_index=False).sum()
        del cpt[var]
        return cpt

    def _pr(self, x: str):
        """
        Computes probability of x Pr(x) in graph by marginalizing on 
        dependencies
        """
        cpt = self.bn.get_cpt(x)
        deps = [col for col in cpt.columns if col not in ['p', x]]
        if len(deps) == 0:
            return cpt[cpt[x] == True].p.values[0]
        cpt = cpt.copy()
        for dep in deps:
            pr_dep = self._pr(dep)
            cpt = BNReasoner.marginalize_pr(cpt, dep, pr_dep)
        return cpt[cpt[x] == True].p.values[0]

    def prune(self, Q: set[str], e: pd.Series):
        """
        Given a set of query variables Q and evidence e, node- and edge-prune the 
        Bayesian network s.t. queries of the form P(Q|E) can still be correctly 
        calculated. (3.5 pts)
        """
        e_vars = set(e.index)
        # Delete all edges outgoing from evidence e and replace with reduced factor
        for e_var in e_vars:
            children = self.bn.get_children(e_var)
            for child in children:
                self.bn.del_edge((e_var, child)) # Edge prune
                child_cpt = self.bn.get_cpt(child)
                reduced = BayesNet.reduce_factor(e, child_cpt) # Replace child cpt by reduced cpt
                reduced = reduced[reduced.p > 0.] # Delete any rows where probability is 0
                self.bn.update_cpt(child, reduced)

        # Node pruning after edge pruning
        leaves = BNReasoner.leaves(self.bn) - set(e_vars.union(Q))
        while len(leaves) > 0:
            leaf = leaves.pop()
            self.bn.del_var(leaf)
            leaves = leaves.union(BNReasoner.leaves(self.bn) - set(e_vars.union(Q)))

    def dsep(self, X: set[str], Y: set[str], Z: set[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated 
        of Y given Z. (4pts)
        """
        bn = deepcopy(self.bn)
        while True:
            leaves = BNReasoner.leaves(bn) - X.union(Y).union(Z)
            had_leaves = bool(len(leaves))
            zout_edges = []
            for z in Z:
                zout_edges.extend((z, child) for child in bn.get_children(z))
            had_edges = bool(len(zout_edges))
            if not had_edges and not had_leaves:
                break
            for u,v in zout_edges:
                bn.del_edge((u, v))
            for leaf in leaves:
                bn.del_var(leaf)
        return BNReasoner.disconnected(bn, X, Y)

    def independent(self, X, Y, Z):
        """
        Independence: Given three sets of variables X, Y, and Z, determine whether X
        is independent of Y given Z. (Hint: Remember the connection between d-separation 
        and independence) (1.5pts)
        """
        return self.dsep(X, Y, Z)

    def _compute_sum_trivial(self, original_ft: pd.DataFrame, ft: pd.DataFrame):
        """
        Computes the resulting trivial factor the result of summing out the last
        variable in a factor table.
        """
        ft.loc[0, 'p'] = original_ft.p.sum()
        return ft

    def _maxing_out(self, factor_table: pd.DataFrame, x: str):
        """
        Helper function to max out a variable directly from a given table
        """
        group_by_vars = [col for col in factor_table if col not in ['p', x]]
        if len(group_by_vars) == 0:
            idx = factor_table.p.idxmax()
            assignment = factor_table.loc[idx][x]
            return pd.DataFrame(), pd.Series([assignment], index=[idx])
        lines = factor_table.groupby(group_by_vars).p.idxmax()
        ft = factor_table.loc[lines].copy()
        assignments = factor_table.loc[lines, x].copy()
        del ft[x]
        return ft, assignments

    def _sum_out(self, factor_table: pd.DataFrame, x: str):
        """
        Helper function to sum out a variable directly from a given table
        """
        group_by_vars = [col for col in factor_table if col not in ['p', x]]
        if len(group_by_vars) == 0:
            return self._compute_sum_trivial(factor_table, pd.DataFrame())
        table = factor_table.groupby(group_by_vars, as_index=False).sum()
        del table[x]
        return table

    def marginalize(self, factor: str, x: str):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        """
        return self._sum_out(self.bn.get_cpt(factor), x)

    def maxing_out(self, factor: str, x: str):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember
        to also keep track of which instantiation of X led to the maximized value. (5pts)
        
        """
        return self._maxing_out(self.bn.get_cpt(factor), x)

    def _factor_mult(self, factor_table1: pd.DataFrame, factor_table2: pd.DataFrame) -> pd.DataFrame:
        common_columns = list(set(factor_table1.columns).intersection(set(factor_table2.columns)) - {'p'})
        if len(common_columns) == 0: # If no common column then create one
            factor_table1['temp'] = 1
            factor_table2['temp'] = 1
            new_cpt = pd.merge(factor_table1, factor_table2, on=['temp'], how='outer')
            del factor_table1['temp']
            del factor_table2['temp']
            del new_cpt['temp']
        else:
            new_cpt = pd.merge(factor_table1, factor_table2, on = common_columns, how='outer')
        new_cpt['p'] = new_cpt['p_x'] * new_cpt['p_y']
        new_cpt.drop(['p_x', 'p_y'], axis=1, inplace=True)
        return new_cpt

    def factor_mult(self, factor_f, factor_g):
        """
        Given two factors f and g, compute the multiplied factor h=fg. (5pts)
        """
        factor_table_f = self.bn.get_cpt(factor_f)
        factor_table_g = self.bn.get_cpt(factor_g)
        return self._factor_mult(factor_table_f, factor_table_g)

    def _get_elim_order(self, vars, elim_method):
        if elim_method == 'min_degree':
            return self.min_degree_ordering(vars)
        elif elim_method == 'min_fill':
            return self.minfill_ordering(vars)
        else: # Random 
            varsshuffle = list(vars) 
            random.shuffle(varsshuffle)
            return varsshuffle

    def get_factors_using(self, var: str):
        cpts = self.bn.get_all_cpts()
        factors = set()
        for f, cpt in cpts.items():
            if var in cpt.columns:
                factors.add(f)
        return factors
    
    def _reduce_variable(self, var: str):
        """Once again trying to implement elim var"""
        neighbors = self.get_factors_using(var)
        if len(neighbors) == 0:
            return None, None
        cpt = None 
        for neighbor in neighbors: # Multiply all neighbors together in factor multiply
            if cpt is None:
                cpt = self.bn.get_cpt(neighbor)
                continue
            neighbor_cpt = self.bn.get_cpt(neighbor)
            cpt = self._factor_mult(cpt, neighbor_cpt) 
        cpt = self._sum_out(cpt, var) # Sum out variable
        return cpt, neighbors

    def variable_elimination(self, X: set[str], elim_method: Union[Literal['min_fill'], Literal['min_degree'], Literal['random']] = 'min_degree'):
        """
        Variable Elimination: Sum out a set of variables by using variable elimination.
        (5pts)
        """
        to_eliminate = set(self.bn.get_all_variables()) - X
        to_eliminate = self._get_elim_order(to_eliminate, elim_method)
        while len(to_eliminate) > 0:
            var = to_eliminate.pop(0)
            cpt, neighbors = self._reduce_variable(var)
            fname = self._get_factor_name(cpt)
            for neighbor in neighbors:
                self.bn.del_var(neighbor)
            if fname in self.bn.get_all_variables():
                self.bn.update_cpt(fname, cpt)
            else:
                self.bn.add_var(fname, cpt)

    @staticmethod
    def _get_factor_name(cpt: pd.DataFrame):
        cols = sorted([c for c in cpt.columns if c != 'p'])
        return 'f_'+','.join(cols)

    def min_degree_ordering(self, X: set[str]):
        """Given a set of variables X in the Bayesian network, 
        compute a good ordering for the elimination of X based on the min-degree heuristics (2pts) 
        and the min-fill heuristics (3.5pts). (Hint: you get the interaction graph ”for free” 
        from the BayesNet class.)"""
        interaction_graph = self.bn.get_interaction_graph()
        order = []

        for i in range(len(X)): 
            Lowest_degree = self.lowest_degree(interaction_graph, X)
            order.append(Lowest_degree)
            interaction_graph = self.remove_node_interaction(Lowest_degree, interaction_graph)
            X.remove(Lowest_degree)
        return order
    
    def remove_node_interaction(self, node, graph): 
        neighbours = [n for n in nx.neighbors(graph, node)]
        for neighbour in neighbours:
            copy_neighbours = deepcopy(neighbours)
            copy_neighbours.remove(neighbour)
            for i in copy_neighbours: 
                if not(self.nodes_connected(neighbour, i, graph)): 
                    graph.add_edge(neighbour, i)
        graph.remove_node(node)               
        return graph
    
    def minfill_ordering(self, X):
        interaction_graph = self.bn.get_interaction_graph()
        order = []
        for i in range(len(X)): 
            lowest_fill = self.lowest_fill(interaction_graph, X)
            order.append(lowest_fill)
            interaction_graph = self.remove_node_interaction(lowest_fill, interaction_graph)
            X.remove(lowest_fill)
        return order

    def lowest_fill(self, graph, X):
        lowest_fill = 1e10
        name = "test"
        for var in X: 
            amount = self.amount_added_interaction(var, graph)
            if amount < lowest_fill: 
                lowest_fill = amount 
                name = var 
        return name

    def amount_added_interaction(self, x, graph): 
        neighbours = [n for n in nx.neighbors(graph, x)]
        added_count = 0 
        for neighbour in neighbours:
            copy_neighbours = deepcopy(neighbours)
            copy_neighbours.remove(neighbour)
            for i in copy_neighbours: 
                if not(self.nodes_connected(neighbour, i, graph)): 
                    added_count += 1   
        return added_count/2
       
    def nodes_connected(self,u,v,graph): 
        return u in graph.neighbors(v)    

    def lowest_degree(self, graph, X):
        lowest_degree = 100
        name = "test" 

        for i in X:
            value = graph.degree[i]
            if value < lowest_degree:
                lowest_degree = value
                name = i 
        return name 

    def _reduce_all_factors(self, e: pd.Series):
        for var in self.bn.get_all_variables():
            cpt = self.bn.get_cpt(var)
            cpt = BayesNet.get_compatible_instantiations_table(e, cpt)
            self.bn.update_cpt(var, cpt)
    
    def marginal_distribution(self, Q: set[str], e: pd.Series, elim_method='min_degree'):
        """Given query variables Q and possibly empty evidence e, 
        compute the marginal distribution P(Q|e). Note that Q is a subset of 
        the variables in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)"""
        vars_to_keep = Q.union(set(e.index))
        self._reduce_all_factors(e)
        self.variable_elimination(vars_to_keep, elim_method)
        jptd = self.multiply_all_tables() # joint probability distribution Pr(Q ^ e)
        ept = jptd
        for q in Q:# Sum out all variabes in Q from Pr(Q ^ e) to obtain Pr(e)
            ept = self._sum_out(ept, q)

        pre = BayesNet.get_compatible_instantiations_table(e, ept)
        if len(pre) != 1:
            return pre
    
        pre = pre.p.values[0]
        jptd = BayesNet.get_compatible_instantiations_table(e, jptd)
        jptd.p /= pre # Compute Pr(Q ^ e) / Pr(e)
        return jptd

    def map(self, Q: set[str], e: pd.Series, elim_method='min_degree'):
        """Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)"""
        # self.prune(Q, e)
        vars_to_keep = Q.union(set(e.index))
        self.variable_elimination(vars_to_keep, elim_method)
        jptd = self.multiply_all_tables() # joint probability distribution Pr(Q ^ e)
        jptd = BayesNet.get_compatible_instantiations_table(e, jptd)
        ept = jptd
        q_assignments = {}
        for q in Q:
            ept, assignments = self._maxing_out(ept, q)
            q_assignments[q] = assignments
        final = {}
        i = assignments.index.values[0]
        for q, a in q_assignments.items():
            final[q] = a[i]
        return pd.Series(final)

    def mpe(self, e): 
        #Start with edge pruning and node pruning 
        #Get elimination order
        #maximize out for order 
        e_vars = set(e.index)
        Q = set(self.bn.get_all_variables()) - e_vars
        self.prune(Q,e)
        vars_to_keep = Q.union(set(e.index))
        self._reduce_all_factors(e)
        self.variable_elimination(vars_to_keep) # elminate all variables we don't need
        q_assignments = {}
        table = self.multiply_all_tables()
        for var in Q:
            table, assignments = self._maxing_out(table, var)
            q_assignments[var] = assignments
        final = {}
        i = assignments.index.values[0]
        for q, a in q_assignments.items():
            final[q] = a[i]
        return pd.Series(final)

    def multiply_all_tables(self): 
        """
        Multiplies all tables stored in graph together using factor multiplication
        and returns the resulting cpt.
        """
        all_tables = self.bn.get_all_cpts()
        if len(all_tables) == 0: # There are no tables left, return empty dataframe
            return pd.DataFrame()
        all_tables_list = list(all_tables.values())
        end_table  = all_tables_list[0]
        for i in range(1, len(all_tables_list)): 
            end_table = self._factor_mult(end_table,  all_tables_list[i])

        return end_table

def profile_ve():
    reasoner = BNReasoner('parkinsons.BIFXML')
    # Q = {'Treatment?'}
    e = pd.Series({'Parkinsons?': True})
    print(reasoner.mpe(e))

def main():
    # cProfile.run('profile_ve()')
    profile_ve()

if __name__ == '__main__':
    main()
