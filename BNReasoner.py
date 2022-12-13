import pandas as pd
from functools import partial
from typing import List
from typing import Union, Literal
from BayesNet import BayesNet
import itertools 
import networkx as nx
from copy import deepcopy


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
    def ancestors(bn: BayesNet, x: str) -> set[str]:
        ancs = set()
        all_vars = set(bn.get_all_variables())
        for var in all_vars - {x}:
            if BNReasoner.has_path(var, {x}):
                ancs.add(var)
        return ancs

    @staticmethod
    def has_path(bn: BayesNet, x: str, Y: set[str]) -> bool:
        """
        Determines whether there is a path from x to any element in Y.
        Simple BFS
        """
        visited = [x]
        while len(visited) > 0:
            node = visited.pop(0)
            children = bn.get_children(node)
            if any(y in children for y in Y):
                return True
            visited.extend(children)
        return False

    @staticmethod
    def disconnected(bn: BayesNet, X: set[str], Y: set[str]) -> bool:
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

    def num_deps(self, x):
        return len(self.bn.get_cpt(x).columns)-2

    def _get_default_ordering(self, deps):
        return deps
    
    def get_ordered(self, deps):
        """
        This can call any number of heuristics-based implementations that
        order the dependencies of a variable in a particular order. For now
        it just returns the same ordering as given.
        """
        return self._get_default_ordering(deps)

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
        deps = self.get_ordered(deps)
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

        TODO: Upate that it also checks other independencies
        """
        return self.dsep(X, Y, Z)
    
    def _compute_new_cpt(self, factor, x, which):
        """
        Given a factor and a variable X, compute the CPT in which X is either summed-out or maxed-out. (3pts)
        """
        factor_table = self.bn.get_cpt(factor)

        f = pd.DataFrame(columns = factor_table.columns)
        f["p"]= []
        del f[x]

        l = [False, True]
        instantiations = [list(i) for i in itertools.product(l, repeat = len(factor_table.columns) - 2)]

        Y = factor_table.columns
        count = 0

        for inst in instantiations:
            inst_dict = {}
            inst_list = []
            for j in range((len(factor_table.columns) - 2)):
                inst_dict[Y[j]] = inst[j]
                inst_list.append(inst[j])
            inst_series = pd.Series(inst_dict)
            comp_inst = self.bn.get_compatible_instantiations_table(inst_series, self.bn.get_cpt(factor))
            print(comp_inst)
            if which == 'max':
                new_p = comp_inst.p.max()  
            elif which == 'sum':  
                new_p = comp_inst.p.sum()
            inst_list.append(new_p)
            f.loc[count] = inst_list

            count += 1

        return f

    def marginalize(self, factor, x):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        """
        return self._compute_new_cpt(factor, x, 'sum')

    def maxing_out(self, factor, x):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember
        to also keep track of which instantiation of X led to the maximized value. (5pts)
        
        TODO: Keep track of which value of X this comes from
        """
        return self._compute_new_cpt(factor, x, 'max')

    def factor_mult(self, factor_f, factor_g):
        """
        Given two factors f and g, compute the multiplied factor h=fg. (5pts)
        """
        factor_table_f = self.bn.get_cpt(factor_f)
        factor_table_g = self.bn.get_cpt(factor_g)
        print(factor_table_f)
        print(factor_table_g)
        
        X = pd.DataFrame(columns = factor_table_f.columns)
        Y = pd.DataFrame(columns = factor_table_g.columns)
        
        Z = pd.merge(X,Y)

        l = [False, True]
        instantiations = [list(i) for i in itertools.product(l, repeat = len(Z.columns) - 1)]

        inst_df = pd.DataFrame(instantiations, columns=Z.columns[:-1])

        Z = Z.merge(inst_df, how='right')
        
        for i in range(len(inst_df)):
            # for j in range(inst_df):
            f = {}
            g = {} 
            for variable in inst_df.columns:
                if variable in factor_table_f.columns:
                    f[variable] = inst_df[variable][i]
                if variable in factor_table_g.columns:
                    g[variable] = inst_df[variable][i]
            
            f_series = pd.Series(f)
            g_series = pd.Series(g)

            comp_inst_f = self.bn.get_compatible_instantiations_table(f_series, self.bn.get_cpt(factor_f))
            comp_inst_g = self.bn.get_compatible_instantiations_table(g_series, self.bn.get_cpt(factor_g))
            value = comp_inst_f.p.values[0] * comp_inst_g.p.values[0]
            
            Z.at[i,'p'] = value 
        print(Z)
        return Z

    def variable_elimination(self, X: set[str], elim_method: Union[Literal['min_fill'], Literal['min_degree']]):
        """
        Variable Elimination: Sum out a set of variables by using variable elimination.
        (5pts)

        set X contains all the variables to eliminate.

        Following https://ermongroup.github.io/cs228-notes/inference/ve/
        """
        # to_eliminate = set(self.bn.get_all_variables()) - X
        if elim_method == 'min_degree':
            elim_order = self.min_degree_ordering(X)
        elif elim_method == 'min_fill':
            elim_order = self.get_minfill_orering(X)
        else:
            raise ValueError(f'elim_method {elim_method} not supported')
        for var in elim_order:
            children = self.bn.get_children(var)



    def min_degree_ordering(self, X):
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
        print(order)
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
        print(order)
        return order
   
    def lowest_fill(self, graph, X):
        lowest_fill = 100
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

    def marginal_distribution(self, Q, e):
        """Given query variables Q and possibly empty evidence e, 
        compute the marginal distribution P(Q|e). Note that Q is a subset of 
        the variables in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)"""
        #Reduce all factors w.r.t. e
        if not len(e)==0 :
            reduce_tables = self.bn.reduce_factor(e)
        else: 
            tables = self.bn.get_all_cpts()
         
        # calculation of joint marginal 
        #use variable elimination 

        # joint marginal by chain rule
        
        #sum out Q, to calculate Pr(e) 

        #Compute p(Q|e) = joint marginal/pr(e)


def test_prune():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    e = pd.Series({'Rain?': False})
    Q = {'Slippery Road?', 'Winter?'}
    reasoner.prune(Q, e)
    assert 'Wet Grass?' not in reasoner.bn.get_all_variables()
    assert 'Slippery Road?' not in reasoner.bn.get_children('Rain?')
    assert reasoner._pr('Slippery Road?') == 0.0
    assert reasoner._pr('Winter?') == 0.6


def test_dsep():
    """
    This tests the d-separation method on our BNReasoner class taking examples form the lecture
    notes PGM2_22.pdf page 33.
    """
    reasoner = BNReasoner('testing/lecture_example3.BIFXML')
    assert reasoner.dsep(set(['Visit to Asia?', 'Smoker?']), set(['Dyspnoea?', 'Positive X-Ray?']), set(['Bronchitis?', 'Tuberculosis or Cancer?']))
    assert reasoner.dsep(set(['Tuberculosis?', 'Lung Cancer?']), set(['Bronchitis?']), set(['Smoker?', 'Positive X-Ray?']))
    assert reasoner.dsep(set(['Positive X-Ray?', 'Smoker?']), set(['Dyspnoea?']), set(['Bronchitis?', 'Tuberculosis or Cancer?']))
    assert reasoner.dsep(set(['Positive X-Ray?']), set(['Smoker?']), set(['Lung Cancer?']))
    assert not reasoner.dsep(set(['Positive X-Ray?']), set(['Smoker?']), set(['Dyspnoea?', 'Lung Cancer?']))
    breakpoint()


def main():
    test_dsep()
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    reasoner.maxing_out('Wet Grass?', 'Sprinkler?')
    reasoner = BNReasoner('testing/lecture_example3.BIFXML')
    reasoner.factor_mult('Visit to Asia?', 'Tuberculosis?')


if __name__ == '__main__':
    main()
