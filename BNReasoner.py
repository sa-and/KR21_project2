import pandas as pd
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
    
    def _compute_new_cpt(self, factor_table, x, which):
        """
        Given a factor and a variable X, compute the CPT in which X is either summed-out or maxed-out. (3pts)
        """
        cols = [*[c for c in factor_table.columns if c != 'p'], 'p']
        f = pd.DataFrame(columns = cols)
        f["p"]= []
        del f[x]
        if which == "max":
            f["instantiation"] = []

        l = [False, True]
        instantiations = [list(i) for i in itertools.product(l, repeat = len(factor_table.columns) - 2)]

        Y = cols[:-1]
        count = 0

        for inst in instantiations:
            inst_dict = {}
            inst_list = []
            for j in range((len(factor_table.columns) - 2)):
                inst_dict[Y[j]] = inst[j]
                inst_list.append(inst[j])
            inst_series = pd.Series(inst_dict)
            comp_inst = self.bn.get_compatible_instantiations_table(inst_series, factor_table)
            if which == 'max':
                new_p = comp_inst.p.max()  
                instantiation = comp_inst.loc[comp_inst["p"] == new_p][x].values[0]
                inst_list.append(new_p)
                inst_list.append(instantiation)
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
        return self._compute_new_cpt(self.bn.get_cpt(factor), x, 'sum')

    def maxing_out(self, factor: str, x):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember
        to also keep track of which instantiation of X led to the maximized value. (5pts)
        
        TODO: Keep track of which value of X this comes from
        """
        return self._compute_new_cpt(self.bn.get_cpt(factor), x, 'max')

    def _mult_as_factor(self, factor_table_f, factor_table_g):
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

            comp_inst_f = self.bn.get_compatible_instantiations_table(f_series, factor_table_f)
            comp_inst_g = self.bn.get_compatible_instantiations_table(g_series, factor_table_g)

            value = comp_inst_f.p.sum() * comp_inst_g.p.sum()
            
            Z.at[i,'p'] = value 

        return Z


    def factor_mult(self, factor_f, factor_g):
        """
        Given two factors f and g, compute the multiplied factor h=fg. (5pts)
        """
        factor_table_f = self.bn.get_cpt(factor_f)
        factor_table_g = self.bn.get_cpt(factor_g)
        return self._mult_as_factor(factor_table_f, factor_table_g)

    def _eliminate_factor_or_variable(self, vorf: str, nname: str) -> pd.DataFrame:
        d_factors = set(self.bn.get_children(nname))
        if len(d_factors) == 0: # if no children, eliminate immediately
            return None
        cpt = self.bn.get_cpt(nname)
        for child in d_factors:
            child_cpt = self.bn.get_cpt(child)
            cpt = self._mult_as_factor(cpt, child_cpt)
        
        cpt = self._compute_new_cpt(cpt, vorf, 'sum')
        return cpt

    def _get_elim_order(self, vars, elim_method):
        if elim_method == 'min_degree':
            return self.min_degree_ordering(vars)
        elif elim_method == 'min_fill':
            return self.minfill_ordering(vars)
        else:
            raise ValueError(f'elim_method {elim_method} not supported')

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
            cpt = self._mult_as_factor(cpt, neighbor_cpt) 
        cpt = self._compute_new_cpt(cpt, var, 'sum') # Sum out variable
        cpt.p /= cpt.p.sum() # normalize for joint probability
        return cpt, neighbors

    def variable_elimination(self, X: set[str], elim_method: Union[Literal['min_fill'], Literal['min_degree']] = 'min_degree'):
        """
        Variable Elimination: Sum out a set of variables by using variable elimination.
        (5pts)

        set X contains all the variables to not eliminate.

        Following https://ermongroup.github.io/cs228-notes/inference/ve/
        """
        to_eliminate = set(self.bn.get_all_variables()) - X
        to_eliminate = self._get_elim_order(to_eliminate, elim_method)
        while len(to_eliminate) > 0:
            var = to_eliminate.pop(0)
            cpt, neighbors = self._reduce_variable(var)
            fname = self.get_factor_name(cpt)
            for neighbor in neighbors:
                self.bn.del_var(neighbor)
            self.bn.add_var(fname, cpt)
        return cpt

    @staticmethod
    def get_factor_name(cpt: pd.DataFrame):
        cols = sorted([c for c in cpt.columns if c != 'p'])
        return 'f_'+','.join(cols)

    def get_node_or_factor_name(self, norf: str):
        all_vars = self.bn.get_all_variables()
        if norf in all_vars:
            return norf
        factor_names = [v for v in all_vars if v.startswith('f_') and (f'_{norf}' in v or f',{norf}' in v)]
        if len(factor_names) == 0:
            return None
        return factor_names[0]

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

    def map(self, Q: set[str], e: pd.Series, ret_jptd=False, elim_method='min_degree'):
        """Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)"""
        vars_to_keep = Q.union(set(e.index))
        self.prune(Q, e)
        jptd = self.variable_elimination(vars_to_keep, elim_method)
        jptd = BayesNet.get_compatible_instantiations_table(e, jptd)
        # for q in Q:
        #     jptd = self._compute_new_cpt(jptd, q, which='max')

        row = jptd.loc[pd.to_numeric(jptd.p).idxmax()]
        for i in e.index:
            del row[i]
        del row['p']
        if ret_jptd:
            return row, jptd
        return row


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

def test_map():
    """
    Test taken from PGM4_22.pdf page 20.
    """
    reasoner = BNReasoner('testing/lecture_example2.BIFXML')
    Q = {'I', 'J'}
    e = pd.Series({'O': True})
    assignments, jpt = reasoner.map(Q, e, ret_jptd=True)
    assert assignments['I'] == True
    assert assignments['J'] == False


def test_variable_elimination():
    reasoner = BNReasoner()


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


def main():
    test_map()
    # reasoner = BNReasoner('testing/lecture_example3.BIFXML')

    # e = pd.Series({'Smoker?': True})
    # Q = {'Lung Cancer?', 'Dyspnoea?', 'Positive X-Ray?'}
    # # jptd = reasoner.variable_elimination(set(['Smoker?', 'Tuberculosis?']), 'min_degree')
    # print(reasoner.map(Q, e))
    # reasoner.variable_elimination(Q)

if __name__ == '__main__':
    main()
