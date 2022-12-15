from typing import Union
from BayesNet import BayesNet
from copy import deepcopy
from networkx.utils import UnionFind
import pandas as pd
import networkx as nx
import itertools


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

    # network pruning
    def prune(self, Q: list, E: dict) -> BayesNet:

        output_net = deepcopy(self.bn)

        # edge prunning
        for var in E.keys():
            # update conditional probability table based on evidence
            cpt = output_net.get_cpt(var)
            output_net.update_cpt(var, cpt[cpt[var] == E[var]])

            children = output_net.get_children(var)
            for child in children:
                # remove the edge
                output_net.del_edge((var, child))
                # update the conditional probability table
                cpt = output_net.get_cpt(child)
                output_net.update_cpt(child, cpt[cpt[var] == E[var]])

        # leaf nodes prunning
        while True:
            leaf_vars = [var for var in output_net.get_all_variables() if not output_net.get_children(var)]

            # if there are no more nodes to prune
            if not set(leaf_vars) - set(Q + list(E.keys())):
                break

            for leaf in leaf_vars:
                if leaf not in Q and leaf not in E:
                    output_net.del_var(leaf)

        return output_net

    # d-separation
    def are_d_seperated(self, X: list, Y: list, Z: list) -> bool:
        # make sure that X, Y and Z are sets and conists of unique variables
        X = set(X)
        Y = set(Y)
        Z = set(Z)

        graph = deepcopy(self.bn)
        XYZ = X.union(Y).union(Z)

        # remove leaf nodes that are not in union of X, Y, Z
        while True:
            leaf_vars = [var for var in graph.get_all_variables() if not graph.get_children(var)]
            # if there are no more nodes to prune
            if not set(leaf_vars) - XYZ:
                break

            for leaf in leaf_vars:
                if leaf not in XYZ:
                    graph.del_var(leaf)

        # remove outgoing edges for each node from Z
        edges = deepcopy(graph.structure.out_edges(Z))
        for edge in edges:
            graph.del_edge(edge)

        # check if X and Y are disconnected in the graph
        # create empty UnionFind instance
        disjoint_set = UnionFind(graph.structure.nodes())
        # get weakly connected components
        for wcc in nx.weakly_connected_components(graph.structure):
            disjoint_set.union(*wcc)

        disjoint_set.union(*X)
        disjoint_set.union(*Y)

        # early condition if any of sets is empty
        if not X or not Y:
            return False

        # to check if X is not d-seperated from Y, we just have to check if any of vars from X is in Y
        for x in X:
            for y in Y:
                if disjoint_set[y] == disjoint_set[x]:
                    return False

        # X and Y are disconnected in pruned graph
        return True

    def are_independent(self, X: list, Y: list, Z: list) -> bool:
        # d-separation implies an independence
        if self.are_d_seperated(X, Y, Z):
            return True

        # if there is a direct edge then X and Y are not independent
        edges = self.bn.structure.edges()
        for x in X:
            for y in Y:
                if (x, y) in edges or (y, x) in edges:
                    return False

        return True

    # marginalize by summing-out
    def marginalize(self, factor: pd.DataFrame, var: str) -> pd.DataFrame:
        vars = [col_name for col_name in factor.columns if col_name not in [var, "p", " "]]
        marginalized = factor.groupby(vars).sum().reset_index()
        if 'p' in marginalized.columns:
            marginalized = marginalized.drop(var, axis=1)

        return marginalized

    def factors_multiplication(self, factors: list) -> pd.DataFrame:
        # remove None elements from factors list
        factors = [factor for factor in factors if factor is not None]
        # construct empty output factor as a truth table
        vars = [x.columns.values.tolist() for x in factors]
        vars = set(itertools.chain.from_iterable(vars))
        vars.remove('p')
        vars = list(vars)

        # get all possible instantiations per var
        vars_values = dict()
        for factor in factors:
            for col in factor:
                if col == 'p':
                    continue
                if col not in vars_values:
                    vars_values[col] = factor[col].unique().tolist()
                else:
                    vars_values[col] += factor[col].unique().tolist()

        # remove duplicates
        for key, val in vars_values.items():
            vars_values[key] = list(set(val))

        # dict to list in order of columns
        possible_values_ordered_by_column_name = []
        for var in vars:
            possible_values_ordered_by_column_name.append(vars_values[var])

        # construct truth_table as cartesian product of all possible instantiations
        truth_table = pd.DataFrame(list(itertools.product(*possible_values_ordered_by_column_name)), columns=vars)
        truth_table['p'] = 1.0

        for factor in factors:
            vars = set(factor.columns.values.tolist())
            vars.remove('p')
            vars = list(vars)
            for index, row in truth_table.iterrows():
                series = pd.Series(row)
                series = series[vars]
                p = self.bn.get_compatible_instantiations_table(series, factor).iloc[0]['p']
                truth_table.at[index, 'p'] *= p

        return truth_table

    def min_degree_ordering(self, X: list) -> list:
        graph = self.bn.get_interaction_graph()
        degrees = [x for x in graph.degree() if x[0] in X]
        degrees.sort(key=lambda x: x[1])
        ordered = [x[0] for x in degrees]
        return ordered

    def min_fill_ordering(self, X: list):
        graph = self.bn.get_interaction_graph()
        neighbours = [(x, [n for n in graph.neighbors(x)]) for x in X]
        edges = graph.edges
        min_fill_degrees = []
        for n in neighbours:
            node_name = n[0]
            score = 0
            n_len = len(n[1])
            for i in range(n_len):
                for j in range(i, n_len):
                    # if edge not in graph
                    if not ((i, j) in edges or (j, i) in edges):
                        score += 1
            min_fill_degrees.append((node_name, score))

        min_fill_degrees.sort(key=lambda x: x[1])
        ordered = [x[0] for x in min_fill_degrees]

        return ordered

    def variable_elimination(self, factors: list, X: list, ordering="min_degree") -> pd.DataFrame:
        if ordering == "min_degree":
            X = self.min_degree_ordering(X)
        else:
            X = self.min_fill_ordering(X)

        prev_factor = None
        for x in X:
            x_cpts = [cpt for cpt in factors if x in cpt.columns]
            factors = [cpt for cpt in factors if x not in cpt.columns]
            # add prev_factor
            x_cpts += [prev_factor]
            multiplied = self.factors_multiplication(x_cpts)
            marginalized = self.marginalize(multiplied, x)
            prev_factor = marginalized

        # prev_factor after all iterations is a final marginalized factor
        return prev_factor

    def marginal_distribution(self, Q: list, E: dict) -> pd.DataFrame:
        all_cpts = self.bn.get_all_cpts()
        # reduce factors due to evidence E
        reduced_cpts = []
        vars_to_eliminate = set()
        for cpt in all_cpts.values():
            # remove rows that are not compatible with E
            evidence = pd.Series(E)
            reduced_cpt = self.bn.get_compatible_instantiations_table(evidence, cpt)
            reduced_cpts.append(reduced_cpt)
            vars = [x for x in reduced_cpt.columns if x not in Q]
            vars_to_eliminate.update(vars)

        # variable elimination
        vars_to_eliminate.remove('p')
        vars_to_eliminate = list(vars_to_eliminate)

        posterior_marginal = self.variable_elimination(reduced_cpts, vars_to_eliminate)

        return posterior_marginal

    # marginalize by maxing-out
    def max_out(self, factor: pd.DataFrame, vars: list) -> tuple:
        grouping_vars = [col_name for col_name in factor.columns if col_name not in ["p", " "] + vars]
        # it is the same as summing out but instead of summing grouped rows we choose the one with max p
        extended = factor.groupby(grouping_vars).max().reset_index()

        marginalized = extended
        for v in vars:
            marginalized = marginalized.drop(v, axis=1)

        return marginalized, extended

    # Maximum A-Posteriori Query
    # it also checks for MPE case
    def MAP(self, Q: list, E: dict) -> dict:
        all_variables = self.bn.get_all_variables()
        MPE = Q == [var for var in all_variables if var not in E.keys()]

        if MPE:
            print("using mpe")
            all_cpts = self.bn.get_all_cpts()
            factor = self.factors_multiplication(list(all_cpts.values()))
            # drop rows incompatible with instantiation
            for key, val in E.items():
                factor = factor.loc[factor[key] == val]
            distribution = factor.reset_index()


        else:
            distribution = self.marginal_distribution(Q, E)

        # return row with highest p
        idx_max = distribution['p'].idxmax()
        max = distribution.iloc[idx_max]

        p = 0
        instantiations = dict()
        for val in max.items():
            if val[0] == 'p':
                p = val[1]
            else:
                instantiations[val[0]] = val[1]

        return p, instantiations