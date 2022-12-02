from typing import Union
from BayesNet import BayesNet
from copy import deepcopy
import itertools
import random
import pandas as pd


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
        self.options = [True, False]
        self.ordering = {"minFillOrder" : self.order_min_fill(), "minDegreeOrder": self.order_min_degree()}


    # TODO: This is where your methods should go

    # UTIL FUNCTIONS----------------------------------------------------------------------------------------------------

    def get_leaf_nodes(self, bn):
        """
        Checks for the variables that do not have any children
        :param bn: Bayesian network
        :returns: a list of variable names that are leaf nodes
        """
        return [var for var in bn.get_all_variables() if not bn.get_children(var)]
    
    def get_all_paths(self, bn, start_nodes: list, end_nodes: list):
        """
        Find a path from starting_node to ending_node
        :param bn: Bayesian Network
        :param start_nodes: The nodes we want to start the search from
        :param end_nodes: The nodes we want to find a path to
        :return False if the path is found, True otherwise
        """
        visited = []
        queue = start_nodes
        while queue:
            node = queue.pop(0)
            if node in end_nodes:
                return False
            if node not in visited:
                visited.append(node)
                neighbors = itertools.chain(bn.structure.neighbors(node), bn.structure.predecessors(node))
                for neighbor in neighbors:
                    queue.append(neighbor)
        return True

    def union_of_cpts(self, cpts):
        """
        :return A union of the cpts given as input
        """
        temp_cpt = [cpt.drop(columns='p') for cpt in cpts]
        temp_cpt = list(temp_cpt)
        z = temp_cpt[0]
        for i in range(1, len(temp_cpt)):
            z = pd.merge(z, temp_cpt[i])
        z['p'] = [1] * z.shape[0]
        z.reset_index(inplace=True, drop=True)
        return z

    # ALGORITHMS -------------------------------------------------------------------------------------------------------

    def network_pruning(self, query: list, evidence: dict) -> BayesNet:
        """
        Prunes edges and nodes given a set of query variables Q and evidence e
        :param query: variables to prune
        :param evidence: evidence as dictionary of vars and values
        :returns: pruned bayesian network
        """
        bn_new = deepcopy(self.bn)

        # Remove edges
        for var in evidence.keys():
            cpt = bn_new.get_cpt(var)
            bn_new.update_cpt(var, cpt[cpt[var] == evidence[var]])  # update cpt vars
            for children in bn_new.get_children(var):
                bn_new.del_edge([var, children])  # Remove edges
                cpt = bn_new.get_cpt(children)
                bn_new.update_cpt(children, cpt[cpt[var] == evidence[var]])  # update cpt children

        # Remove leaf nodes
        while True:  # Create an iterative loop
            leaf_nodes = self.get_leaf_nodes(bn_new)
            for var in leaf_nodes:
                if var not in query + list(evidence.keys()):
                    bn_new.del_var(var)
            if not set(self.get_leaf_nodes(bn_new)) - set(query + list(evidence.keys())):  # Break when only leaf nodes are in x + y + z
                break
        return bn_new

    def d_separation(self, X: list, Y: list, Z: list) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z.
        :param X: variable possibly d-separated of Y given Z
        :param Y: variable
        :param Z: variable given
        :returns: Boolean; z if X is d-separated of Y given Z, False otherwise
        """
        bn_new = deepcopy(self.bn)
        
        # Delete all leaf nodes
        leaf_nodes = self.get_leaf_nodes(bn_new)
        for var in leaf_nodes:
            if var not in X + Y + Z:
                bn_new.del_var(var)

        # Delete edges from Z
        for var in Z:
            for children in bn_new.get_children(var):
                bn_new.del_edge([var, children])
        while True:
            leaf_nodes = self.get_leaf_nodes(bn_new)
            for var in leaf_nodes:
                if var not in X + Y + Z:
                    bn_new.del_var(var)
            if not set(self.get_leaf_nodes(bn_new)) - set(X + Y + Z):  # Break when only leaf nodes are in X + Y + Z
                break

        return self.get_all_paths(bn_new, X, Y)

    def independence(self):
        pass

    def sum_out(self, f, var):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out
        :return A cpt with variables in ordering summed out
        """
        f.drop(columns=var, inplace=True)
        cols = [col for col in f.columns if col != 'p']
        toAdd = []
        for i, row in f.iterrows():
            toCheck = [row[col] for col in cols]
            for i2, row2 in f.iterrows():
                toCompare = [row2[col] for col in cols]
                if toCheck == toCompare and i2 != i and (i2, i) not in toAdd:
                    toAdd.append((i, i2))
        for i in range(len(toAdd)):
            f.iloc[toAdd[i][0], -1] += f.iloc[toAdd[i][1], -1]
        for i in range(len(toAdd)):
            f.drop(toAdd[i][1], inplace=True)
        f.reset_index(inplace=True, drop=True)
        return f

    def maxing_out(self, f, checked):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out
        :return A cpt with variables in ordering summed out
        """
        cols = [col for col in f.columns if (col != 'p') and (col not in checked)]
        toAdd = []
        for i, row in f.iterrows():
            toCheck = [row[col] for col in cols]
            for i2, row2 in f.iterrows():
                toCompare = [row2[col] for col in cols]
                if toCheck == toCompare and i2 != i and (i2, i) not in toAdd:
                    toAdd.append((i, i2))
        toDrop = set()
        for i in range(len(toAdd)):
            if f.iloc[toAdd[i][1], -1] <= f.iloc[toAdd[i][0], -1]: toDrop.add(toAdd[i][1])
            else: toDrop.add(toAdd[i][0])
        for i in toDrop:
            f.drop(i, inplace=True)
        f.reset_index(inplace=True, drop=True)
        return f

    def factor_multiplication(self, cpts: list):
        """
        Multiplies the corresponding rows of the cpts
        :return z: a cpt containing all the p-values of the cpts
        """
        z = self.union_of_cpts(cpts)
        for cpt in cpts:
            for i, row_content in cpt.iterrows():
                for i_z, row_content_z in z.iterrows():
                    if all([row_content_z[c] == row_content[c] for c in cpt.columns if c != 'p']):
                        z.iloc[i_z, -1] *= row_content['p']
        return z


    def order_random(self, query: list, bn=None):
        """
        Given a set of variables X, compute a random ordering
        :returns an ordering pi of variables of the BN
        """
        if not bn:
            bn_new = deepcopy(self.bn)

        vars = bn_new.get_all_variables()
        if "p" in vars:
            bn_new.del_var('p')
            vars = bn_new.get_all_variables()
        for item in query:
            vars.remove(item)
        random.shuffle(vars)
        order = vars
        order.extend(query)
        return order

    def order_min_degree(self, bn=None):
        """
        Given a set of variables X, compute a good ordering for the elimination of X based on the min-degree heuristics
        :returns an ordering pi of variables of the BN
        """
        if not bn:
            bn_new = deepcopy(self.bn)

        graph = bn_new.get_interaction_graph()
        vars = bn_new.get_all_variables()
        pi = []
        for i in range(len(vars)):
            n_neighbours = [len(graph.adj[node].items()) for node in vars]
            pi.append(vars[n_neighbours.index(min(n_neighbours))])
            # Add edge between every pair of non-adjacent neighbors of pi(i) in G
            neighbors = [s for s, _ in graph.adj[pi[i]].items()] + [pi[i]]
            for subset in itertools.combinations(neighbors, 2):
                if not graph.has_edge(*subset):
                    graph.add_edge(*subset)
            # delete variable pi(i) from G and from X
            graph.remove_node(pi[i])
            vars.remove(pi[i])
        return pi

    def order_min_fill(self, bn=None):
        """
        Given a set of variables X, compute a good ordering for the elimination of X based on the min-fill heuristics
        :returns an ordering pi of variables of the BN
        """
        if not bn:
            bn_new = deepcopy(self.bn)

        graph = bn_new.get_interaction_graph()
        vars = bn_new.get_all_variables()
        pi = []
        for i in range(len(vars)):
            n_edges = []
            for node in vars:
                n = [s for s, _ in graph.adj[node].items()]
                if len(n) == 1:
                    n_edges.append(0)
                else:
                    n_edges.append(sum([0 if graph.has_edge(*subset) else 1 for subset in itertools.combinations(n, 2)]))
            pi.append(vars[n_edges.index(min(n_edges))])
            # Add edge between every pair of non-adjacent neighbors of pi(i) in G
            neighbors = [s for s, _ in graph.adj[pi[i]].items()] + [pi[i]]
            for subset in itertools.combinations(neighbors, 2):
                if not graph.has_edge(*subset):
                    graph.add_edge(*subset)
            # delete variable pi(i) from G and from X
            graph.remove_node(pi[i])
            vars.remove(pi[i])
        return pi

    def variable_elimination(self):
        pass

    def marginal_distributions(self, query: list, evidence: dict):
        """
        Calculates the marginal probability of the query variables Q, given possible evidence
        :param query: variables to prune
        :param evidence: evidence as dictionary of vars and values
        :returns: CPT for the query variables Q
        """
        bn_new = self.network_pruning(query, evidence)
        cpts = bn_new.get_all_cpts()
        pi = self.order_min_degree(bn_new)
        for evidence in evidence.keys():
            del cpts[evidence]
            pi.remove(evidence)
        for node in query:
            pi.remove(node)

        for i in range(len(pi)):
            fk = {cp: cpt for cp, cpt in cpts.items() if pi[i] in cpt.columns}
            f = self.factor_multiplication(fk.values())
            fi = self.sum_out(f, pi[i])
            for k in fk.keys(): del cpts[k]
            cpts['f' + str(i)] = fi
        return cpts

    def map(self, vars, evidence, ordering="minFill"):
        """
        Maximises the value and instantiation of variables m given evidence e
        :return instantiation and value
        """
        global factor
        bn = self.network_pruning(vars, evidence)  # Prune network
        pi = [self.order_min_fill(bn) if ordering == "minFill" else self.order_min_degree(bn)][0]  # Elimination order
        pi = [v for v in pi if v not in vars] + vars
        q = bn.get_all_variables()
        s = {var: bn.get_cpt(var) for var in q}  # Create a single table based on pi (and possible evidence e)
        for i in range(len(pi)):
            fk = {cp: cpt for cp, cpt in s.items() if pi[i] in cpt.columns}
            f = self.factor_multiplication(fk.values())
            if pi[i] in vars:
                fi = self.maxing_out(f, [pi[i]])
            else:
                fi = self.sum_out(f, pi[i])
            for k in fk.keys(): del s[k]
            factor = 'f' + str(i)
            s[factor] = fi
        # Return the most likely instantiation and it's probability
        return s[factor].iloc[s[factor]['p'].idxmax()]

    def mpe(self, evidence, ordering="minFill"):
        """
        Finds the Most Probable Explanation given possible evidence
        :return instantiation and value
        """
        bn_new = deepcopy(self.bn)
        checked_vars = []

        # Remove edges
        for var in evidence.keys():
            cpt = bn_new.get_cpt(var)
            bn_new.update_cpt(var, cpt[cpt[var] == evidence[var]])  # update cpt vars
            for children in bn_new.get_children(var):
                bn_new.del_edge([var, children])  # Remove edges
                cpt = bn_new.get_cpt(children)
                bn_new.update_cpt(children, cpt[cpt[var] == evidence[var]])  # update cpt children
        
        q = bn_new.get_all_variables()
        pi = [self.order_min_fill(bn_new) if ordering == "minFill" else self.order_min_degree(bn_new)][0]
        s = {var: bn_new.get_cpt(var) for var in q}
        for i in range(len(q)):
            fk = {cp : cpt for cp, cpt in s.items() if pi[i] in cpt.columns}
            f = self.factor_multiplication(fk.values())
            checked_vars.append(pi[i])
            fi = self.maxing_out(f, checked_vars)
            for k in fk.keys(): del s[k]
            s['f' + str(i)] = fi
        return s

    # def extend_random_network(self, num_nodes=int, num_edges=tuple):
    #     """
    #     Extends the already existing bayesian network with variables and edges connecting them to random nodes
    #     For the purposes of this project the probabilities are always initialized to be 0.5
    #     :arg num_nodes: The number of nodes the network should be extended to
    #     :arg num_edges: A tuple containing the minimum and maximum number of edges each node should be connected with
    #     :returns a randomly generated bayesian network
    #     """
    #     for i in range(num_nodes):
    #         new_node = "node_" + str(i)
    #         edges = random.sample(self.bn.get_all_variables(), k=random.choice([i for i in range(*num_edges)]))
    #         cols = edges + [new_node]
    #         cpt = pd.DataFrame(list(itertools.product(*[self.options for i in range(len(cols))])), columns=cols)
    #         cpt['p'] = 0.5
    #         self.bn.add_var(new_node, cpt=cpt)
    #         for node in edges:
    #             self.bn.add_edge((node, new_node))
    #     return self.bn
