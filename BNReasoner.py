import pandas as pd
from typing import List
from typing import Union
from BayesNet import BayesNet


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

    def __get_leaves(self, exclude=None) -> List[str]:
        exclude = set() if exclude is None else exclude
        return [v for v in self.bn.get_all_variables() 
                        if len(self.bn.get_children(v)) == 0]

    def __disconnected(self, X: set[str], Y: set[str]) -> bool:
        # TODO: Follow all children
        for x in X:
            if any(child in Y for child in self.bn.get_children(x)):
                return False
        for y in Y:
            if any(child in X for child in self.bn.get_children(y)):
                return False
        return True

    @staticmethod
    def __prune_variable(cpt: pd.DataFrame, var, pr: float) -> pd.DataFrame:
        cpt.loc[cpt[var] == True, 'p'] = cpt[cpt[var] == True].p.multiply(pr)
        cpt.loc[cpt[var] == False, 'p'] = cpt[cpt[var] == False].p.multiply(1-pr)
        all_other_vars = [col for col in cpt.columns if col not in [var, 'p']]
        cpt = cpt.groupby(all_other_vars, as_index=False).sum()
        del cpt[var]
        return cpt

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
            cpt = BNReasoner.__prune_variable(cpt, dep, pr_dep)
        return cpt[cpt[x] == True].p.values[0]

    def prune(self, Q: set[str], E: set[str]):
        """
        Given a set of query variables Q and evidence e, node- and edge-prune the 
        Bayesian network s.t. queries of the form P(Q|E) can still be correctly 
        calculated. (3.5 pts)
        """
        ignore_vars = ['p', *(list(Q.union(E)))]
        breakpoint()
        for q in Q.union(E):
            cpt = self.bn.get_cpt(q)
            deps = [col for col in cpt.columns if col not in ignore_vars]
            deps = self.get_ordered(deps)
            for dep in deps:
                pr = self._pr(dep)
                cpt = self.__prune_variable(cpt, dep, pr)
                self.bn.update_cpt(q, cpt)
            self.bn.update_cpt(q, cpt)
        
    def dsep2(self, X: set[str], Y: set[str], Z: set[str]) -> bool:
        pass

    def dsep(self, X: set[str], Y: set[str], Z: set[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated 
        of Y given Z. (4pts)
        """
        while True:
            leaves = self.__get_leaves(exclude=X.union(Y).union(Z))
            had_leaves = bool(len(leaves))
            zout_edges = []
            for z in Z:
                zout_edges.extend((z, child) for child in self.bn.get_children(z))
            had_edges = bool(len(zout_edges))
            if not had_edges and not had_leaves:
                break
            for edge in zout_edges:
                self.bn.del_edge(edge)
            for leaf in leaves:
                self.bn.del_var(leaf)
        return self.__disconnected(X, Y)

    def independent(self, X, Y, Z):
        """
        Independence: Given three sets of variables X, Y, and Z, determine whether X
        is independent of Y given Z. (Hint: Remember the connection between d-separation 
        and independence) (1.5pts)

        TODO: Upate that it also checks other independencies
        """
        return self.dsep(X, Y, Z)

    def marginalize(self, factor, x):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        """
        pass

    def maxing_out(self, factor, x):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember
        to also keep track of which instantiation of X led to the maximized value. (5pts)
        """
        pass


def main():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner._pr('Slippery Road?'))
    print(reasoner._pr('Rain?'))
    reasoner.prune(set(['Slippery Road?']), set(['Rain?']))
    breakpoint()

if __name__ == '__main__':
    main()
