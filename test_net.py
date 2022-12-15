from BNReasoner import BNReasoner
import time


def test_d_separation(reasoner):
    print("--- test d separation ---")
    X = ["Winter?", "Slippery Road?"]
    Y = ["Rain?"]
    Z = ["Sprinkler?"]
    print(f"Are X: {X} and Y: {Y} d-separated by Z: {Z}?")
    print(reasoner.are_d_seperated(X, Y, Z))


def test_independence(reasoner):
    print("--- test independence ---")
    X = ["Sprinkler?"]
    Y = ["Rain?"]
    Z = ["Winter?"]
    print(f"Are X: {X} and Y: {Y} independent given Z: {Z}?")
    print(reasoner.are_independent(X, Y, Z))


def test_marginalization(reasoner):
    print("--- test marginalization ---")
    factor = reasoner.bn.get_all_cpts()["Wet Grass?"]
    print(f"before maginalization:\n{factor}")
    marginalized = reasoner.marginalize(factor, "Wet Grass?")
    print(f"after marginalization:\n{marginalized}")


def test_pruning(reasoner):
    Q = ["Rain?"]
    E = {"Winter?": True, "Sprinkler?": False, "Wet Grass?": True}
    print("==map on unpruned net==")
    p, instantiations = reasoner.MAP(Q, E)
    print(f"most likely instantion of Q: {Q}:\n{instantiations}\n with p={p}")
    pruned_net = reasoner.prune(Q, E)
    reasoner.bn = pruned_net
    # run MAP on pruned_net
    print("==map on pruned net==")
    p, instantiations = reasoner.MAP(Q, E)
    print(f"most likely instantion of Q: {Q}:\n{instantiations}\n with p={p}")


def test_factors_multiplication(reasoner):
    print("--- test factors multiplication ---")
    # test factor multiplication
    f1 = reasoner.bn.get_cpt("Winter?")
    f2 = reasoner.bn.get_cpt("Slippery Road?")
    print("Factors")
    print(f"f1: \n {f1}")
    print(f"f2: \n {f2}")
    out = reasoner.factors_multiplication([f1, f2])
    print(f"output factor: \n {out}")


def test_min_degree_ordering(reasoner):
    print("--- test min degree ordering ---")
    X = ["Winter?", "Sprinkler?", "Slippery Road?"]
    order = reasoner.min_degree_ordering(X)
    print(f"order:\n {order}")


def test_min_fill_ordering(reasoner):
    print("--- test min fill ordering ---")
    X = ["Winter?", "Sprinkler?", "Slippery Road?"]
    order = reasoner.min_fill_ordering(X)
    print(f"order:\n {order}")


def test_variable_elimination(reasoner):
    print("--- test variable elimination ---")
    all_cpts = reasoner.bn.get_all_cpts()
    print("==ALL CPTS==")
    for cpt in all_cpts.values():
        print(cpt)
    print("===========")
    X = ["Winter?", "Sprinkler?", "Slippery Road?", "Wet Grass?"]
    print(f"vars: {X}")
    all_cpts = reasoner.bn.get_all_cpts().values()
    eliminated = reasoner.variable_elimination(all_cpts, X)
    print("==========")
    print(eliminated)


def test_marginal_distribution(reasoner):
    print("--- test marginal distribution ---")
    Q = ["Rain?"]
    E = {"Winter?": True, "Sprinkler?": False, "Wet Grass?": True}
    print("==marginal distribution==")
    posterior_marginal = reasoner.marginal_distribution(Q, E)
    print(posterior_marginal)


def test_max_out(reasoner):
    print("--- test maxing out ---")
    vars = ["Rain?", "Sprinkler?"]
    factor = reasoner.bn.get_all_cpts()["Wet Grass?"]
    print(f"maxing out {vars} in factor: \n {factor}")
    maxed_out, extended = reasoner.max_out(factor, vars)
    print("==maxed out==")
    print(maxed_out)
    print(f"extended:\n {extended}")


def test_map(reasoner):
    print("--- test MAP ---")
    Q = ["Rain?", "Wet Grass?"]
    E = {"Winter?": False}
    p, instantiations = reasoner.MAP(Q, E)
    print(f"most likely instantion of Q: {Q}:\n{instantiations}\n with p={p}")


def test_mpe(reasoner):
    print("--- test MPE ---")
    Q = ["Rain?", "Wet Grass?"]
    E = {"Winter?": False, "Sprinkler?": True, "Slippery Road?": True}
    p, instantiations = reasoner.MAP(Q, E)
    print(f"most likely instantion of Q: {Q} given evidence E: {E}:\n{instantiations}\n with p={p}")


if __name__ == "__main__":
    reasoner = BNReasoner("testing/lecture_example.BIFXML")
    # test_min_degree_ordering(reasoner)
    # test_min_fill_ordering(reasoner)
    test_variable_elimination(reasoner)
    # test_marginal_distribution(reasoner)
    # test_max_out(reasoner)
    # test_map(reasoner)
    # test_mpe(reasoner)
    # test_d_separation(reasoner)
    # reasoner.bn.draw_structure()
    # test_independence(reasoner)
    # test_marginalization(reasoner)
    # test_pruning(reasoner)
