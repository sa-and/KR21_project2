import pytest

from BayesNet import BayesNet
from BNReasoner import BNReasoner

# --------------------------------- Test BNs ----------------------------------
# Dog problem (given)
@pytest.fixture
def bn1():
    return BNReasoner("./testing/dog_problem.BIFXML")

# Lecture example 1 (given)
@pytest.fixture
def bn2():
    return BNReasoner("./testing/lecture_example.BIFXML")

# Lecture example 2 (given)
@pytest.fixture
def bn3():
    return BNReasoner("./testing/lecture_example2.BIFXML")

# Example for d-sep from lecture 2
@pytest.fixture
def bn4():
    variables = ['A', 'S', 'T', 'C', 'P', 'B', 'X', 'D']
    edges = [('A', 'T'), ('S', 'C'), ('S', 'B'), ('T', 'P'), ('C', 'P'), ('P', 'X'), ('P', 'D'), ('B', 'D')]
    cpts = {}
    for v in variables:
        cpts[v] = None

    bn = BayesNet()
    bn.create_bn(variables, edges, cpts)

    return BNReasoner(bn)

# ----------------------------------- Tests -----------------------------------

class TestDSeparation:
    def test_case1(self, bn4):
        X = ['B']
        Y = ['C']
        Z = ['S']
        assert bn4.is_dsep(X, Y, Z)

    def test_case2(self, bn4):
        X = ['X']
        Y = ['S']
        Z = ['C', 'D']
        assert not bn4.is_dsep(X, Y, Z)

