import pytest
import pandas as pd

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

class TestMarginalisation:

    def test_case1(self, bn2):
        bayes = BayesNet()
        bayes.load_from_bifxml("./testing/lecture_example.BIFXML")

        all_ctp = bayes.get_all_cpts()
        test_cpt = all_ctp["Sprinkler?"]
        X = 'Winter?'
    
        outcome = bn2.marginalization(X, test_cpt)
        expected_outcome = pd.DataFrame({"Sprinkler?": [False, True], "p": [1.05, 0.95]})
   
        assert outcome.equals(expected_outcome)
        
class TestMaxingOut:
    
    def testcase1(self, bn2):
        bayes = BayesNet()
        bayes.load_from_bifxml("./testing/lecture_example.BIFXML")
        all_cpt = bayes.get_all_cpts()
        test_cpt = all_cpt["Wet Grass?"]
        X = 'Rain?'
       
        cpt = bn2.maxing_out(X, test_cpt)
        expected_cpt = pd.DataFrame({"Sprinkler": [False, False, True, True], "Wetgrass": [False, True, False, True], "p": [1.00, 0.80, 0.10, 0.95],
        f'extended factor {X}': [False, True, False, True]})

        assert not cpt.equals(expected_cpt)

class TestFactorMultiplication:

    def testcase1(self, bn2):
        bayes = BayesNet()
        bayes.load_from_bifxml("./testing/lecture_example.BIFXML")
        all_cpt = bayes.get_all_cpts()
        test_cpt1 = all_cpt["Winter?"]
        test_cpt2 = all_cpt["Rain?"]

        multiplication = bn2.factor_multiplication(test_cpt1, test_cpt2)
        expected = pd.DataFrame({"Winter": [False, False, True, True], "Rain?": [False, True, False, True], "p": [0.36, 0.04, 0.12, 0.48]})
        
        assert not multiplication.equals(expected)




        

