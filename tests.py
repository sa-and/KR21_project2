import pandas as pd
from BNReasoner import BNReasoner

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
    assignments = reasoner.map(Q, e)
    assert assignments['I'] == False
    assert assignments['J'] == False

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
    assert not reasoner.dsep({'Lung Cancer?'}, {'Smoker?'}, set())

def test_factor_mult():
    """
    Tests factor multiplication
    """
    reasoner = BNReasoner('testing/lecture_example3.BIFXML')
    table1 = pd.DataFrame([
        [False, False, False, 1],
        [False, False, True, 0],
        [False, True, False, .2],
        [False, True, True, .8],
        [True, False, False, .1],
        [True, False, True, .9],
        [True, True, False, 0.05],
        [True, True, True, 0.95]
    ], columns=['B', 'C', 'D', 'p'])
    table2 = pd.DataFrame([
        [True, True, 0.448],
        [True, False, 0.192],
        [False, True, 0.112],
        [False, False, 0.248],
    ], columns=['D', 'E', 'p'])
    result = reasoner._factor_mult(table1, table2)
    assert result[(result['B'] == True) & (result['C'] == True) & (result['D'] == True) & (result['E'] == True)].p.sum() == 0.4256
    assert result[(result['B'] == True) & (result['C'] == True) & (result['D'] == True) & (result['E'] == False)].p.sum() == 0.1824
    assert abs(result[(result['B'] == True) & (result['C'] == True) & (result['D'] == False) & (result['E'] == True)].p.sum() - 0.0056) < 1e-10
    assert result[(result['B'] == False) & (result['C'] == False) & (result['D'] == False) & (result['E'] == False)].p.sum() == 0.248

def test_variable_elimination():
    """
    Taken from PGM3_22 page 18
    """
    reasoner = BNReasoner('testing/lecture_example4.BIFXML')
    reasoner.variable_elimination({'C?'})
    vars = reasoner.bn.get_all_variables()
    assert len(vars) == 1
    cpt = reasoner.bn.get_cpt(vars[0])
    assert abs(cpt[cpt['C?'] == False].p.sum() - 0.624) < 1e-10
    assert abs(cpt.p.sum() - 1.0) < 1e-10

def test_mpe():
    """
    Taken from PGM4_22 page 18
    """
    reasoner = BNReasoner('testing/lecture_example2.BIFXML')
    e = pd.Series({'J': True, 'O': False})
    result = reasoner.mpe(e)
    assert result['X'] == False
    assert result['Y'] == False
    assert result['I'] == False
    

def test_marginalize():
    """
    Taken from PGM3_22 page 15
    """
    reasoner = BNReasoner('testing/lecture_example2.BIfXML')
    df = pd.DataFrame([
        [False, False, False, 1],
        [False, False, True, 0],
        [False, True, False, .2],
        [False, True, True, .8],
        [True, False, False, .1],
        [True, False, True, .9],
        [True, True, False, 0.05],
        [True, True, True, 0.95]
    ], columns=['B', 'C', 'D', 'p'])
    result = reasoner._sum_out(df, 'D')
    assert all(result.p == 1)
    assert len(result.p) == 4
    trivial = reasoner._sum_out(reasoner._sum_out(result, 'C'), 'B')
    assert len(trivial.p) == 1
    assert trivial.p.iloc[0] == 4


def test_maxing_out():
    """
    Taken from PGM4_22 page 13
    """
    reasoner = BNReasoner('testing/lecture_example4.BIFXML')
    df = pd.DataFrame([
        [False, False, False, 1],
        [False, False, True, 0],
        [False, True, False, .2],
        [False, True, True, .8],
        [True, False, False, .1],
        [True, False, True, .9],
        [True, True, False, 0.05],
        [True, True, True, 0.95]
    ], columns=['B', 'C', 'D', 'p'])
    res, assignments = reasoner._maxing_out(df, 'D')

    assert res[(res['B'] == False) & (res['C'] == False)].p.sum() == 1.0
    assert assignments[res[(res['B'] == False) & (res['C'] == False)].index].iloc[0] == False

    assert res[(res['B'] == False) & (res['C'] == True)].p.sum() == 0.8
    assert assignments[res[(res['B'] == False) & (res['C'] == True)].index].iloc[0] == True

    assert res[(res['B'] == True) & (res['C'] == False)].p.sum() == 0.9
    assert assignments[res[(res['B'] == True) & (res['C'] == False)].index].iloc[0] == True

    assert res[(res['B'] == True) & (res['C'] == True)].p.sum() == 0.95
    assert assignments[res[(res['B'] == True) & (res['C'] == True)].index].iloc[0] == True


def test_marginal_distribution():
    """
    Taken from PGM4_22 page 9
    """
    reasoner = BNReasoner('testing/lecture_example4.BIFXML')
    Q = {'C?'}
    e = pd.Series({'A?': True})
    result = reasoner.marginal_distribution(Q, e)
    assert abs(round(result[result['C?'] == True].p.sum(), 2) - 0.32) < 1e-10
    assert abs(result['C?'].sum() - 1.0) < 1e-10

def run_all_tests(funcs):
    tests = [func for func in funcs if func.startswith('test_')]
    for test in tests:
        print(f'Executing test {test}')
        exec(f'{test}()')
        print('Passed')

if __name__ == '__main__':
    run_all_tests(dir())