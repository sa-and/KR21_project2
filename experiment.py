from BNReasoner import BNReasoner
import time


def experiment(reasoner):
    # first obj to be compared
    start1 = time.time()
    X = ["Winter?", "Sprinkler?", "Slippery Road?", "Wet Grass?"]
    all_cpts = reasoner.bn.get_all_cpts().values()
    eliminated = reasoner.variable_elimination(all_cpts, X, ordering="min_degree", heuristic="sum")
    end1 = time.time()
    time1 = end1 - start1
    print(time1)

    # second thing
    start2 = time.time()
    X = ["Winter?", "Sprinkler?", "Slippery Road?", "Wet Grass?"]
    all_cpts = reasoner.bn.get_all_cpts().values()
    eliminated = reasoner.variable_elimination(all_cpts, X, ordering="min_fill", heuristic="sum")
    end2 = time.time()
    time2 = end2 - start2
    print(time2)


if __name__ == "__main__":

    test_path = "testing/lecture_example.BIFXML"

    reasoner = BNReasoner(test_path)
    experiment(reasoner)