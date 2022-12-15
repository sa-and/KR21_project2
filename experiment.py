from BNReasoner import BNReasoner
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

results_path = 'experiment_results'
if not os.path.exists(results_path):
    os.makedirs(results_path)
figsize = (8, 6)
save_results = True


def experiment(reasoner, Q: list, E: dict, prune: bool, ordering: str, n: int) -> list:
    if prune:
        pruned_net = reasoner.prune(Q, E)
        reasoner.bn = pruned_net

    runtime = []
    for i in range(n):
        start = time.time()
        reasoner.MAP(Q, E, ordering)
        runtime.append(time.time() - start)

    return runtime


if __name__ == "__main__":

    # set variables
    Q = ["Rain?"]
    E = {"Winter?": True, "Sprinkler?": False, "Wet Grass?": True}
    len_q = len(Q)
    len_e = len(E.keys())
    n = 100

    # load reasoner
    test_path = "testing/lecture_example.BIFXML"
    reasoner = BNReasoner(test_path)

    d = {}
    for pr in [False, True]:
        for ord in ["random", "min_fill", "min_degree"]:
            print(f"--- Running experiment with prune={pr} and ordering={ord} ---")
            if pr == False:
                d["unpruned, {0}".format(ord)] = experiment(reasoner, Q, E, prune=pr, ordering=ord, n=n)
            else:
                d["pruned, {0}".format(ord)] = experiment(reasoner, Q, E, prune=pr, ordering=ord, n=n)

    df_all = pd.DataFrame.from_dict(d)
    df_unpruned = df_all.iloc[:, :3]
    df_unpruned.columns = ["random", "min_fill", "min_degree"]
    df_pruned = df_all.iloc[:, 3:]
    df_pruned.columns = ["random", "min_fill", "min_degree"]

    print(f"Save results is set to {save_results}")
    if save_results:
        print(f"Saving results...")
        df_all.to_csv(os.path.join(results_path, f'results_q{len_q}_e{len_e}.csv'), sep=',')

        fig, axs = plt.subplots(figsize=(16,6))
        df_all.boxplot(ax=axs, fontsize=10)
        plt.ylabel("runtime (s)")
        fig.savefig(os.path.join(results_path, f'boxplot_all_q{len_q}_e{len_e}.png'))

        nr = 1
        for df in [df_unpruned, df_pruned]:
            fig, axs = plt.subplots(figsize=figsize)
            df.boxplot(ax=axs, fontsize=10)
            plt.ylabel("runtime (s)")
            if nr == 1:
                fig.savefig(os.path.join(results_path, f'boxplot_unpruned_q{len_q}_e{len_e}.png'))
            else:
                fig.savefig(os.path.join(results_path, f'boxplot_pruned_q{len_q}_e{len_e}.png'))
            nr += 1

    print(f"All done!")


