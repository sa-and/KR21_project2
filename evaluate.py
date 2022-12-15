import pandas as pd


df = pd.read_csv("./experiment_results_new/all_results.csv", sep=",", index_col=0)
print(df)

df.groupby(["QE", "Prune", "Ordering"]).mean().hist()