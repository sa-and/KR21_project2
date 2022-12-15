import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ['QE', 'Prune', 'Ordering', 'Runtime']
df = pd.read_csv("experiment_results/all_results.csv", sep=",", index_col=0)
print(df)

df_unpruned = df[df['Prune'] == "unpruned"][['QE', 'Ordering', 'Runtime']].reset_index(drop=True)
df_unpruned = df_unpruned.groupby(["QE", "Ordering"]).mean()
df_unpruned.plot(kind='bar')

df_pruned = df[df['Prune'] == "pruned"][['QE', 'Ordering', 'Runtime']].reset_index(drop=True)
df_pruned
df_pruned = df_pruned.groupby(["QE", "Ordering"]).mean()
df_pruned.plot(kind='bar')

X = ['1:3', '2:2', '3:1']
min_degree = [0.022739, 0.044332, 0.049064]
min_fill = [0.021580, 0.044619, 0.044480]
random = [0.022713, 0.044431, 0.043591]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, min_degree, 0.2, label='min_degree')
plt.bar(X_axis, min_fill, 0.2, label='min_fill')
plt.bar(X_axis + 0.2, random, 0.2, label='random')

plt.xticks(X_axis, X)
plt.xlabel("Groups")
plt.ylabel("Average runtime")
plt.legend()
plt.show()

for prune in ['unpruned', 'pruned']:
    df2 = df[df['Prune'] == prune][['QE', 'Ordering', 'Runtime']].reset_index(drop=True)

# ==============================
print(df)
a = df.loc[(df['Prune'] == 'unpruned') & (df['QE'] == "1_3") & (df['Ordering'] == "min_fill")]['Runtime']
b = df.loc[(df['Prune'] == 'unpruned') & (df['QE'] == "1_3") & (df['Ordering'] == "min_degree")]['Runtime']
stats.ttest_ind(a, b, equal_var=False)

np.mean(a)
np.mean(b)

