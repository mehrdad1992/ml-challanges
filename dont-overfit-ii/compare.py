import pandas as pd

res_local = pd.read_csv('dont-overfit-ii/data/submission7.csv')
res_remote = pd.read_csv('../../../Downloads/submission7.csv')
res_diff = abs(res_local['target'] - res_remote['target'])
count = (res_diff > 0.1).sum()
print(res_local.shape)
print(res_remote.shape)
print(count)
