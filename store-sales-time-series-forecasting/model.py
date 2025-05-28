import pandas as pd

df_train = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/train.csv",
    index_col='id',
    parse_dates=['date'],
)

df_test = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/test.csv",
    index_col='id',
    parse_dates=['date'],
)

df_sample = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/sample_submission.csv",
    index_col='id',
)

df_holidays = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/holidays_events.csv",
    parse_dates=['date'],
)

df_oil = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/oil.csv",
    parse_dates=['date'],
)

df_stores = pd.read_csv("store-sales-time-series-forecasting/dataset/stores.csv")

df_transactions = pd.read_csv("store-sales-time-series-forecasting/dataset/transactions.csv")

# get number of stores in train
store_arr = df_train['store_nbr'].unique()
store_nums = len(store_arr)

for store in store_arr:
  x_train = df_train.loc