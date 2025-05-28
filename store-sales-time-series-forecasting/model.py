import pandas as pd
import numpy as np

df_train = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/train.csv",
    index_col='id',
    parse_dates=['date'],
)

# y_train = df_train['sales']
# x_train = df_train.drop('sales', axis=1)

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
df_oil['dcoilwtico'] = df_oil['dcoilwtico'].fillna(method='bfill')
x_oil = df_oil.iloc[:-12,:]
y_oil = df_oil.iloc[-12:,:]

df_stores = pd.read_csv("store-sales-time-series-forecasting/dataset/stores.csv")

df_transactions = pd.read_csv("store-sales-time-series-forecasting/dataset/transactions.csv")

# get number of stores in train
store_arr = df_train['store_nbr'].unique()
store_nums = len(store_arr)

for store in store_arr:
  x_train_store = df_train[df_train['store_nbr'] == store]
  y_train_store = x_train_store['sales']
  x_train_store.drop('sales', axis=1, inplace=True)
#   x_train_store['oil'] = x_oil
  x_train_store = pd.merge(x_train_store, x_oil, on='date', how='outer')

  pass