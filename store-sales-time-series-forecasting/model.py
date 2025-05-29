import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/train.csv",
    index_col='id',
    parse_dates=['date'],
)
# becaues of earthquake
df_train = df_train[~((df_train['date'] >= "2016-05-16") & (df_train['date'] <= "2016-05-31"))]

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
# becaues of earthquake
df_holidays = df_holidays[~((df_holidays['date'] >= "2016-05-16") & (df_holidays['date'] <= "2016-05-31"))]

df_oil = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/oil.csv",
    parse_dates=['date'],
)
# becaues of earthquake
df_oil = df_oil[~((df_oil['date'] >= "2016-05-16") & (df_oil['date'] <= "2016-05-31"))]
x_oil = df_oil.iloc[:-12,:]
y_oil = df_oil.iloc[-12:,:]

df_stores = pd.read_csv("store-sales-time-series-forecasting/dataset/stores.csv")

df_transactions = pd.read_csv(
    "store-sales-time-series-forecasting/dataset/transactions.csv",
    parse_dates=['date'],
)
# becaues of earthquake
df_transactions = df_transactions[~((df_transactions['date'] >= "2016-05-16") & (df_transactions['date'] <= "2016-05-31"))]

# get number of stores in train
store_arr = df_train['store_nbr'].unique()
store_nums = len(store_arr)

for store in store_arr:
  x_train_store = df_train[df_train['store_nbr'] == store]
  y_train_store = x_train_store['sales']
  y_train_store['Lag_1'] = y_train_store.shift(1)
#   y_train_store = y_train_store.reindex(columns=['sales', 'Lag_1'])

  x_train_store.drop('sales', axis=1, inplace=True)
#   x_train_store['oil'] = x_oil
  x_train_store = x_train_store.merge(x_oil, on='date', how='left')
  x_train_store['dcoilwtico'] = x_train_store['dcoilwtico'].fillna(method='bfill')
  x_train_store = x_train_store.merge(df_transactions, on=['date', 'store_nbr'], how='left')
  x_train_store['transactions'] = x_train_store['transactions'].fillna(0)
  x_train_store = x_train_store.merge(df_stores, on='store_nbr', how='left')
  # x_train_store = x_train_store.merge(df_holidays, on='date', how='left')
  le = LabelEncoder()
  x_train_store['family'] = le.fit_transform(x_train_store['family'])
  x_train_store['city'] = le.fit_transform(x_train_store['city'])
  x_train_store['state'] = le.fit_transform(x_train_store['state'])
  x_train_store['type'] = le.fit_transform(x_train_store['type'])

  pass