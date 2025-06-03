import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def main():
    df_train = pd.read_csv(
        "store-sales-time-series-forecasting/dataset/train.csv",
        index_col='id',
        parse_dates=['date'],
    )
    df_train['time'] = np.repeat(np.arange(1684),1782)

    # becaues of earthquake
    df_train = df_train[~((df_train['date'] >= "2016-05-16") & (df_train['date'] <= "2016-05-31"))]

    # y_train = df_train['sales']
    # x_train = df_train.drop('sales', axis=1)

    df_test = pd.read_csv(
        "store-sales-time-series-forecasting/dataset/test.csv",
        index_col='id',
        parse_dates=['date'],
    )
    df_test['time'] = np.repeat(np.arange(16),1782)

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
    test_oil = df_oil.iloc[-12:,:]

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

    all_results = []
    # for each store these will be done
    for store in store_arr:
        # fetch x_train and y_train from df_train
        x_train_store = df_train[df_train['store_nbr'] == store]
        y_train_store = x_train_store['sales']

        #fetch x_test_store from df_test
        x_test_store = df_test[df_test['store_nbr'] == store]


        # build x_train_store
        x_train_store['lag_1782'] = y_train_store.shift(1782)
        x_train_store['lag_1782'][0:1782] = x_train_store['lag_1782'][1782:1782*2] 
        x_train_store.drop('sales', axis=1, inplace=True)
        #   x_train_store['oil'] = x_oil
        x_train_store = x_train_store.merge(x_oil, on='date', how='left')
        x_train_store['dcoilwtico'] = x_train_store['dcoilwtico'].fillna(method='bfill')
        x_train_store = x_train_store.merge(df_transactions, on=['date', 'store_nbr'], how='left')
        x_train_store['transactions'] = x_train_store['transactions'].fillna(0)
        x_train_store = x_train_store.merge(df_stores, on='store_nbr', how='left')
        
        le = LabelEncoder()
        x_train_store['family'] = le.fit_transform(x_train_store['family'])
        x_train_store['city'] = le.fit_transform(x_train_store['city'])
        x_train_store['state'] = le.fit_transform(x_train_store['state'])
        x_train_store['type'] = le.fit_transform(x_train_store['type'])
        # x_train_store = x_train_store.merge(df_holidays, on='date', how='left')

        # build x_test_store
        # x_train_store['lag_1782'] = y_train_store.shift(1782)
        # x_train_store['lag_1782'][0:1782] = x_train_store['lag_1782'][1782:1782*2] 
        x_test_store = x_test_store.merge(test_oil, on='date', how='left')
        x_test_store['dcoilwtico'] = x_test_store['dcoilwtico'].fillna(method='bfill')
        x_test_store = x_test_store.merge(df_transactions, on=['date', 'store_nbr'], how='left')
        x_test_store['transactions'] = x_test_store['transactions'].fillna(0)
        x_test_store = x_test_store.merge(df_stores, on='store_nbr', how='left')

        le = LabelEncoder()
        x_test_store['family'] = le.fit_transform(x_test_store['family'])
        x_test_store['city'] = le.fit_transform(x_test_store['city'])
        x_test_store['state'] = le.fit_transform(x_test_store['state'])
        x_test_store['type'] = le.fit_transform(x_test_store['type'])
        
        # reorder columns to prevent predicttion method error
        x_train_store = x_train_store.loc[:, x_test_store.columns.to_list() + ['lag_1782']]
        
        ## drawing correlation plot of lag feature 
        # draw_plot(x_train_store['lag_1782'], y_train_store)

        x_train_store.drop('date', axis=1, inplace=True)

        # normalize x_train_store
        columns_to_normalize = [col for col in x_train_store.columns if col != 'time']
        # Fit-transform and rewrap in a DataFrame
        scalar = MinMaxScaler(feature_range=(0,1))
        normalized = scalar.fit_transform(x_train_store[columns_to_normalize])
        x_train_store[columns_to_normalize] = pd.DataFrame(normalized, columns=columns_to_normalize, index=x_train_store.index)


        model = LinearRegression(positive=True)
        model.fit(x_train_store, y_train_store)

        # get number of date in x_train
        x_test_store.drop('date', axis=1, inplace=True)
        time_arr = x_test_store['time'].unique()

        # normalize x_test_store
        columns_to_normalize = [col for col in x_test_store.columns if col != 'time']
        # Fit-transform and rewrap in a DataFrame
        normalized = scalar.fit_transform(x_test_store[columns_to_normalize])
        x_test_store[columns_to_normalize] = pd.DataFrame(normalized, columns=columns_to_normalize, index=x_test_store.index)

        date_nums = len(time_arr)
        lag_feature = np.zeros(33)
        s_results = []
        for time in time_arr:
            x_train_day = x_train_store[x_train_store['time'] == time]
            x_test_day = x_test_store[x_test_store['time'] == time]
            x_test_day['lag_1782'] = lag_feature
            result = model.predict(x_test_day)
            lag_feature = result
            s_results.append(result)
        
        all_results.append(np.array(s_results))
    
    all_results = np.array(all_results).flatten()
    submission = pd.DataFrame({
        'id': df_test.index,
        'sales': all_results
    })
    submission.to_csv('store-sales-time-series-forecasting/submission.csv', index=False)


def draw_plot(a, b):
    plt.plot(a, b, 'o')
    plt.xlabel('X Vector')
    plt.ylabel('Y Vector')
    plt.title('Line Plot of Two Vectors')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

