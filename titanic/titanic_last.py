# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('data/train.csv', index_col='PassengerId')
x_test = pd.read_csv('data/test.csv', index_col='PassengerId')

print(train.describe())
print(train.shape)
print(train.head())

nan_column = train.columns[train.isna().sum() > len(train)/2]
print(nan_column)

y_train = train['Survived']
x_train = train.drop('Survived' , axis=1)

# clean unrelated data, 'Name'
x_train.drop('Name', axis=1, inplace=True)
x_test.drop('Name', axis=1, inplace=True)

print(y_train.shape)
print(y_train.head())
print(x_train.shape)
print(x_train['Ticket'].head(100))

x_train['Cabin'] = x_train['Cabin'].fillna('Unknown')
x_train['Embarked'] = x_train['Embarked'].fillna('U')
mean_age = x_train['Age'].mean()
x_train['Age'] = x_train['Age'].fillna(mean_age)
x_train['Ticket'] = x_train['Ticket'].fillna('110287')
x_train['Sex'] = x_train['Sex'].map({'male': 0, 'female': 1})

print(x_train.head(10))

# Do the same for test
# print(test.isna().sum())
x_test['Cabin'] = x_test['Cabin'].fillna('Unknown')
x_test['Embarked'] = x_test['Embarked'].fillna('U')
mean_age_test = x_test['Age'].mean()
x_test['Age'] = x_test['Age'].fillna(mean_age)
x_test['Fare'] = x_test['Fare'].bfill()
x_test['Sex'] = x_test['Sex'].map({'male': 0, 'female': 1})

print(x_test.head(10))

def split_ticket(ticket):
    parts = ticket.split(' ')
    if len(parts) == 1:
        return pd.Series(['None', parts[0]])
    else:
        return pd.Series([' '.join(parts[:-1]), parts[-1]])

le_ticket = LabelEncoder()
le_cabin = LabelEncoder()
le_embarked = LabelEncoder()

x_train[['TicketPrefix', 'TicketNumber']] = x_train['Ticket'].apply(split_ticket)
# Step 2: Convert TicketNumber to numeric (optional, helps models)
x_train['TicketNumber'] = pd.to_numeric(x_train['TicketNumber'], errors='coerce')
x_train['TicketPrefixEncoded'] = le_ticket.fit_transform(x_train['TicketPrefix'])
x_train['CabinEncoded'] = le_cabin.fit_transform(x_train['Cabin'])
x_train['EmbarkedEncoded'] = le_embarked.fit_transform(x_train['Embarked'])

x_train.drop(['TicketPrefix'], axis=1, inplace=True)
x_train.drop(['Ticket'], axis=1, inplace=True)
x_train.drop(['Cabin'], axis=1, inplace=True)
x_train.drop(['Embarked'], axis=1, inplace=True)
print(x_test.head(10))

x_test[['TicketPrefix', 'TicketNumber']] = x_test['Ticket'].apply(split_ticket)
x_test['TicketNumber'] = pd.to_numeric(x_test['TicketNumber'], errors='coerce')
x_test['TicketPrefixEncoded'] = le_ticket.fit_transform(x_test['TicketPrefix'])
x_test['CabinEncoded'] = le_cabin.fit_transform(x_test['Cabin'])
x_test['EmbarkedEncoded'] = le_embarked.fit_transform(x_test['Embarked'])
x_test.drop(['TicketPrefix'], axis=1, inplace=True)
x_test.drop(['Ticket'], axis=1, inplace=True)
x_test.drop(['Cabin'], axis=1, inplace=True)
x_test.drop(['Embarked'], axis=1, inplace=True)

print(x_test.head(10))

model = XGBClassifier(use_label_encoder=False)
model.fit(x_train, y_train)


y_test = model.predict(x_test)

print("y_train shape: ", y_train.shape)
print(x_test.index)
print(y_test)

file = pd.DataFrame({"PassengerId":x_test.index, "Survived":y_test})
file.to_csv("data/gender_submission.csv", index=False)