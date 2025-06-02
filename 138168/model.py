import pandas as pd
import ast
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error


train = pd.read_csv('138168/data/train.csv')
test = pd.read_csv('138168/data/test.csv')


train['product_description'] = train['product_description'].apply(ast.literal_eval)
test['product_description'] = test['product_description'].apply(ast.literal_eval)

desc_train = pd.json_normalize(train['product_description'])
desc_test = pd.json_normalize(test['product_description'])

# سپس آن‌ها را به دیتافریم اصلی اضافه کنید
train = pd.concat([train[['id', 'price']], desc_train], axis=1)
test = pd.concat([test[['id']], desc_test], axis=1)


encoder = OneHotEncoder(handle_unknown='ignore')
encoded_train = encoder.fit_transform(train[['برند', 'دسته بندی']])
encoded_test = encoder.transform(test[['برند', 'دسته بندی']])

X_train = encoded_train
y_train = train['price']
X_test = encoded_test

model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

output = pd.DataFrame({'id': test['id'], 'price': y_pred})
output.to_csv('138168/output.csv', index=False)



# mape = mean_absolute_percentage_error(y_true, y_pred) * 100
# score = 100 - mape
# print("Score:", score)