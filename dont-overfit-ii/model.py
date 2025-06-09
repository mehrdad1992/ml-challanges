import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

train = pd.read_csv('dont-overfit-ii/data/train.csv', index_col='id')
X_test = pd.read_csv('dont-overfit-ii/data/test.csv', index_col='id')
# sample_submission = pd.read_csv('dont-overfit-ii/data/sample_submission.csv')

# X_test = test.drop(['id'], axis=1)


X = train.drop(['target'], axis=1)
y = train['target']

X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_eval = LogisticRegression(max_iter=1000)
model_eval.fit(X_train, y_train)

y_pred_proba = model_eval.predict_proba(X_eval)[:, 1]
y_pred = model_eval.predict(X_eval)
y_eval_test = model_eval.predict(X_test).astype(int)

print("Accuracy:", accuracy_score(y_eval, y_pred))
print("ROC AUC:", roc_auc_score(y_eval, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_eval, y_pred))

# Use the model on real train and test
model_full = LogisticRegression(max_iter=1000)
model_full.fit(X, y) # x is x_train and y is y_train here
y_test = model_full.predict(X_test).astype(int)

file = pd.DataFrame({"id":X_test.index, "target":y_test})
file.to_csv("dont-overfit-ii/data/sample_submission.csv", index=False)
