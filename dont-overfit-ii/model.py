import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report



train = pd.read_csv('dont-overfit-ii/data/train.csv', index_col='id')
# test = pd.read_csv('dont-overfit-ii/data/test.csv', index_col='id')
# sample_submission = pd.read_csv('dont-overfit-ii/data/sample_submission.csv')

# X_test = test.drop(['id'], axis=1)


X = train.drop(['target'], axis=1)
y = train['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
