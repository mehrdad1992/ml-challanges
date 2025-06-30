import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


train = pd.read_csv('dont-overfit-ii/data/train.csv', index_col='id')
X_test = pd.read_csv('dont-overfit-ii/data/test.csv', index_col='id')
# sample_submission = pd.read_csv('dont-overfit-ii/data/sample_submission.csv')

# X_test = test.drop(['id'], axis=1)


X = train.drop(['target'], axis=1)
y = train['target']

X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model_eval = LogisticRegression(max_iter=1000)
model_eval = LogisticRegression(max_iter=1000,penalty='l1', solver='liblinear', C=0.1)
model_eval.fit(X_train, y_train)

y_pred_proba = model_eval.predict_proba(X_eval)[:, 1]
y_pred = model_eval.predict(X_eval)
y_eval_test = model_eval.predict(X_test).astype(int)

print("Accuracy:", accuracy_score(y_eval, y_pred))
print("ROC AUC:", roc_auc_score(y_eval, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_eval, y_pred))

# Search the best parameters for the model
pipeline = Pipeline([
    ('select', SelectKBest(score_func=f_classif)),
    ('model', LogisticRegression(solver='saga', max_iter=1000))
])

param_grid = {
    'select__k': [10, 20, 50],
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2'],
    # 'model__max_iter': [10, 100, 1000]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X, y)
y_test = grid.predict(X_test).astype(int)

print("Best CV score:", grid.best_score_)
print("Best params:", grid.best_params_)

# Use the model on real train and test
# model_full = LogisticRegression(max_iter=1000) # score=0.51, expected 0.76 so we have overfitting
# Add regularization to the model to improve it and prevent overfitting
# model_full = LogisticRegression(max_iter=1000,penalty='l1', solver='liblinear', C=0.1) # score was 0.52 so we have overfitting
# model_full.fit(X, y) # x is x_train and y is y_train here
# y_test = model_full.predict(X_test).astype(int)

file = pd.DataFrame({"id":X_test.index, "target":y_test})
file.to_csv("dont-overfit-ii/data/sample_submission.csv", index=False)
