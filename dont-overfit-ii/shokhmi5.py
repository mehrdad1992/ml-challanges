import sklearn
print(sklearn.__version__)
import pandas as pd
import numpy as np
import gc
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix, roc_curve
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
# warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Step 1: Load the Data
train = pd.read_csv('dont-overfit-ii/data/train.csv')
test = pd.read_csv('dont-overfit-ii/data/test.csv')

# Step 2: Preprocess the Data
X = train.drop(columns=["id", "target"])
y = train["target"]
X_test = test.drop(columns=["id"])
print(train.shape)
print(test.shape)

# Step 3: Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Step 4: Apply PCA to Retain 90% Variance
pca = PCA(0.90)  # Retain 90% of the variance
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Plotting the first two PCA components
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(label='Target')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('First Two PCA Components Colored by Target')
plt.show()

# Step 3: Cross-Validation and Training Function
# Cross-Validation and Training Function

def logreg_cross_validation(X, y, params, num_folds=5, repeats=20, random_state=3210):

    clfs = []
    folds = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=repeats, random_state=random_state)
    valid_pred = pd.Series(0, index=y.index)

    # Cross-validation loop
    for train_idx, valid_idx in folds.split(X, y):
        # Split data
        train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]

        # Train Logistic Regression
        clf = LogisticRegression(**params)
        clf.fit(train_x, train_y)
        clfs.append(clf)

        # Predict on validation set and accumulate predictions
        valid_pred.loc[valid_idx] += clf.predict_proba(valid_x)[:, 1] / repeats

    return clfs, valid_pred

# Define Logistic Regression Parameters
params = {
    'C': 0.1,
    'penalty': 'l1',
    'class_weight': 'balanced',
    'solver': 'liblinear',
    'random_state': 300
}

# Step 5: Perform Cross-Validation and Train Models
clfs, pred_mean = logreg_cross_validation(pd.DataFrame(X_scaled), y, params)

# Step 6: Calculate Average Predictions and Metrics
scores = {
    'AUC': roc_auc_score(y, pred_mean),
    'Accuracy': accuracy_score(y, (pred_mean >= 0.5).astype(int)),
    'Log Loss': log_loss(y, pred_mean)
}

print(f"AUC: {scores['AUC']:.4f}, Accuracy: {scores['Accuracy']:.4f}, Log Loss: {scores['Log Loss']:.4f}")
subm = pd.DataFrame(0, index=test.index, columns=['target'])

for clf in clfs:
    subm['target'] += clf.predict_proba(X_test_scaled)[:, 1]

subm['target'] /= len(clfs)
subm = subm.reset_index()
subm.columns = ['id', 'target']
subm['id'] = test['id']
subm.to_csv("dont-overfit-ii/data/submission7.csv", index=False)