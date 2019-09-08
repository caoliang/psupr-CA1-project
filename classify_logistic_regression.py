from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import warnings

warnings.filterwarnings('ignore')
#%matplotlib inline

# Step 1. Prepare data
from common_functions import *

# create logistic regression model for multi-class and one vs rest mode
logreg = OneVsRestClassifier(LogisticRegression(solver='sag', multi_class='ovr', random_state=1))
# print("parameters: ", logreg.get_params().keys())

# Create GridSearch to find best model
penalty_param = ['l2']
solver_param = ['sag', 'saga']
C_param = [0.0001, 0.001, 0.01, 1, 50, 100, 200, 1000]
hyperparameters = dict(estimator__C=C_param, estimator__penalty=penalty_param, estimator__solver=solver_param)

# Fit model using gridsearch
score_making_func = make_scorer(roc_auc_score, average='macro')
grid_logreg = GridSearchCV(logreg, hyperparameters, scoring=score_making_func, verbose=0)

# Best model
best_logreg = grid_logreg.fit(X_train, y_train)

# Print all the Parameters that gave the best results:
print('Best Parameters', grid_logreg.best_params_)

# Compute test scores
y_score = best_logreg.fit(X_train, y_train).decision_function(X_test)

# Compute ROC AUC score
logreg_result = compute_roc_auc_in_classes(y_test, y_score, num_classes=n_classes)

# Draw ROC plot
draw_roc_auc_in_classes(logreg_result, 'Logistic Regression', num_classes=n_classes)


# Assign Best score
roc_logreg = logreg_result['roc_auc']['macro']
print("Best ROC score for logistic regression: {0:0.4f}".format(roc_logreg))
