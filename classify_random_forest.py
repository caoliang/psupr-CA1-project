
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# Step 1. Prepare data
from common_functions import *

# create model for multi-class and one vs rest mode
rnd_forest = OneVsRestClassifier(RandomForestClassifier(warm_start=False, oob_score=True, n_jobs=-1, random_state=1))

print('Created Random Forest')
# print("parameters: ", rnd_forest.get_params().keys())

# Create GridSearch to find best model
# Tested:
#      n_estimators_param = [ 50, 100, 200]
#      max_depth_param = [10, 20, 40]
#      max_leaf_nodes_param = [ 20, 50, 100 ]
#      max_features_param = [ 20, X_train.shape[1] ]
#      criterion_param = ['entropy', 'gini' ]
#      class_weight_param = ['balanced', 'balanced_subsample', None]
# To save time, only test partial parameters

n_estimators_param = [100, 200, 500]
max_depth_param = [10, 20, 40]
max_leaf_nodes_param = [20, 50, 100]
max_features_param = [20, X_train.shape[1]]
criterion_param = ['gini']
class_weight_param = ['balanced']

hyperparameters = dict(estimator__n_estimators=n_estimators_param,
                       estimator__max_depth=max_depth_param,
                       estimator__max_leaf_nodes=max_leaf_nodes_param,
                       estimator__max_features=max_features_param,
                       estimator__criterion=criterion_param,
                       estimator__class_weight=class_weight_param)

# Fit model using gridsearch
score_making_func = make_scorer(roc_auc_score, average='macro')
grid_rnd_forest = GridSearchCV(rnd_forest, hyperparameters, scoring=score_making_func, verbose=0)


timer_check = time.time()
print('start training')

# Best model
best_rnd_forest = grid_rnd_forest.fit(X_train, y_train)

#Print all the Parameters that gave the best results:
print('Best Parameters', grid_rnd_forest.best_params_)

print('training spent: ', show_time_spent(timer_check))

# Compute test scores
y_pred_proba = best_rnd_forest.predict(X_test)
print('y_pred_proba: ', y_pred_proba.shape)

# Compute ROC AUC score
rnd_forest_result = compute_roc_auc_in_classes(y_test, y_pred_proba, num_classes=n_classes)

# Draw ROC plot
draw_roc_auc_in_classes(rnd_forest_result, 'Random Forest', num_classes=n_classes)

# Assign Best ROC value
roc_rnd_forest = rnd_forest_result['roc_auc']["macro"]
print('Best ROC score for Random Forest: {0:0.4f}'.format(roc_rnd_forest))
