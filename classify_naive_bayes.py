import warnings
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

warnings.filterwarnings('ignore')
#%matplotlib inline

# Step 1. Prepare data
from common_functions import *

# Step 2. Solve with Naive Bayes
kernel = 1.0 * RBF(1.0)

# create Naive Bayes model for multi-class and one vs rest mode
gaussian_proc = OneVsRestClassifier(GaussianProcessClassifier(kernel=kernel, multi_class='one_vs_rest', random_state=1))
print('Created gaussian_proc')
#print("parameters: ", gaussian_proc.get_params().keys())
#print(sorted(metrics.SCORERS.keys()))

# Create GridSearch to find best model
# Tested:
#    max_iter_param = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#    warm_state_param = [ True, False ]
# And best parameters: {'estimator__max_iter_predict': 20, 'estimator__warm_start': False}
# So to save time, only run with best parameter
max_iter_param = [20]
warm_state_param = [ False ]
hyperparameters = dict(estimator__max_iter_predict=max_iter_param, estimator__warm_start=warm_state_param)

# Fit model using gridsearch
score_making_func = make_scorer(roc_auc_score, average='macro')
grid_gaussian_proc = GridSearchCV(gaussian_proc, hyperparameters, scoring=score_making_func, verbose=0)
#%%
# Compute test scores
timer_check = time.time()
print('start training')

# Best model
best_gaussian_proc = grid_gaussian_proc.fit(X_train, y_train)

#Print all the Parameters that gave the best results:
print('Best Parameters', grid_gaussian_proc.best_params_)

print('training spent: ', show_time_spent(timer_check))

#timer_check = time.time()
print('start testing')
y_pred_proba = best_gaussian_proc.predict_proba(X_test)
#print('testing spent: ', show_time_spent(timer_check))
print('y_pred_proba: ', y_pred_proba.shape)

# Compute ROC AUC score
gaussion_proc_result = compute_roc_auc_in_classes(y_test, y_pred_proba, num_classes=n_classes)

# Draw ROC plot
draw_roc_auc_in_classes(gaussion_proc_result, 'Naive Bayes', num_classes=n_classes)

# Assign Best ROC value
roc_gaussian_proc = gaussion_proc_result['roc_auc']["macro"]
print('Best ROC score for Naive Bayes: {0:0.4f}'.format(roc_gaussian_proc))