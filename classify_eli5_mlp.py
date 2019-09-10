from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, roc_curve, auc
from eli5.sklearn import PermutationImportance
from itertools import cycle, product
import seaborn as sns
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# Step 1. Prepare data
from common_functions import *

# %%
# Set up space dictionary with specified hyperparameters
space = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', range(1, 56)),
         'activation': hp.choice('activation', ['logistic', 'tanh', 'relu', 'identity']),
         'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
         'alpha': hp.uniform('alpha', 0.0001, 1000),
         'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
         'learning_rate_init': hp.uniform('learning_rate_init', 0.001, 0.99),
         'power_t': hp.uniform('power_t', 0.01, 0.99)}

target_clf = [None] * 10
target_clf_fitted = [None] * 10
target_y_pred_proba = [None] * 10
target_y_pred = [None] * 10


# Set up objective function
def objective(params):
    params = {'hidden_layer_sizes': params['hidden_layer_sizes'],
              'activation': params['activation'],
              'solver': params['solver'],
              'alpha': params['alpha'],
              'learning_rate': params['learning_rate'],
              'learning_rate_init': params['learning_rate_init'],
              'power_t': params['power_t']}

    hyperopt_clf = MLPClassifier(max_iter=4000,
                                 random_state=42,
                                 warm_start=True,
                                 early_stopping=True,
                                 **params)

    best_score = cross_val_score(estimator=hyperopt_clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=3,
                                 n_jobs=4,
                                 pre_dispatch=10).mean()

    loss = 1 - best_score
    return {'best_score': best_score, 'loss': loss, 'status': STATUS_OK, 'eval_time': time.time()}


# Run the algorithm
trials = Trials()
best = fmin(fn=objective,
            space=space,
            max_evals=1000,
            rstate=np.random.RandomState(42),
            algo=tpe.suggest,
            trials=trials)

best_score = [None] * len(trials.trials)
best_score[0] = trials.results[0].get('best_score')

searched_params_df = pd.DataFrame(trials.trials[0].get('misc').get('vals').values())
searched_params_df = searched_params_df.transpose()

for i in list(range(1, len(trials.trials))):
    new_df = pd.DataFrame(trials.trials[i].get('misc').get('vals').values())
    searched_params_df = searched_params_df.append(new_df.transpose())
    best_score[i] = trials.results[i].get('best_score')

searched_params_df = searched_params_df.rename(columns={0: 'activation', 1: 'alpha', 2: 'hidden_layer_sizes',
                                                        3: 'learning_rate', 4: 'learning_rate_init',
                                                        5: 'power_t', 6: 'solver'})

blankIndex = [''] * len(searched_params_df)
searched_params_df.index = blankIndex

i = ['activation', 'learning_rate', 'solver']
j = ['logistic', 'constant', 'lbfgs']
k = ['tanh', 'invscaling', 'sgd']
l = ['relu', 'adaptive', 'adam']
m = ['identity']

for (i, j, k, l) in zip(i, j, k, l):
    searched_params_df[i] = searched_params_df[i].replace({0: j, 1: k, 2: l, 3: m})

searched_params_df['best_score'] = best_score

searched_params_df_sorted = searched_params_df.sort_values(by='best_score',
                                                           axis=0,
                                                           ascending=False)

searched_params_df_sorted = searched_params_df_sorted[searched_params_df_sorted.hidden_layer_sizes != 0]

print(searched_params_df_sorted.head(5).transpose())

# Learn to predict each class against the other

for i in range(5):
    target_clf[i] = MLPClassifier(max_iter=4000,
                                  random_state=42,
                                  warm_start=True,
                                  early_stopping=True,
                                  activation=searched_params_df_sorted.values[i][0],
                                  alpha=searched_params_df_sorted.values[i][1],
                                  hidden_layer_sizes=int(searched_params_df_sorted.values[i][2]),
                                  learning_rate=searched_params_df_sorted.values[i][3],
                                  learning_rate_init=searched_params_df_sorted.values[i][4],
                                  power_t=searched_params_df_sorted.values[i][5],
                                  solver=searched_params_df_sorted.values[i][6])

for i in range(5):
    target_clf_fitted[i] = target_clf[i].fit(X_train, y_train)
    target_y_pred_proba[i] = target_clf[i].predict_proba(X_test)
    target_y_pred[i] = target_clf[i].predict(X_test)

# Create the parameter grid
param_grid = {'hidden_layer_sizes': list(product(range(1, 56), range(1, 56)))}

# Create a random search object
ran_clf = RandomizedSearchCV(estimator=MLPClassifier(max_iter=4000,
                                                     random_state=42,
                                                     warm_start=True,
                                                     early_stopping=True,
                                                     activation=searched_params_df_sorted.values[0][0],
                                                     alpha=searched_params_df_sorted.values[0][1],
                                                     learning_rate=searched_params_df_sorted.values[0][3],
                                                     learning_rate_init=searched_params_df_sorted.values[0][4],
                                                     power_t=searched_params_df_sorted.values[0][5],
                                                     solver=searched_params_df_sorted.values[0][6]),
                             param_distributions=param_grid,
                             n_iter=465,
                             n_jobs=4,
                             pre_dispatch=100,
                             cv=3,
                             random_state=42)

ran_clf_fitted = ran_clf.fit(X_train, y_train)

# Configuration of Layer 1 & 2
mean_test_score = list(ran_clf_fitted.cv_results_.get('mean_test_score'))
hidden_layers = list(ran_clf_fitted.cv_results_.get('param_hidden_layer_sizes'))

best_hidden_layers_df = pd.DataFrame({'Hidden Layers': hidden_layers, 'Mean Test Score': mean_test_score})

blankIndex = [''] * len(best_hidden_layers_df)
best_hidden_layers_df.index = blankIndex

# Print out the best configuration of hidden layers
best_hidden_layers_df_sorted = best_hidden_layers_df.sort_values(by='Mean Test Score',
                                                                 axis=0,
                                                                 ascending=False)

print(best_hidden_layers_df_sorted.head(5).transpose())

for i in range(5, 10):
    target_clf[i] = MLPClassifier(max_iter=4000,
                                  random_state=42,
                                  warm_start=True,
                                  early_stopping=True,
                                  activation=searched_params_df_sorted.values[0][0],
                                  alpha=searched_params_df_sorted.values[0][1],
                                  hidden_layer_sizes=best_hidden_layers_df_sorted['Hidden Layers'].values[i - 6],
                                  learning_rate=searched_params_df_sorted.values[0][3],
                                  learning_rate_init=searched_params_df_sorted.values[0][4],
                                  power_t=searched_params_df_sorted.values[0][5],
                                  solver=searched_params_df_sorted.values[0][6])

for i in range(5, 10):
    target_clf_fitted[i] = target_clf[i].fit(X_train, y_train)
    target_y_pred_proba[i] = target_clf[i].predict_proba(X_test)
    target_y_pred[i] = target_clf[i].predict(X_test)

sns.set(rc={'figure.figsize': (35, 40), 'legend.fontsize': 25, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
            'lines.markersize': 10, 'axes.labelsize': 0, 'axes.titlesize': 20})

fig, ax = plt.subplots(7, 1)

j = ['Activation Function for the Hidden Layer', 'L2 Penalty', 'Number of Neurons',
     'Learning Rate Schedule for Weight Updates', 'Initial Learning Rate', 'Exponent for Inverse Scaling Learning Rate',
     'Solver for Weight Optimization']
k = ['activation', 'alpha', 'hidden_layer_sizes', 'learning_rate', 'learning_rate_init', 'power_t', 'solver']

for (i, j, k) in zip(range(7), j, k):
    ax[i].set_title('Searched Parameters of the {}'.format(j))
    sns.scatterplot(x=list(range(0, len(trials.trials))),
                    y=searched_params_df[k].values,
                    ax=ax[i])

# Print classification report
i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
j = ['1stBest_1Layer', '2ndBest_1Layer', '3rdBest_1Layer', '4thBest_1Layer', '5thBest_1Layer',
     '1stBest_2Layers', '2ndBest_2Layers', '3rdBest_2Layers', '4thBest_2Layers', '5thBest_2Layers']

for (i, j) in zip(i, j):
    print('{}----------'.format(j))
    print()
    print(multilabel_confusion_matrix(y_true=y_test,
                                      y_pred=target_y_pred[i]))
    print(classification_report(y_true=y_test,
                                y_pred=target_y_pred[i],
                                digits=2))
# %%
# Computing feature importance
print('Best MLP estimator: {}'.format(target_clf[0]))
print()
print('Best results')
print(multilabel_confusion_matrix(y_true=y_test,
                                  y_pred=target_y_pred[0]))
print(classification_report(y_true=y_test,
                            y_pred=target_y_pred[0],
                            digits=2))

perm = PermutationImportance(estimator=target_clf[0],
                             n_iter=100,
                             random_state=42).fit(X_test, y_test)

# Create a dataframe of the variables and feature importances
feature_importances_df = pd.DataFrame({'Variable': X.columns, 'Feature_Importances': perm.feature_importances_})

# Print out the top 3 positive variables
feature_importances_df_sorted = feature_importances_df.sort_values(by='Feature_Importances',
                                                                   axis=0,
                                                                   ascending=False)
print()
print(feature_importances_df_sorted)