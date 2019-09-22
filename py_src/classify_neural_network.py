from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# Step 1. Prepare data
from py_src.common_functions import *

class classify_neural_network:

    @staticmethod
    def main():

        # create model for multi-class and one vs rest mode
        mlp_nn = OneVsRestClassifier(MLPClassifier(early_stopping=True, random_state=1))
        print('Created MLPClassifier')
        #print("parameters: ", mlp_nn.get_params().keys())

        # Create GridSearch to find best model
        # Tested:
        #   hidden_layer_sizes_param = [(2,),(3,),(4,),(5,),(10,),(2,1),(2,2),(3,3),(4,4),(5,5),(10,10)]
        #   max_iter_param = [10, 50, 60, 70, 80, 100, 500]
        #   activation_param = ['relu','tanh']
        #   learning_rate_param = ['constant', 'invscaling', 'adaptive']
        #   solver_param = ['lbfgs', 'sgd', 'adam']
        # Best Parameters {'estimator__activation': 'tanh',
        #                  'estimator__hidden_layer_sizes': (3,),
        #                  'estimator__learning_rate': 'constant',
        #                  'estimator__max_iter': 60,
        #                  'estimator__solver': 'adam'}
        # So to save time, run with best parameters
        hidden_layer_sizes_param = [(3,)]
        max_iter_param = [ 60 ]
        activation_param = ['tanh']
        learning_rate_param = ['constant']
        solver_param = ['adam']

        hyperparameters = dict(estimator__hidden_layer_sizes=hidden_layer_sizes_param,
                               estimator__max_iter=max_iter_param,
                               estimator__activation=activation_param,
                               estimator__learning_rate=learning_rate_param,
                               estimator__solver=solver_param)

        # Fit model using gridsearch
        score_making_func = make_scorer(roc_auc_score, average='macro')
        grid_mlp_nn = GridSearchCV(mlp_nn, hyperparameters, scoring=score_making_func, verbose=0)

        timer_check = time.time()
        print('start training')

        # Best model
        best_mlp_nn = grid_mlp_nn.fit(X_train, y_train)

        #Print all the Parameters that gave the best results:
        print('Best Parameters', grid_mlp_nn.best_params_)

        print('training spent: ', show_time_spent(timer_check))

        # Compute test scores
        y_pred_proba = best_mlp_nn.predict_proba(X_test)
        print('y_pred_proba: ', y_pred_proba.shape)

        # Compute ROC AUC score
        mlp_nn_result = compute_roc_auc_in_classes(y_test, y_pred_proba, num_classes=n_classes)

        # Draw ROC plot
        draw_roc_auc_in_classes(mlp_nn_result, 'Neural Network', num_classes=n_classes)

        # Assign Best ROC value
        roc_mlp_nn = mlp_nn_result['roc_auc']["macro"]
        print('Best ROC score for Neural Network: {0:0.4f}'.format(roc_mlp_nn))

        score_logreg = grid_mlp_nn.score(X_test, y_test)
        print('Model accuracy is: {0:0.4f}'.format(score_logreg))

        return grid_mlp_nn

if __name__ == '__main__':
    classify_neural_network.main()