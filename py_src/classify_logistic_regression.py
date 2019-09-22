from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# Step 1. Prepare data
from py_src.common_functions import *

class classify_logistic_regression:

    @staticmethod
    def main():
        # create logistic regression model for multi-class and one vs rest mode
        logreg = OneVsRestClassifier(LogisticRegression(solver='sag', multi_class='ovr', random_state=1, penalty='elasticnet'))
        #logreg = OneVsRestClassifier(LogisticRegression(solver='sag', multi_class='ovr', random_state=1))
        #logreg = OneVsRestClassifier(LogisticRegression())
        # print("parameters: ", logreg.get_params().keys())

        '''
        # Create GridSearch to find best model
        penalty_param = ['l2']
        solver_param = ['sag', 'saga']
        C_param = [0.0001, 0.001, 0.01, 1, 50, 100, 200, 1000]
        hyperparameters = dict(estimator__C=C_param, estimator__penalty=penalty_param, estimator__solver=solver_param)
        
        # Fit model using gridsearch
        score_making_func = make_scorer(roc_auc_score, average='macro')
        grid_logreg = GridSearchCV(logreg, hyperparameters, scoring=score_making_func, verbose=0)
        '''

        # Create GridSearch to find best model
        penalty_param = ['l1', 'l2']
        class_weight_param = ['balanced']
        solver_param = ['liblinear']
        C_param = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 2000, 3000]
        max_iter_param = [1000]
        hyperparameters = dict(estimator__C=C_param, estimator__penalty=penalty_param,
                               estimator__class_weight=class_weight_param,
                               estimator__solver=solver_param, estimator__max_iter=max_iter_param)

        # Fit model using gridsearch
        grid_logreg = GridSearchCV(logreg, hyperparameters, scoring = 'accuracy', cv=5, verbose=True, n_jobs=-1)
        timer_check = time.time()
        print('start training')

        # Best model
        best_logreg = grid_logreg.fit(X_train, y_train)

        # Print all the Parameters that gave the best results:
        print('Best Parameters', grid_logreg.best_params_)
        print('training spent: ', show_time_spent(timer_check))


        timer_check = time.time()
        print('start validating')
        # Compute test scores
        y_score = best_logreg.predict(X_test)
        print_confusion_matix(y_test, y_score)

        # Compute ROC AUC score
        logreg_result = compute_roc_auc_in_classes(y_test, y_score, num_classes=n_classes)

        print('Validation spent: ', show_time_spent(timer_check))

        # Draw ROC plot
        draw_roc_auc_in_classes(logreg_result, 'Logistic Regression', num_classes=n_classes)


        # Assign Best score
        roc_logreg = logreg_result['roc_auc']['macro']
        print("Best ROC score for logistic regression: {0:0.4f}".format(roc_logreg))

        # score_logreg = best_logreg.score(X_test, y_test)
        # print("Model accuracy is {0:0.4f}".format(score_logreg))

        score_logreg = average_precision_score(y_test, y_score, average='weighted')
        print('Model accuracy is: {0:0.4f}'.format(score_logreg))

        show_feature_importance(grid_logreg.best_estimator_, feature_importance_func=get_log_reg_feature_importance)

        return best_logreg

if __name__ == '__main__':
    classify_logistic_regression.main()