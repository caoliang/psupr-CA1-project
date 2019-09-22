from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# Step 1. Prepare data
from py_src.common_functions import *

class classify_svm:

    @staticmethod
    def main():

        # create model for multi-class and one vs rest mode
        # linear_svc = OneVsRestClassifier(BaggingClassifier(LinearSVC(multi_class='ovr', random_state=1),
        #                                 bootstrap=True, oob_score = True, n_jobs=-1, random_state=1))
        # linear_svc = OneVsRestClassifier(BaggingClassifier(LinearSVC(multi_class='ovr', random_state=1),
        #                                 bootstrap=True, oob_score = True, n_jobs=-1, random_state=1))

        #linear_svc = OneVsRestClassifier(LinearSVC(multi_class='ovr', random_state=1), bootstrap=True, oob_score = True, n_jobs=-1, random_state=1)
        linear_svc = OneVsRestClassifier(LinearSVC(multi_class='ovr', random_state=1))
        print('Created SVM')
        #print("parameters: ", linear_svc.get_params().keys())

        # Create GridSearch to find best model
        # Tested:
        #    max_iter_param = [50, 100, 200, 500, 1000]
        #    dul_param = [True, False]
        #    C_param = [ 1, 4, 6, 10, 20 ]
        #    fit_intercept_param  = [True, False]
        #    n_estimators_param = [ 100, 500, 1000 ]
        #    max_samples_param = [ 100, 500, 1000 ]
        # Best Parameters {'estimator__base_estimator__C': 6,
        #                  'estimator__base_estimator__dual': True,
        #                  'estimator__base_estimator__fit_intercept': False,
        #                  'estimator__base_estimator__max_iter': 100,
        #                  'estimator__max_samples': 1000,
        #                  'estimator__n_estimators': 100}
        # To save time, only run with best parameters
        # max_iter_param = [1000]
        # dul_param = [True]
        # C_param = [ 6 ]
        # fit_intercept_param  = [False]
        # n_estimators_param = [ 100 ]
        # max_samples_param = [ 1000 ]

        # hyperparameters = dict(estimator__base_estimator__C=C_param,
        #                        estimator__base_estimator__fit_intercept=fit_intercept_param,
        #                        estimator__base_estimator__dual=dul_param,
        #                        estimator__base_estimator__max_iter=max_iter_param,
        #                        estimator__n_estimators=n_estimators_param,
        #                        estimator__max_samples=max_samples_param )

        max_iter_param = [10000]
        dul_param = [True]
        C_param = [0.01, 0.1, 1]
        fit_intercept_param = [False]
        loss_param = ['squared_hinge', 'hinge']
        class_weight_param = ['balanced']

        hyperparameters = dict(estimator__C=C_param,
                               estimator__max_iter=max_iter_param,
                               estimator__fit_intercept=fit_intercept_param,
                               estimator__dual=dul_param,
                               estimator__loss=loss_param,
                               estimator__class_weight=class_weight_param
                               )
        # Fit model using gridsearch
        score_making_func = make_scorer(roc_auc_score, average='macro')
        grid_linear_svc = GridSearchCV(linear_svc, hyperparameters, scoring=score_making_func, verbose=0)

        timer_check = time.time()
        print('start training')

        # Best model
        best_linear_svc = grid_linear_svc.fit(X_train, y_train)

        #Print all the Parameters that gave the best results:
        print('Best Parameters', grid_linear_svc.best_params_)

        print('training spent: ', show_time_spent(timer_check),
              ',Accuracy on training set: {:.4f}'.format(grid_linear_svc.score(X_train, y_train)))

        # Compute test scores
        y_pred = best_linear_svc.predict(X_test)
        print('y_pred: ', y_pred.shape)


        # Compute ROC AUC score
        linear_svc_result = compute_roc_auc_in_classes(y_test, y_pred, num_classes=n_classes)

        # Draw ROC plot
        draw_roc_auc_in_classes(linear_svc_result, 'Linear SVC', num_classes=n_classes)

        # Assign Best ROC value
        roc_linear_svc = linear_svc_result['roc_auc']["macro"]
        print('Best ROC score for Linear SVC: {0:0.4f}'.format(roc_linear_svc))

        score_linear_svc = best_linear_svc.score(X_test, y_test)
        print('Model accuracy is', score_linear_svc)

        return best_linear_svc

if __name__ == '__main__':
    classify_svm.main()