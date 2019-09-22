import warnings
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
# %matplotlib inline

# Step 1. Prepare data
from py_src.common_functions import *


class classify_decision_tree:

    @staticmethod
    def main():
        # create Decision Tree model for multi-class and one vs rest mode
        # decision_tree = OneVsRestClassifier(BaggingClassifier(DecisionTreeClassifier(random_state=1), bootstrap=True, oob_score = True, n_jobs=-1, random_state=1))
        decision_tree = OneVsRestClassifier(DecisionTreeClassifier(random_state=1))
        # decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=29, max_depth=30)
        print('Created decision_tree')

        # print("parameters: ", decision_tree.get_params().keys())

        # Create GridSearch to find best model
        # Tested:
        #    criterion_param = ['entropy', 'gini']
        #    max_depth_param = [ 10, 20, 40]
        #    max_leaf_nodes_param = [ 20, 50, 100 ]
        #    max_features_param = [ 20, 40, X_train.shape[1]]
        #    n_estimators_param = [ 100, 500, 1000 ]
        #    max_samples_param = [ 100, 500, 1000 ]
        #    Best Parameters {'estimator__base_estimator__criterion': 'gini',
        #                     'estimator__base_estimator__max_depth': 20,
        #                     'estimator__base_estimator__max_features': 56,
        #                     'estimator__base_estimator__max_leaf_nodes': 50,
        #                     'estimator__max_samples': 1000,
        #                     'estimator__n_estimators': 500}
        # So to save time, only run with best parameters

        ######################### PREVIOUS DT PARAMS #########################
        # criterion_param = [ 'gini']
        # max_depth_param = [ 20 ]
        # max_leaf_nodes_param = [ 50 ]
        # max_features_param = [ X_train.shape[1] ]
        # n_estimators_param = [ 500 ]
        # max_samples_param = [ 1000 ]
        #
        # hyperparameters = dict(estimator__base_estimator__criterion=criterion_param,
        #                        estimator__base_estimator__max_depth=max_depth_param,
        #                        estimator__base_estimator__max_leaf_nodes=max_leaf_nodes_param,
        #                        estimator__base_estimator__max_features=max_features_param,
        #                        estimator__n_estimators=n_estimators_param,
        #                        estimator__max_samples=max_samples_param )

        criterion_param = ['gini']
        # criterion_param = [ 'entropy']
        max_depth_param = [10, 20, 30, 40]
        max_leaf_nodes_param = [50]
        max_features_param = [10, 20, 30, X_train.shape[1]]

        hyperparameters = dict(estimator__criterion=criterion_param,
                               estimator__max_depth=max_depth_param,
                               estimator__max_leaf_nodes=max_leaf_nodes_param,
                               estimator__max_features=max_features_param)

        # Fit model using gridsearch
        score_making_func = make_scorer(roc_auc_score, average='macro')
        grid_decision_tree = GridSearchCV(decision_tree, hyperparameters, scoring=score_making_func, verbose=0)
        # grid_decision_tree = GridSearchCV(decision_tree, hyperparameters, scoring = score_making_func, cv=5, verbose=False, n_jobs=-1)
        # grid_decision_tree = decision_tree
        # print(sorted(metrics.SCORERS.keys()))

        # Compute test scores
        timer_check = time.time()
        print('start training')

        # Best model
        best_decision_tree = grid_decision_tree.fit(X_train, y_train)

        print('training spent: ', show_time_spent(timer_check),
              ',Accuracy on training set: {:.4f}'.format(grid_decision_tree.score(X_train, y_train)))

        # print(.format())

        # Print all the Parameters that gave the best results:
        # print('Best Parameters', grid_decision_tree.best_estimator_)

        # Compute test scores
        # y_pred_proba = best_decision_tree.predict_proba(X_test)
        # print('y_pred_proba: ', y_pred_proba.shape)

        # Compute test scores
        y_score = best_decision_tree.predict(X_test)
        print('y_score: ', y_score.shape)

        # Print confusion scores
        print_confusion_matix(y_test, y_score)

        print('training spent: ', show_time_spent(timer_check))

        ''' OLD PARAMS
        # Compute ROC AUC score
        decision_tree_result = compute_roc_auc_in_classes(y_test, y_pred_proba, num_classes=n_classes)
        
        # Draw ROC plot
        draw_roc_auc_in_classes(decision_tree_result, 'Decision Tree', num_classes=n_classes)
        
        # Assign Best ROC value
        roc_decision_tree = decision_tree_result['roc_auc']["macro"]
        print('Best ROC score for Decision Tree: {0:0.4f}'.format(roc_decision_tree))
        '''

        # Compute ROC AUC score
        decision_tree_result = compute_roc_auc_in_classes(y_test, y_score, num_classes=n_classes)

        # Draw ROC plot
        draw_roc_auc_in_classes(decision_tree_result, 'Decision Tree', num_classes=n_classes)

        # Assign Best ROC value
        roc_decision_tree = decision_tree_result['roc_auc']["macro"]
        print('Best ROC score for Decision Tree: {0:0.4f}'.format(roc_decision_tree))

        score_decision_tree = best_decision_tree.score(X_test, y_test)
        print('Model accuracy is: {0:0.4f}'.format(score_decision_tree))

        return best_decision_tree

if __name__ == '__main__':
    classify_decision_tree.main()
