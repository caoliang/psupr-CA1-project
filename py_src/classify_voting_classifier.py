# Store best results for different model
from sklearn.svm.libsvm import predict

from py_src.classify_decision_tree import classify_decision_tree
from py_src.classify_logistic_regression import classify_logistic_regression
from py_src.classify_neural_network import classify_neural_network
from py_src.classify_random_forest import classify_random_forest
from py_src.common_functions import *
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np

# Since logistic and RF got better results compare other models, so we do voting on top of these 2.
roc_logreg = classify_logistic_regression.main()
best_rnd_forest = classify_random_forest.main()
#best_decision_tree = classify_decision_tree.main()
#best_mlp_nn = classify_neural_network.main()

avail_models = [
    # {'name': 'Logistic Regression', 'roc_auc': roc_logreg, 'model': best_logreg, 'score': score_logreg},
    # {'name': 'Decision Tree', 'roc_auc': roc_decision_tree, 'model': best_decision_tree, 'score': score_decision_tree},
    # {'name': 'Linear SVC', 'roc_auc': roc_linear_svc, 'model': best_linear_svc, 'score': score_linear_svc},
    # {'name': 'Neural Network', 'roc_auc': roc_mlp_nn, 'model': best_mlp_nn, 'score': score_mlp_nn},
    # {'name': 'Random Forest', 'roc_auc': roc_rnd_forest, 'model': best_rnd_forest, 'score': score_rnd_forest},
    # {'name': 'XGBoost', 'roc_auc': roc_xgboost, 'model': best_xgb_clf, 'score': score_xgb_clf}
]

# Sort based on ROC area values
avail_models.sort(key = lambda sortKey : sortKey['score'], reverse=True)
models_list = [(model['name'], model['score']) for model in avail_models]
print(*models_list, sep='\n')

# We will compare and choose the best 2 models to create voting classifier to estimate the results,
voting_estimators = [roc_logreg, best_rnd_forest]
sel_estimators = [
    ("Logistic Regression", roc_logreg),
    ("Random Forest", best_rnd_forest)
]

# for i in (0, 1):
#     voting_estimators.append(avail_models[i]['model'])
#     sel_estimators.append(("Logistic Regression", roc_logreg))
#     sel_estimators.append(("Decision Tree", best_decision_tree))
#     sel_estimators.append(("Neural Network", best_mlp_nn))
#     sel_estimators.append(("Random Forest", best_rnd_forest))


class HybridClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """

    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights
        # Validation index
        self.v_index = -1

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for clf in self.clfs:
            clf.fit(X, y)

    def set_max_value(self, avg_arr):
        max_item_index = np.argmax(avg_arr)
        for index, value in enumerate(avg_arr):
            if index == max_item_index:
                avg_arr[index] = 1
            else:
                avg_arr[index] = 0

        # print(avg_arr)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """

        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        # print("self.classes_: ", self.classes_.shape)
        # print("self.classes_[:,c]: ", self.classes_[:,0])
        # print("np.argmax(np.bincount(self.classes_[:,c])): ", np.argmax(np.bincount(self.classes_[:,0])))

        if self.weights:
            avg = self.predict_proba(X)
            maj = avg

            for row in range(maj.shape[0]):
                self.set_max_value(maj[row])

            print('maj shape: ', maj.shape)
            # maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
        else:
            pred_y_arr = None

            for clf in self.clfs:
                if pred_y_arr is None:
                    pred_y_arr = np.array(clf.predict(X))
                else:
                    pred_y_arr += np.array(clf.predict(X))

            maj = pred_y_arr

            for row in range(maj.shape[0]):
                self.set_max_value(maj[row])

            print('maj shape (no weight): ', maj.shape)
            # maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])
        return maj

    def decision_function(self, X):
        return predict(X)

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]

        probas_arr = np.array(self.probas_)
        print('probas_arr: ', probas_arr.shape)

        avg = np.average(self.probas_, axis=0, weights=self.weights)
        print('avg: ', avg.shape)

        if self.v_index >= 0:
            avg = avg[:self.v_index]

        return avg

    def set_v_index(self, index=-1):
        self.v_index = index

    def score(self, X, Y):
        score_list = [clf.score(X, Y) for clf in self.clfs]
        avg_score = np.mean(score_list)

        return avg_score

# Create customized voting classifier
#voting_model = HybridClassifier(clfs=voting_estimators)
weights_list = [1 for i in voting_estimators]

voting_model = HybridClassifier(clfs=voting_estimators, weights=weights_list)

# Compute test scores
y_score = voting_model.predict(X_test)
print('y_score: ', y_score.shape)

# Print confusion scores
print_confusion_matix(y_test, y_score)

# Compute ROC AUC score
vt_models_result = compute_roc_auc_in_classes(y_test, y_score, num_classes=n_classes)

# Draw ROC plot
draw_roc_auc_in_classes(vt_models_result, 'Voting (soft)', num_classes=n_classes)

# Show final ROC value which is better than 3 best models
roc_vt_models = vt_models_result['roc_auc']["macro"]
print('ROC score for Voting Classifiers: {0:0.4f}'.format(roc_vt_models))

score_vt_models = voting_model.score(X_test, y_test)
print('Accuracy score for Voting Classifiers: {0:0.4f}'.format(score_vt_models))

avail_models.append({'name': 'Voting', 'roc_auc': roc_vt_models,
                     'model': voting_model, 'score': score_vt_models})

# Sort based on ROC area values
avail_models.sort(key = lambda sortKey : sortKey['score'], reverse=True)

# Even the voting classifier is not performed better than Logistic Regression,
# but voting classifier is more stable than Logistic Regress,
# so we will still choose Voting classifier model as final model
for i in range(len(avail_models)):
    best_model_name, model_score = (avail_models[i]['name'], avail_models[i]['score'])
    print('{0} - model name: {1}, score: {2:0.4f}'.format(i, best_model_name, model_score))

# The features importance can be shown from Logistic Regression
# and Random Forest
# for i in range(n_classes):
#     show_feature_importance(grid_logreg.best_estimator_, feature_importance_func=get_log_reg_feature_importance,
#                             index=i, max_cols=5)
#     show_feature_importance(grid_rnd_forest.best_estimator_, index=i, max_cols=5)
#     print()
