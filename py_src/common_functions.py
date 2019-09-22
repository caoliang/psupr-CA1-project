import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, multilabel_confusion_matrix
#import warnings
#warnings.filterwarnings('ignore')

from sklearn.preprocessing import label_binarize

#sample_data = pd.read_csv('./../fifa19_ready_data.csv', encoding='utf-8')
#sample_data = pd.read_csv('./../fifa19_features_reduced_data.csv', encoding='utf-8')
sample_data = pd.read_csv('./../fifa19_features_reduced_data.csv', encoding='utf-8')


# Separate international rating result with rest
y = sample_data['International Reputation']
X = sample_data.drop('International Reputation', axis=1)

print('X: ', X.shape, 'y:', y.shape)

# Binarize the rating result
# Rating score 4 and 5 records has only a few records of out total 18208, so it cannot be predicated with too little data,
# we convert the score 5 and 4 to 3, so only choose rating scores 1, 2, 3 to classify.
sample_data['International Reputation'].loc[sample_data['International Reputation'] == 5] = 3
sample_data['International Reputation'].loc[sample_data['International Reputation'] == 4] = 3
sample_data.hist(column='International Reputation')
print(sample_data['International Reputation'].value_counts())

y = label_binarize(y, classes=[1, 2, 3])
n_classes = y.shape[1]
print('n_classes: ', n_classes)

# Divide data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=1)
print('X_train: ', X_train.shape, 'X_test: ', X_test.shape)

# print(y_train)
for i in range(n_classes):
    print("Rating Score value counts: ", i + 1)
    print(pd.value_counts(y_train[:, i]))
    plt.hist(y_train[:, i], bins=3)
    plt.show()
    print(" ")


# Standardize records
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('X_train: ', X_train.shape, 'X_test: ', X_test.shape)


from sklearn.metrics import roc_auc_score, roc_curve, auc, make_scorer
from scipy import interp
from itertools import cycle


# Create function to store ROC computation results for all classification classes
def compute_roc_auc_in_classes(test_data, test_result, num_classes=n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    roc_result = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_data[:, i], test_result[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_data.ravel(), test_result.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Compute macro-average ROC curve and ROC area
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_result


# Draw ROC plot based on ROC computation result
def draw_roc_auc_in_classes(roc_auc_result, learn_method, num_classes=n_classes):
    fpr = roc_auc_result['fpr']
    tpr = roc_auc_result['tpr']
    roc_auc = roc_auc_result['roc_auc']

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='Average ROC (area = {0:0.4f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='ROC of rating {0} (area = {1:0.4f})'.format((i + 1), roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for {}'.format(learn_method))
    plt.legend(loc="lower right")
    plt.show()


def show_time_spent(start_time):
    time_used = time.time() - start_time
    return "{}:{}:{}".format(int(time_used / 3600), int(time_used % 3600 / 60), int(time_used % 60))

def print_confusion_matix(y_test_data, y_pred_data):
    # Print accuracy and confusion matrix
    print()
    print(multilabel_confusion_matrix(y_true=y_test_data, y_pred=y_pred_data))
    print(classification_report(y_true=y_test_data, y_pred=y_pred_data, digits=4))
    print()

def get_log_reg_feature_importance(log_reg_clf):
    feature_importance = abs(log_reg_clf.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.sum())
    return feature_importance

# Draw feature importance
def show_feature_importance(one_vs_rest_clf, feature_importance_func=None, columns_names=X.columns, index=-1,
                            max_cols=-1):
    clfs = one_vs_rest_clf.estimators_

    if index >= 0:
        if feature_importance_func is None:
            feature_importance = clfs[index].feature_importances_
        else:
            feature_importance = feature_importance_func(clfs[index])

        last_col = max_cols if max_cols > 0 else len(feature_importance)

        feat_imp = pd.Series(feature_importance[0:last_col], index=columns_names[0:last_col]).sort_values(
            ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances for Rating {}'.format(index + 1))
        plt.show()
        return

    for i in range(len(clfs)):
        if feature_importance_func is None:
            feature_importance = clfs[i].feature_importances_
        else:
            feature_importance = feature_importance_func(clfs[i])

        last_col = max_cols if max_cols > 0 else len(feature_importance)
        feat_imp = pd.Series(feature_importance[0:last_col], index=columns_names[0:last_col]).sort_values(
            ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances for Rating {}'.format(i + 1))
        plt.show()