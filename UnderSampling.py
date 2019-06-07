import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import StratifiedKFold

from EscalationClassifier import merge_narrative_processed_and_sentiment_metrics, auc_analysis_with_cv
from SMOTEOverSampling import smote_over_sampling
from Utilities import VALIDATION_SIZE


def auc_analysis_with_cv(classifier, X, y, useSMOTE, use_under_sampling):
    """
    Draw roc curves using cross validation
    :param classifier:
    :param X:
    :param y:
    :param useSMOTE:
    :param under_sampling:
    :return:
    """
    cv = StratifiedKFold(n_splits=6)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    X = np.array(X)
    y = np.array(y)
    for train, test in cv.split(X, y):
        if useSMOTE == True:
            X_train_res, y_train_res = smote_over_sampling(X[train], y[train])
            probas_ = classifier.fit(X_train_res, y_train_res).predict_proba(X[test])
        elif under_sampling:
            X_train_under, y_train_under = under_sampling(X[train], y[train])
            tmp = X[test]

            # scale
            #X_train_under, X_test = scale_features(X_train_under, X[test])

            # Feature engineering by vectorizing and generate sentiment metrics
            X_train_under, X_test = feature_engineer(X_train_under, X[test], "undersampling")

            probas_ = classifier.fit(X_train_under, y_train_under).predict_proba(X[test])
        else:
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        # Store roc_auc of each cv fold
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return mean_auc, std_auc


def under_sampling(X_train, y_train):
    sampler = RandomUnderSampler(sampling_strategy='majority',
                                 random_state=0)
    X_train_under, y_train_under = sampler.fit_sample(X_train, y_train)
    return X_train_under, y_train_under


"""
def choose_best_undersampling_ratio(classifier, X, y):
    ratios = [np.arange(0.2, 1, 0.1)]
    mean_auc_list = []
    std_auc_list = []

    useSMOTE = False
    under_sampling = True
    for ratio in ratios:
        mean_auc, std_auc = auc_analysis_with_cv(classifier, X, y, useSMOTE, under_sampling,ratio)
        mean_auc_list.append(mean_auc)
        std_auc_list.append(std_auc)

    print(ratios)
    print(mean_auc_list)
    print(std_auc_list)
    plt.plot(ratios, mean_auc_list)
    plt.line()
"""


def main():
    # Load the feature and label csv file, join them according to complaint ID
    complaints_with_sentiment = "data/complaints_with_sentiment_metric.csv"
    narrative_preprocessed_file = "data/narrative_preprocessed.csv"
    complaints_features = merge_narrative_processed_and_sentiment_metrics(narrative_preprocessed_file,
                                                                          complaints_with_sentiment)
    print(complaints_features.head())
    # omit the first validation_size complaints to be validation set
    complaints_features = complaints_features[VALIDATION_SIZE:]

    X = complaints_features
    y = [0 if x == "No" else 1 for x in complaints_features["Consumer disputed?"]]

    classifier = LogisticRegression(C=1, solver="lbfgs", max_iter=2000)

    useSMOTE = False
    use_under_sampling = True
    auc_analysis_with_cv(classifier, X, y, useSMOTE, use_under_sampling)

#main()