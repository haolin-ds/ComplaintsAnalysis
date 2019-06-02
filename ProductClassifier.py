import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from SMOTEOverSampling import smote_over_sampling

from Utilities import save_model

PRODUCT_LABELS = ['Bank account or service',
                  'Credit reporting',
                  'Mortgage',
                  'Debt collection',
                  'Credit card',
                  'Student loan',
                  'Payday loan',
                  'Consumer Loan',
                  'Money transfers',
                  'Prepaid card',
                  'Other financial service',
                  'Virtual currency']


def load_tf_idf_vecterizer(model_pickle_file):
    """
    Load pre-trained tf-idf vectorizer
    :param vectors_pickle_file:
    :return:
    """
    with open(model_pickle_file, "rb") as f:
        tf_idf_vectorizer = pickle.load(f)

    return tf_idf_vectorizer


def load_narratives_vectors(vectors_pickle_file):
    """
    Load vectorized narratives
    :param vectors_pickle_file:
    :return:
    """
    with open(vectors_pickle_file, 'rb') as f:
        narratives_vectorized = pickle.load(f)

    return narratives_vectorized


def feature_engineering(vectors_pickle_file):
    complaints = pd.read_csv("data/complaints-2019-05-16_13_17.csv")
    narrative_processed = pd.read_csv("data/narrative_preprocessed.csv")
    data = pd.merge(complaints[["Complaint ID", "Product"]],
                    narrative_processed[["Complaint ID", "Consumer disputed?", "processed_narrative"]], how='inner',
                    on=['Complaint ID', 'Complaint ID'])

    n_classes = len(PRODUCT_LABELS)
    print("Number of Product categories:", n_classes)

    X = load_narratives_vectors(vectors_pickle_file)
    y = label_binarize(data["Product"], classes=PRODUCT_LABELS)
    print(X.shape)
    print(y.shape)

    return X, y


def multi_classifier(X, y, classifier, product_labels_name, use_SMOTE):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)

    multi_class_classifier = OneVsRestClassifier(classifier)
    if use_SMOTE:
        X_trainval_res, y_trainval_res = smote_over_sampling(X_trainval, y_trainval)
        y_score = multi_class_classifier.fit(X_trainval_res, y_trainval_res).decision_function(X_test)
    else:
        y_score = multi_class_classifier.fit(X_trainval, y_trainval).decision_function(X_test)

    n_classes = len(product_labels_name)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area. Adopt micro-average ROC other
    # than macro-average ROC for imbalanced data
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("average AUC score: ", roc_auc["micro"])

    draw_roc_curve(fpr, tpr, roc_auc, product_labels_name)

    return multi_class_classifier


def draw_roc_curve(fpr, tpr, roc_auc, product_labels_name):
    # Plot all ROC curves
    plt.figure(figsize=(9, 8))

    n_classes = len(product_labels_name)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lw = 2 # line width

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw = lw,
                 label='{0} (area = {1:0.2f})'
                       ''.format(product_labels_name[i], roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of multi-class classification of different Product categories')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("figs/ROC_Curve_Product.png")


def main():
    vectors_pickle_file = "trained_models/narratives_vectorized_tf-idf_max10000.all.pickle"
    X, y = feature_engineering(vectors_pickle_file)

    classifier = LogisticRegression(C=1, solver="lbfgs",
                                    max_iter=2000,
                                    random_state=0)
    use_SMOTE = True
    product_classifier = multi_classifier(X, y, classifier, PRODUCT_LABELS, use_SMOTE)

    print("Saving the product_classifier")
    model_export_file = "trained_models/product_classifier_lgreg.sav"
    save_model(product_classifier, model_export_file)

#main()