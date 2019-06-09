import pandas as pd
from joblib import dump, load

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from SMOTEOverSampling import smote_over_sampling
from TextPreprocess import tf_idf_vectorize, dump_tf_idf_model

from Utilities import save_model, draw_roc_curve, VALIDATION_SIZE, PRODUCT_LABELS


def load_tf_idf_vecterizer(model_joblib_file):
    """
    Load pre-trained tf-idf vectorizer
    :param vectors_joblib_file:
    :return:
    """
    with open(model_joblib_file, "rb") as f:
        tf_idf_vectorizer = load(f)

    return tf_idf_vectorizer


def load_narratives_vectors(vectors_joblib_file):
    """
    Load vectorized narratives
    :param vectors_joblib_file:
    :return:
    """
    with open(vectors_joblib_file, 'rb') as f:
        narratives_vectorized = load(f)

    return narratives_vectorized


def feature_engineering():
    complaints = pd.read_csv("data/complaints-2019-05-16_13_17.clean.csv")
    narrative_processed = pd.read_csv("data/narrative_preprocessed.csv")
    data = pd.merge(complaints[["Complaint ID", "Product"]],
                    narrative_processed[["Complaint ID", "Consumer disputed?", "processed_narrative"]], how='inner',
                    on=['Complaint ID', 'Complaint ID'])
    # Only use non_validation data
    data = data[VALIDATION_SIZE:]
    n_classes = len(PRODUCT_LABELS)
    print("Number of Product categories:", n_classes)

    X = data["processed_narrative"]
    y = label_binarize(data["Product"], classes=PRODUCT_LABELS)
    print(X.shape)
    print(y.shape)

    return X, y


def multi_classifier(X, y, classifier, product_labels_name, use_SMOTE):
    """
    Based on vectorized narrative X, and labels in y, build a multi-classifier
    to assign product type for complaints
    :param X: processed narratives
    :param y: product label
    :param classifier: the classifier to be used
    :param product_labels_name: the product label names
    :param use_SMOTE: whether to use SMOTE or not
    :return: the classifier and the roc curve is drawn
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)

    print("tf-idf vectorizing...")
    tf_idf_vectorizer, X_trainval_vectorized, \
        max_feature_num = tf_idf_vectorize(X_trainval)
    X_test_vectorized = tf_idf_vectorizer.transform(X_test)

    print("Saving Tf-idf model")
    save_dir = "trained_models"
    tag = "product_classifier"
    dump_tf_idf_model(tf_idf_vectorizer, max_feature_num, save_dir, tag)

    multi_class_classifier = OneVsRestClassifier(classifier)

    print("Fit the model...")
    if use_SMOTE:
        X_trainval_res, y_trainval_res = smote_over_sampling(X_trainval_vectorized, y_trainval)
        y_score = multi_class_classifier.fit(X_trainval_res, y_trainval_res).decision_function(X_test_vectorized)
    else:
        y_score = multi_class_classifier.fit(X_trainval_vectorized, y_trainval).decision_function(X_test_vectorized)

    print("Draw the Roc curve...")
    # Compute ROC curve and ROC area for each class
    n_classes = len(product_labels_name)
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

    title = "ROC Curve of multi-class classification of different Product categories"
    save_file = "figs/ROC_Curve_Product.png"
    draw_micro = True
    draw_roc_curve(title, save_file, fpr, tpr, roc_auc, product_labels_name, True)

    return multi_class_classifier


def main():
    X, y = feature_engineering()

    classifier = LogisticRegression(C=1, solver="lbfgs",
                                    max_iter=2000,
                                    random_state=0)
    use_SMOTE = True
    product_classifier = multi_classifier(X, y, classifier, PRODUCT_LABELS, use_SMOTE)

    print("Saving the product_classifier")
    model_export_file = "trained_models/product_classifier_lgreg.sav"
    save_model(product_classifier, model_export_file)

#main()