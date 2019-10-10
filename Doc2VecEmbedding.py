import gensim
import numpy as np
import pandas as pd
import os

from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from Utilities import load_model, save_model
from UnderSampling import under_sampling
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from EscalationClassifier import model_evaluate
from ImbalancedDataSampling import smote_over_sampling
from Utilities import scale_features, draw_roc_curve, VALIDATION_SIZE


def tokenize_narratives(complaints, tokens_only=False):
    """
    Tokenize narratives and tag them if tokens_only=False
    :param complaints:
    :param tokens_only:
    :return: tokenized narratives
    """
    results = []

    for i in np.arange(len(complaints)):
        narrative = complaints["Consumer complaint narrative"].values[i]
        complaint_id = complaints["Complaint ID"].values[i]

        if tokens_only:
            results.append(gensim.utils.simple_preprocess(narrative))
        else:
            results.append(TaggedDocument(gensim.utils.simple_preprocess(narrative), [complaint_id]))

    return results


def dump_doc2vec_model(model, save_dir, tag):
    fname = save_dir + os.sep + "doc2vec_" + tag

    model.save(fname)


def build_scaler(X_train):
    scaler = MinMaxScaler()
    scaler.fit(X_train[:, :-4])

    save_model(scaler, "trained_models/scaler.doc2vec.joblib")


def scale(scaler, X):
    X[:, :-4] = scaler.transform(X[:, :-4])


def load_doc2vec_model(model_file):
    model = Doc2Vec.load(model_file)
    return model


def train_doc2vec(X_train, vector_size, min_count, epochs, save_dir, tag):
    """
    Train the doc2vec model based on X_train and save the model.
    :param X_train: data to train doc2vec model
    :param vector_size: The size of doc2vec vector
    :param min_count:
    :param epochs:
    :param save_dir:
    :param tag:
    :return: The doc2vec model trained based on X_train
    """
    tokenized_narratives = tokenize_narratives(X_train)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size,
                                          min_count=min_count,
                                          window=10,
                                          dm=1,
                                          epochs=epochs)
    model.build_vocab(tokenized_narratives)

    print("Saving doctvec_model...")
    dump_doc2vec_model(model, save_dir, tag)

    return model


def doc2vec_vectorize(model, tokenized_narratives):
    """
    Using pre-trained doc2vec model, vectorized complaint narratives
    :param model: doc2vec model trained on training data
    :param tokenized_narratives: a list of narrative tokens
    :return: vectorized complaints
    """
    narratives_vectorized = []
    for tokenized_narrative in tokenized_narratives:
        narratives_vectorized.append(model.infer_vector(tokenized_narrative))
    return narratives_vectorized


def combine_features(X_vectorized, X_sentiment_metric):
    """
    Combine vectorized X and sentiment_metrics as input features for escalate classifier
    :param X_vectorized:
    :param X_sentiment_metric:
    :return:
    """
    print("Combing features...")
    size = len(X_vectorized)
    # col_num = X_sentiment_metric.shape[1]
    for i in np.arange(size):
        X_vectorized[i] = np.append(X_vectorized[i], X_sentiment_metric.values[i])
        """
        sentiment_metric = X_sentiment_metric.values[i]
        for col_index in np.arange(col_num):
            X_vectorized[i].append(sentiment_metric[col_index])
        """

    return X_vectorized


def feature_engineering(X, doc2vec_model):
    """
    Vectorize X using pre-trained doc2vec_model, and combine sentiment metrics to
    form features
    :param X:
    :param doc2vec_model:
    :return: features ready for training(when X is X_train) or evaluating(when
     X is X_test) classifier
    """
    print("Vectorizing...")
    X_narratives = tokenize_narratives(X, True)
    X_vectorized = doc2vec_vectorize(doc2vec_model, X_narratives)

    print(X_vectorized[0])

    X_sentiment_metric = X.loc[:, "corpus_score_sum":"company_response_Closed with non-monetary relief"]

    X_features = combine_features(X_vectorized, X_sentiment_metric)
    print(len(X_features[0]))

    return X_features


def build_classifier(X_train, y_train, escalate_classifier, tag, save_dir):
    # under sampling
    #X_train_res, y_train_res = under_sampling(X_train, X_test)

    print("Training doc2vec_model...")
    vector_size, min_count, epochs = 300, 2, 20
    doc2vec_model = train_doc2vec(X_train, vector_size, min_count, epochs, save_dir, tag)

    X_train = feature_engineering(X_train, doc2vec_model)

    """
    # scale
    scaler = build_scaler(X_train)
    X_train = scale(scaler, X_train)
    """

    # Oversampling using SMOTE
    print("Oversampling...")
    X_train, y_train = smote_over_sampling(X_train, y_train)

    print("Fit the classifier")
    escalate_classifier.fit(X_train, y_train)

    # evaluating on training data
    fpr, tpr, model_auc = model_evaluate(escalate_classifier, X_train, y_train, False)

    print("Save the escalate classifier...")
    dump(escalate_classifier, open(save_dir + os.sep + "{}.joblib".format(tag), "wb"))

    return escalate_classifier, doc2vec_model


def evaluate_model(escalate_classifier, doc2vec_model, X_test, y_test, is_rf, tag):
    print("Vectorize and feature engineering for X_test...")
    X_test = feature_engineering(X_test, doc2vec_model)

    """
    # scale
    X_test = scale(scaler, X_test)
    """

    print("Evaluating the model...")
    fpr, tpr, model_auc = model_evaluate(escalate_classifier, X_test, y_test, is_rf)

    return fpr, tpr, model_auc


def main():
    complaints_with_sentiment = pd.read_csv("data/complaints_with_sentiment_metric.csv")
    complaints_with_sentiment = complaints_with_sentiment[VALIDATION_SIZE:]

    X = complaints_with_sentiment.drop(columns=["dispute"])
    y = [0 if x == "No" else 1 for x in complaints_with_sentiment["Consumer disputed?"]]

    # One hot coding for company_response category
    X = pd.get_dummies(X, columns=["company_response"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model_save_dir = "trained_models"

    # The classifier model to run
    classifier_model = LogisticRegression(C=100, solver="lbfgs", max_iter=2000)
    model_tag = "lgrg"
    # classifier_model = GradientBoostingClassifier(random_state=0)
    # model_tag = "gbm"

    tag = "doc2vec.all"
    # Build the doc2vec_model and escalate classifier using training data
    escalate_classifier, doc2vec_model = build_classifier(X_train,
                                                          y_train,
                                                          classifier_model,
                                                          model_tag+"_"+tag,
                                                          model_save_dir)

    # Load pre-trained doc2vec and classifier models
    # escalate_classifier = load_model(model_save_dir + os.sep + "lgrg_doc2vec.all.joblib")
    # doc2vec_model = load_doc2vec_model(model_save_dir + os.sep + "doc2vec_lgrg_doc2vec.all")

    is_rf = False
    fpr, tpr, model_auc = evaluate_model(escalate_classifier,
                                         doc2vec_model,
                                         X_test,
                                         y_test,
                                         is_rf,
                                         tag)
    print("The auc score for {} is {:.3f}".format(tag, model_auc))
    title = "ROC curve for escalate classifier for " + model_tag + "." + tag
    save_file = "figs/roc_escalation_classifier_" + model_tag + "." + tag + ".png"
    draw_roc_curve(title, save_file, [fpr], [tpr], [model_auc], [model_tag])

main()
