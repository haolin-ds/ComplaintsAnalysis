import gensim
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from UnderSampling import under_sampling

from EscalationClassifier import logistic_regression_model
from SMOTEOverSampling import smote_over_sampling
from Utilities import scale_features, draw_roc_curve, VALIDATION_SIZE


def read_narratives(complaints, tokens_only=False):
    results = []

    for i in np.arange(len(complaints)):
        narrative = complaints["Consumer complaint narrative"].values[i]
        complaint_id = complaints["Complaint ID"].values[i]
        if tokens_only:
            results.append(gensim.utils.simple_preprocess(narrative))
        else:
            results.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(narrative), [complaint_id]))

    return results


def dump_doc2vec_model(model, save_dir, tag):
    fname = save_dir + os.sep + "doc2vec_" + tag

    model.save(fname)


def load_doc2vec_model(model_file):
    model = gensim.Doc2Vec.load(model_file)
    return model


def train_doc2vec(X_train, vector_size, min_count, epochs):
    tokenized_narratives = read_narratives(X_train)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=20)
    model.build_vocab(tokenized_narratives)

    return model


def doc2vec_vectorize(model, tokenized_narratives):
    narratives_vectorized = []
    for tokenized_narrative in tokenized_narratives:
        narratives_vectorized.append(model.infer_vector(tokenized_narrative))
    return narratives_vectorized


def combine_features(X_vectorized, X_sentiment_metric):
    size = len(X_vectorized)
    features = []
    for i in np.arange(size):
        feature = []
        feature.append(X_vectorized[i])
        feature.append(X_sentiment_metric.values[i])
        features.append(feature)
    return features


def feature_engineering(X_train, X_test, save_dir, tag):
    print("Training doc2vec_model...")
    vector_size, min_count, epochs = 50, 2, 20
    doc2vec_model = train_doc2vec(X_train, vector_size, min_count, epochs)

    print("Saving doctvec_model...")
    dump_doc2vec_model(doc2vec_model, save_dir, tag)

    print("Vectorizing...")
    X_train_narratives = read_narratives(X_train, True)
    X_train_vectorized = doc2vec_vectorize(doc2vec_model, X_train_narratives)
    print(X_train.shape)

    X_test_narratives = read_narratives(X_test, True)
    X_test_vectorized = doc2vec_vectorize(doc2vec_model, X_test_narratives)


    X_train_sentiment_metric = X_train.loc[:, "corpus_score_sum":"company_response_Closed with non-monetary relief"]


    X_train = combine_features(X_train_vectorized, X_train_sentiment_metric)
    print(X_train.shape)

    # X_test
    X_test_sentiment_metric = X_test.loc[:, "corpus_score_sum":"company_response_Closed with non-monetary relief"]
    X_test = combine_features(X_test_vectorized, X_test_sentiment_metric)

    return X_train, X_test


def build_classifier(complaints_with_sentiment, classifier_model, tag, save_dir):

    y = [0 if x == "No" else 1 for x in complaints_with_sentiment["Consumer disputed?"]]
    X = complaints_with_sentiment

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # scale
    X_train, X_test = scale_features(X_train, X_test)

    # under sampling
    #X_train_res, y_train_res = under_sampling(X_train, X_test)

    X_train, X_test = feature_engineering(X_train, X_test, save_dir, tag)


    # Oversampling using SMOTE
    X_trainval_res, y_trainval_res = smote_over_sampling(X_train, y_train)

    fpr, tpr, model_auc = classifier_model(X_trainval_res, X_test, y_trainval_res, y_test, tag, save_dir)

    return fpr, tpr, model_auc


def main():
    complaints_with_sentiment = pd.read_csv("data/complaints_with_sentiment_metric.csv")

    complaints_with_sentiment = complaints_with_sentiment[VALIDATION_SIZE:]

    # One hot coding for company_response category
    complaints_with_sentiment = pd.get_dummies(complaints_with_sentiment, columns=["company_response"])

    model_save_dir = "trained_models"

    # The classifier model to run
    classifier_model = logistic_regression_model
    model_tag = "lgrg"
    # classifier_model = gradient_boosting_model
    # model_tag = "gbm"

    # Build a classifier for all category together
    tag = "doc2vec.all"
    fpr, tpr, model_auc = build_classifier(complaints_with_sentiment, classifier_model, tag, model_save_dir)
    print("The auc score for {} is {:.3f}".format(tag, model_auc))
    title = "ROC curve for escalate classifier for " + model_tag + "." + tag
    save_file = "figs/roc_escalation_classifier_" + model_tag + "." + tag + ".png"
    draw_roc_curve(title, save_file, [fpr], [tpr], [model_auc], [model_tag])

main()