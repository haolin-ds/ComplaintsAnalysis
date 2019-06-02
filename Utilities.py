import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def save_model(model, output_file):
    # save the model as final_model
    pickle.dump(model, open(output_file, "wb"))


def load_model(saved_model_file):
    with open(saved_model_file, 'rb') as f:
        temp_model = pickle.load(f)
    return temp_model


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train.loc[:, ["word_num", "sentence_num"]])

    save_model(scaler, "trained_models/scaler.pickle")

    X_train.loc[:, ["word_num", "sentence_num"]] = scaler.transform(X_train.loc[:, ["word_num", "sentence_num"]])
    X_test.loc[:, ["word_num", "sentence_num"]] = scaler.transform(X_test.loc[:, ["word_num", "sentence_num"]])

    return X_train, X_test


def get_response_types():
    response_column_names_file = "trained_models/company_corresponse_variable_names.csv"

    with open(response_column_names_file, "r") as fobj:
        line = fobj.readline()
        response_types = line.rstrip().split(",")

    return response_types



def load_models(clf_product_file, clf_escalation_file, tf_idf_vectorizer_file, scaler_file):
    clf_product = load_model(clf_product_file)
    clf_escalation = load_model(clf_escalation_file)
    tf_idf_vectorizer = load_model(tf_idf_vectorizer_file)
    scaler = load_model(scaler_file)

    return clf_product, clf_escalation, tf_idf_vectorizer, scaler


