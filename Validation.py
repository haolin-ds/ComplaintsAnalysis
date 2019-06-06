import pandas as pd
import numpy as np

from EscalationClassifier import merge_narrative_processed_and_sentiment_metrics
from Utilities import load_models, get_response_types
from Predict import predict


def will_escalate(escalation_probas_according_response):
    """If no probability is larger than 0.5, there won't be escalation"""
    for proba in escalation_probas_according_response:
        if proba > 0.5:
            return True

    return False

def get_response_index(binary_list_of_company_response):
    """Find the index where element is 1"""
    for i in np.arange(len(binary_list_of_company_response)):
        if binary_list_of_company_response[i] == 1:
            return i


def validate():
    # Load the feature and label csv file, join them according to complaint ID
    complaints_with_sentiment = "data/complaints_with_sentiment_metric.csv"
    narrative_preprocessed_file = "data/narrative_preprocessed.csv"
    complaints_features = merge_narrative_processed_and_sentiment_metrics(narrative_preprocessed_file,
                                                                          complaints_with_sentiment)
    # Extract the validation data
    #print(complaints_features.head())
    validation_size = 10
    complaints_features_for_validation = complaints_features[:validation_size]
    #print(complaints_features_for_validation.columns)

    company_response_columns = complaints_features_for_validation.loc[:, 'company_response_Closed':'company_response_Untimely response']

    narratives = complaints_features_for_validation["Consumer complaint narrative"]
    company_responses = []
    predicted_product_types = []
    predicted_disputes = []
    suggested_responses = []
    escalate_or_not = []

    print("Loading models...")
    model_dir = "trained_models"
    clf_product_file = model_dir + "/" + "product_classifier_lgreg.sav"
    clf_escalation_file = model_dir + "/" + "lgreg.all.pickle"
    tf_idf_vectorizer_file = model_dir + "/" + "tfidf_vectorizer_max10000.all.pickle"
    scaler_file = model_dir + "/" + "scaler.pickle"
    clf_product, clf_escalation, tf_idf_vectorizer, scaler = load_models(clf_product_file,
                                                                         clf_escalation_file,
                                                                         tf_idf_vectorizer_file,
                                                                         scaler_file)
    print("Predicting...")
    i = 0
    for narrative in narratives:
        product_type, escalation_prob_fig, \
            suggest_response, escalation_probas_of_responses = predict(narrative,
                                                                      clf_product,
                                                                      clf_escalation,
                                                                      tf_idf_vectorizer,
                                                                      scaler)
        company_response_index = get_response_index(company_response_columns.loc[i, :])
        i += 1
        company_responses.append(get_response_types()[company_response_index])
        escalate_or_not.append(will_escalate(escalation_probas_of_responses))
        predicted_product_types.append(product_type)
        predict_dispute = (1 if escalation_probas_of_responses[company_response_index] > 0.5 else 0)
        predicted_disputes.append(predict_dispute)
        suggested_responses.append(suggest_response)

    predict_result = pd.DataFrame()

    predict_result["Complaint ID"] = complaints_features_for_validation["Complaint ID"]
    predict_result["Consumer complaint narrative"] = complaints_features_for_validation["Consumer complaint narrative"]
    predict_result["Company response to consumer"] = company_responses
    predict_result["Product"] = complaints_features_for_validation["Product"]

    # Assign the predicted result
    predict_result["predicted_product"] = predicted_product_types
    predict_result["predicted_dispute"] = predicted_disputes
    predict_result["suggested_response"] = suggested_responses
    predict_result["escalate_or_not"] = escalate_or_not

    print(predict_result.head())
    predict_result.to_csv("data/predicted_result_of_validation.csv")


validate()


