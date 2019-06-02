from SentimentMetricGenerator import generate_sentiment_metric

import pandas as pd
import numpy as np
import pickle

from Utilities import load_model

# TODO outdated
def predict_dispute(saved_model_file, column_name_file, narrative):
    # Load the pre-trained classifier
    model = pickle.load(open(saved_model_file, "rb"))

    # Load in column names
    with open(column_name_file, "r") as fobj:
        line = fobj.readline()
        new_column_names = line.split(",")

    # Transform narrative to sentiment metric
    complaint = pd.DataFrame({"Consumer complaint narrative": [narrative]})
    X_to_predict = generate_sentiment_metric(complaint)

    # Predict probability of dispute according to all different responses
    response_types = ['Closed', 'Closed with explanation', 'Closed with monetary relief', 'Closed with non-monetary relief', 'None', 'Untimely response']
    predict_values = []
    predict_probability_list = []

    column_name_base = "company_response_"

    # Predict for each type of response, whether the narrative will end in a dispute
    for response in response_types:
        # One hot code for X_to_predict
        for new_column_name in new_column_names:
            X_to_predict[new_column_name] = 0
        column_name = column_name_base + response
        X_to_predict[column_name] = 1
        #print(X_to_predict)

        result = model.predict(X_to_predict)
        predict_probability = model.predict_proba(X_to_predict)
        predict_values.append(result[0])
        predict_probability_list.append(predict_probability)

    for i in np.arange(len(response_types)):
        response = response_types[i]
        predict_value = predict_values[i]
        predict_prob = predict_probability_list[i]

        print("If adopt {} response, there will be a {:.2f}% chance to result in a dispute!".format(response, predict_prob[0][1] * 100))


narrative = "I have a complaint regarding the overdraft fees that were billed to my checking account. I have a complaint regarding the overdraft fees that were billed to mychecking account. I was charged XXXX overcharge fees for XXXX withdrawals in which I had funds in the account. I contact your office and spoke with a representativewho credited me with XXXX of the fees back. However, the XXXX fee was never credited. I just do n't understand how I can billed for an overdraft fee when the fundswere in my accounts. I contacted the office of the president for Flagstar Bank and my compliant was pushed aside. Flagstar has now filed a writ of garnishmentwith my employer."
saved_model_file = "trained_models/finalized_model.sav"
column_name_file = "result/column_names.csv"
predict_dispute(saved_model_file, column_name_file, narrative)

def main():
    product_classifier_file = "trained_models/dispute_classifier.lgreg.Debt_collection.pickle"
    product_classifier = load_model(product_classifier_file)
    tfidf_vectorizer_file = "trained_models/tfidf_vectorizer_max10000.Debt_collection.pickle"
    tfidf_vectorizer = load_model(tfidf_vectorizer_file)