import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from scipy.sparse import hstack

from SentimentMetricGenerator import generate_sentiment_metric
from TextPreprocess import pre_process_narrative
from Utilities import load_models, get_response_types
from ProductClassifier import PRODUCT_LABELS


def predict(narrative, clf_product, clf_escalation, tf_idf_vectorizer, scaler):
    """
    Given a narrative,
    1. predict the product category
    2. predict the escalation probability based on different response type
    3. draw a bar chart of the probabilites
    4. return response type with lowest escalation probability
    :param narrative:
    :param clf_product: pre-trained product classifier
    :param clf_escalation: pre-trained escalation classifier
    :return: product category, bar chart, the suggest response type
    """
    sentiment_metric = generate_sentiment_metric([narrative])
    sentiment_metric.loc[:, ["word_num", "sentence_num"]] = scaler.transform(sentiment_metric.loc[:, ["word_num", "sentence_num"]])

    # Transfer narrative to feature vector be used by classifier
    preprocessed_narrative = pre_process_narrative(narrative)
    data = pd.DataFrame({"processed_narrative": [preprocessed_narrative]})
    print(data)
    print(preprocessed_narrative)

    narrative_vectorized = tf_idf_vectorizer.transform(data)
    #print(np.sum(narrative_vectorized.toarray()))


    # Predict the product type of this complaint
    product_type = predict_product_type(clf_product, narrative_vectorized)
    print("The complaint is about " + product_type)

    # Predict the probabilities of escalation when adopting
    escalation_prob_fig, response, escalation_probas_according_response = predict_escalation(clf_escalation,
                                                       narrative_vectorized,
                                                       sentiment_metric)

    response = response.split("_")[-1]

    return product_type, escalation_prob_fig, response, escalation_probas_according_response


def predict_product_type(clf_product, narrative_vectorized):
    #product_type_list = clf_product.predict(narrative_vectorized)[0]
    product_type_prob = clf_product.predict_proba(narrative_vectorized)[0]
    #print(product_type_prob)
    index_product_most_prob = np.argmax(product_type_prob)
    #print(index_product_most_prob)
    product_type = PRODUCT_LABELS[index_product_most_prob]
    return product_type


def predict_escalation(clf_escalation, narrative_vectorized, sentiment_metric):
    # Predict probability of dispute according to all different responses
    response_types = get_response_types()
    predict_probability_list = []

    # Add company_response dummy variable in
    column_name_base = "company_response_"
    new_column_names = [column_name_base + x for x in response_types]


    # Predict for each type of response, whether the narrative will end in a dispute
    for response in response_types:
        # One hot code for X_to_predict
        for new_column_name in new_column_names:
            sentiment_metric[new_column_name] = 0
        column_name = column_name_base + response
        sentiment_metric[column_name] = 1

        X_to_predict = hstack((narrative_vectorized, np.array(sentiment_metric)))
        result = clf_escalation.predict(X_to_predict)
        """
        if result:
            print("If respond with {}, there will have escalation!".format(response))
        else:
            print("If respond with {}, there is no escalation!".format(response))
        """
        predict_probability = clf_escalation.predict_proba(X_to_predict)[0][1]
        #print(predict_probability)
        predict_probability_list.append(predict_probability)


    # Draw bar chart of escalation probability under different responses
    escalation_prob_fig = "figs/escalation_prob.png"


    data = pd.DataFrame()
    data["Company Response"] = response_types
    data["Probability of Escalation"] = predict_probability_list

    data.plot.bar()
    plt.xlabel("Company Response Types")
    plt.ylabel("Probability of Escalate")
    plt.xticks(np.arange(6), response_types, size=12)
    plt.yticks(np.arange(0, 1, step=0.1))
    #plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.savefig(escalation_prob_fig)

    # Suggest the response type to be the one with minimum probability to escalate
    suggested_response = response_types[predict_probability_list.index(min(predict_probability_list))]

    return escalation_prob_fig, suggested_response, predict_probability_list


def main():
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
    narrative = "I have a complaint regarding the overdraft fees that were billed to my checking account. I have a complaint regarding the overdraft fees that were billed to mychecking account. I was charged XXXX overcharge fees for XXXX withdrawals in which I had funds in the account. I contact your office and spoke with a representativewho credited me with XXXX of the fees back. However, the XXXX fee was never credited. I just do n't understand how I can billed for an overdraft fee when the fundswere in my accounts. I contacted the office of the president for Flagstar Bank and my compliant was pushed aside. Flagstar has now filed a writ of garnishmentwith my employer."
    product_type, escalation_prob_fig, \
        suggest_response, escalation_probas_according_response = predict(narrative,
                                                                  clf_product,
                                                                  clf_escalation,
                                                                  tf_idf_vectorizer,
                                                                  scaler)
    print("The complaints is about " + product_type)
    print("Suggested response type is " + suggest_response)
    print(escalation_probas_according_response)

main()