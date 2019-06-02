from TextPreprocess import pre_process_narrative
from Utilities import load_model
import pandas as pd


def predict_product(product_classifier, tfidf_vectorizer, preprocessed_narrative, label_names):
    df = pd.DataFrame({"processed_narrative": [preprocessed_narrative]})
    print(df)

    vectorized_narrative = tfidf_vectorizer.transform(df)
    print(vectorized_narrative.shape)
    y = product_classifier.predict(vectorized_narrative)

    print(y.shape)
    print("Prediction is ", y)
    return label_names[y]


def main():
    product_classifier_file = "trained_models/product_classifier_lgreg.sav"
    product_classifier = load_model(product_classifier_file)
    tfidf_vectorizer_file = "trained_models/tfidf_vectorizer_max10000.all.pickle"
    tfidf_vectorizer = load_model(tfidf_vectorizer_file)

    label_names = ['Bank account or service',
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
    narrative = "I have a complaint regarding the overdraft fees that were billed to my checking account. I have a complaint regarding the overdraft fees that were billed to mychecking account. I was charged XXXX overcharge fees for XXXX withdrawals in which I had funds in the account. I contact your office and spoke with a representativewho credited me with XXXX of the fees back. However, the XXXX fee was never credited. I just do n't understand how I can billed for an overdraft fee when the fundswere in my accounts. I contacted the office of the president for Flagstar Bank and my compliant was pushed aside. Flagstar has now filed a writ of garnishmentwith my employer."

    preprocessed_narrative = pre_process_narrative(narrative)
    print(preprocessed_narrative)
    predict_product(product_classifier, tfidf_vectorizer, preprocessed_narrative, label_names)

#main()