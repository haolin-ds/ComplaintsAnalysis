from EscalationClassifier import EscalationClassifier
from WordEmbedding import WordEmbedding
from TextPreprocess import TextPreprocesser
from ProductClassifier import ProductClassifier
from EscalationClassifier import LOGISTIC_REGRESSION_L1, LOGISTIC_REGRESSION_L2, RANDOM_FOREST, GRADIENT_BOOSTING


def main():

    complaint_file = "data/complaints-2019-05-16_13_17.csv"

    # Extract complaints with narratives
    text_preprocesser = TextPreprocesser(complaint_file)
    output_file = "data/complaints-2019-05-16_13_17.with_narrative.csv"
    # Only need to run once
    text_preprocesser.extract_complaints_with_narrative(output_file)

    # Preprocess text for product categorize task and escalatin prediction task
    preprocessed_file_for_escalation = "data/narrative_preprocessed.for_escalation_prediction.csv"
    preprocessed_file_for_product = "data/narrative_preprocessed.for_product_categorize.csv"
    text_preprocesser.text_preprocess(preprocessed_file_for_escalation, preprocessed_file_for_product)

    # Embedded word separately for two tasks
    word_embeder = WordEmbedding(preprocessed_file_for_escalation, preprocessed_file_for_product)
    word_embeder.run_tf_idf()

    # Product Classifier
    tf_idf_model_file_for_product = "trained_models/tfidf_vectorizer_max50000.for_product_categorize.joblib"
    complaint_file = "data/complaints-2019-05-16_13_17.clean.csv"
    product_classifier = ProductClassifier(tf_idf_model_file_for_product, complaint_file, preprocessed_file_for_product)
    product_classifier.product_classify()

    # Escalation Classifier
    tf_idf_model_file = "trained_models/tfidf_vectorizer_max50000.for_escalation_classifier.joblib"
    complaints_with_sentiment_file = "data/complaints_with_sentiment_metric.csv"
    escalation_classifier = EscalationClassifier(tf_idf_model_file,
                                       complaints_with_sentiment_file,
                                       preprocessed_file_for_escalation)

    model_flag = LOGISTIC_REGRESSION_L2
    escalation_classifier.escalation_classifier(model_flag)


main()