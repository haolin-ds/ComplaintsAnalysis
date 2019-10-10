import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

from TextPreprocess import TextPreprocesser
from Utilities import load_model


class WordEmbedding:
    def __init__(self, preprocessed_file_for_escalation, preprocessed_file_for_product):
        self.preprocessed_file_for_escalation = preprocessed_file_for_escalation
        self.preprocessed_file_for_product = preprocessed_file_for_product

    def tf_idf_vectorize(self, pre_processed_narratives, min_df=5):
        """
        Build tf-idf-vectorizer model using narratives
        :param pre_processed_narratives: tokened narratives in dataframe columns
        :param suffix:  all, product_name
        :return: model, and vectorized narratives
        """
        max_feature_num = 50000
        tf_idf_vectorizer = TfidfVectorizer(min_df=min_df,
                                            ngram_range=(1, 3),
                                            max_features=max_feature_num)

        tf_idf_vectorizer.fit(pre_processed_narratives)

        return tf_idf_vectorizer, max_feature_num

    def dump_tf_idf_model(self, tf_idf_vectorizer, save_file):
        print("Saving tf-idf model to file " + save_file)
        # dump tf-idf vectorizer to file
        dump(tf_idf_vectorizer,
                    open(save_file, "wb"))

    def get_tf_idf_vector(self, tf_idf_vectorizer, narrative):
        """
        Given a new narrative, pre-process it and vectorize it using
        pretrained tf_idf_vectorizer
        :param tf_idf_vectorizer: pre-trained tf-idf vectorizer
        :param narrative:
        :return: narrative_vectorized
        """
        pre_processed_narrative = TextPreprocesser.pre_process_one_narrative(narrative)
        return tf_idf_vectorizer.transform(pre_processed_narrative)

    def run_tf_idf(self):
        """Building two tf-idf models based on pre-processed complaint narratives.
        One for escalation prediction, the other for categrizing product."""

        """
        # run tf-idf on all complaints and dump them for escalation prediction
        print("Loading complaints with processed narrative...")
        complaints_narrative = pd.read_csv(self.preprocessed_file_for_escalation)
        processed_narratives = complaints_narrative["processed_narrative"]
        print("Vectorizing use tf-idf for escalation prediction...")
        tfidf, max_feature_num = self.tf_idf_vectorize(processed_narratives)

        save_dir = "trained_models"
        tag = "for_escalation_classifier"
        save_file = save_dir + "/tfidf_vectorizer_max{}.{}.joblib".format(max_feature_num, tag)
        self.dump_tf_idf_model(tfidf, save_file)
        """

        # run tf-idf for product categorize
        complaints_narrative = pd.read_csv(self.preprocessed_file_for_product)
        processed_narratives = complaints_narrative["processed_narrative"]
        print("Vectorizing use tf-idf for product categorize...")
        tfidf, max_feature_num = self.tf_idf_vectorize(processed_narratives)

        save_dir = "trained_models"
        tag = "for_product_categorize"
        save_file = save_dir + "/tfidf_vectorizer_max{}.{}.joblib".format(max_feature_num, tag)
        self.dump_tf_idf_model(tfidf, save_file)

        print("Done!")



