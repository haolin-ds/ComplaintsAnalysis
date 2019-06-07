import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# from spacy.lang.en.stop_words import STOP_WORDS
import nltk
import pandas as pd
import scipy
from joblib import dump


def merge_stop_word():
    """
    Merge stop words from two libraries and customized words.
    :return: a list of stop_words
    """

    all_stopwords = set(stopwords.words('english'))
    # all_stopwords.add(STOP_WORDS)
    for word in ENGLISH_STOP_WORDS:
        all_stopwords.add(word)

    words_with_emotion = ['never', 'aren', 'neither', 'cannot', 'nobody', 'why', 'not', "don't", 'nor', 'whatever',
                          "aren't", 'should', 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                          "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                          'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                          "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    for x in words_with_emotion:
        all_stopwords.remove(x)

    return list(all_stopwords)


def convert_pos_tag(tag):
    """
    convert the first letter of Penn pos-tag to a form NLTK can use
    :param tag: Parts-of-speech tag in Penn TreeBank format.
    :return: NLTK-compatible pos-tag.
    """

    pos = {'N': 'n', 'V': 'v', 'J': 'a', 'S': 's', 'R': 'r'}
    if tag in pos.keys():
        return pos[tag]
    else:
        return 'n'  # everything else = noun.


def pre_process_narrative(narrative):
    """
    Tokenize, lemmetize, remove stop_words from a complaint narrative
    :param narrative: one complaint narrative
    :return: a list of pre-processed tokens
    """
    # substitute digits in the narrative into digits. There are not useful for text classification
    # narrative = re.sub(r"\d+", "DIGITS", narrative)
    narrative = re.sub(r"\d+", "", narrative)
    narrative = re.sub(r"XXXX", "", narrative)

    # Prepare stop-words list
    all_stopwords = merge_stop_word()

    # Tokenize the narrative
    tokens = [word.lower() for word in nltk.word_tokenize(narrative) if word.isalpha()]
    tokens_pos = nltk.pos_tag(tokens)

    # Lemmetize the tokens
    lmtzr = nltk.WordNetLemmatizer()
    tokens_lemmarized = [lmtzr.lemmatize(x[0], convert_pos_tag(x[1])) for x in tokens_pos]

    min_word_len_thresh = 2
    # Remove stop-words
    tokens_lemmarized_nostop = [token for token in tokens_lemmarized if
                                (len(token) > min_word_len_thresh) and (token not in all_stopwords)]

    return tokens_lemmarized_nostop


# TODO seems no use
def pre_process(complaints):
    """
    Pre-process each narrative in complaints dataframe.
    :param complaints: a dataframe containing a column of narratives
    :return: complaints with a new column "processed narrative" storing a list of tokens
    """
    narratives = complaints["Consumer complaint narrative"]
    processed_narratives = []

    i = 0
    for narrative in narratives:
        if i % 1000 == 0:
            print("Pre processing the {}th complaint narrative!".format(i))

        # Remove digits in the narrative. There are not useful for text classification
        narrative = re.sub(r"\d+", "", narrative)

        # Remove XXXX which is substitute by US govenment to protect privacy
        narrative = re.sub(r"XXXX", "", narrative)
        narrative = re.sub(r"XX", "", narrative)

        # Pre-process each narrative
        processed_narratives.append(pre_process_narrative(narrative))
        i += 1

    complaints["processed_narrative"] = processed_narratives


def export_processed_narratives(complaints_narrative, output_file):
    """
    Export complaints with processed_narrative in a file "data/narrative_preprocessed.csv" file with four columns
    [Complaint ID,Consumer complaint narrative,Consumer disputed?,processed_narrative]
    """
    complaints_narrative.to_csv(output_file, index=False)


def tf_idf_vectorize(pre_processed_narratives, min_df=5):
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

    narratives_vectorized = tf_idf_vectorizer.fit_transform(pre_processed_narratives)
    print("tf-idf vector shape: {}".format(narratives_vectorized.shape))

    return tf_idf_vectorizer, narratives_vectorized, max_feature_num


def dump_tf_idf_model(tf_idf_vectorizer, max_feature_num, save_dir, tag):
    save_file = save_dir + "/tfidf_vectorizer_max{}.{}.joblib".format(max_feature_num, tag)
    print("Saving tf-idf model to file " + save_file)
    # dump tf-idf vectorizer to file
    dump(tf_idf_vectorizer,
                open(save_file, "wb"))


def get_tf_idf_vector(tf_idf_vectorizer, narrative):
    """
    Given a new narrative, pre-process it and vectorize it using
    pretrained tf_idf_vectorizer
    :param tf_idf_vectorizer: pre-trained tf-idf vectorizer
    :param narrative:
    :return: narrative_vectorized
    """

    pre_processed_narrative = pre_process_narrative(narrative)
    return tf_idf_vectorizer.transform(pre_processed_narrative)


def generate_tf_idf_model(processed_narratives, tag):
    save_dir = "trained_models"
    tfidf, narrative_vectorized, max_feature_num = tf_idf_vectorize(processed_narratives)
    dump_tf_idf_model(tfidf, max_feature_num, save_dir, tag)


def text_preprocess():
    # Load in complaints and keep only those contain Labels
    complaints = pd.read_csv("data/complaints-2019-05-16_13_17.clean.csv")
    complaints_narrative = complaints.loc[:, ["Complaint ID", "Consumer complaint narrative", "Consumer disputed?"]]

    # Load in complaint file and generate a new file containing preprocessed narratives
    pre_process(complaints_narrative)
    output_file = "data/narrative_preprocessed.csv"
    export_processed_narratives(complaints_narrative, output_file)


def run_tf_idf():
    # text_preprocess()

    print("Loading complaints with processed narrative...")
    complaints_narrative = pd.read_csv("data/narrative_preprocessed.csv", index=False)
    # run tf-idf on all complaints and dump them
    processed_narratives = complaints_narrative["processed_narrative"]
    print("Vectorizing use tf-idf...")
    tag = "for_product_classifier"
    generate_tf_idf_model(processed_narratives, tag)
    print("Done!")


#text_preprocess()