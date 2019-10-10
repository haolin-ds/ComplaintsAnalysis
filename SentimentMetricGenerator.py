import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_complaints_data(complaints_file):
    complaints = pd.read_csv(complaints_file)

    print("There are {} complaints with narrative and label.".format(len(complaints)))

    return complaints


def num_of_question_mark(narrative):
    num_of_questionmark = 0
    for x in narrative:
        if x == '?':
            num_of_questionmark += 1
    return num_of_questionmark


def num_of_exclaimation_mark(narrative):
    num_of_exclaimationmark = 0
    for x in narrative:
        if x == '?' or x == '!':
            num_of_exclaimationmark += 1
    return num_of_exclaimationmark


def num_of_uppercase_word(narrative):
    words = narrative.split(" ")
    num_uppercase_word = 0
    for word in words:
        if word == "XXXX" or word == "XXXX," or word == "XXXX.":
            continue
        elif len(word) > 2 and word.isupper():
            print(word)
            num_uppercase_word += 1
    return num_uppercase_word


def transfer_label_column(label_column):
    label_column = label_column.apply(lambda x : 1 if x == "Yes" else 0)
    #print(label_column)

    #label_column = pd.Series(np.arange(10))
    return label_column


def generate_sentiment_metric(narratives):
    """
    Generate sentiment metrics for each narrative.
    :param narratives: a pandas column containing complaints narratives
    :return: a dataframe whose columns are several sentiment metrics
    [corpus_score_sum, corpus_score_ave, negative_ratio, most_negative_score,
    word_num, sentence_num, num_of_question_mark, num_of_exclaimation_mark]
    """

    corpus_score_list = []
    word_num_list = []
    sentence_num_list = []
    corpus_score_sum_list = []
    negative_ratio_list = []  # The ratio of sentences with negative score in the corpus
    most_negative_score_list = []
    num_of_question_mark_list = []
    num_of_exclaimation_mark_list = []

    #num_of_uppercase_word_list = []

    """Initialize Vader sentiment analyzer"""
    analyser = SentimentIntensityAnalyzer()

    X = pd.DataFrame()

    i = 0
    for narrative in narratives:
        i += 1
        if i % 1000 == 0:
            print(i)
        sentence_list = sent_tokenize(narrative)
        sentence_score_list = []
        word_num = 0
        copus_score_sum = 0
        negative_num = 0
        most_negative_score = 0

        """Generate sentiment score for each sentence in the narrative"""
        for sentence in sentence_list:
            sentiment_score_dict = analyser.polarity_scores(sentence)
            score = sentiment_score_dict["compound"]  # Use the compound score
            copus_score_sum += score
            word_num += len(word_tokenize(sentence))
            sentence_score_list.append(score)
            if score < -0.05:
                negative_num += 1
            if score < most_negative_score:
                most_negative_score = score

        corpus_score_list.append(sentence_score_list)
        sentence_num_list.append(len(sentence_score_list))
        word_num_list.append(word_num)
        corpus_score_sum_list.append(copus_score_sum)
        negative_ratio_list.append(negative_num / (len(sentence_score_list)))
        most_negative_score_list.append(most_negative_score)
        num_of_question_mark_list.append(num_of_question_mark(narrative))
        num_of_exclaimation_mark_list.append(num_of_exclaimation_mark(narrative))

        #num_of_uppercase_word_list.append(num_of_uppercase_word(narrative))

    # Will not use the list as feature
    # X["sentiment_score"] = corpus_score_list

    X["corpus_score_sum"] = corpus_score_sum_list
    X["corpus_score_ave"] = X["corpus_score_sum"] / sentence_num_list
    X["negative_ratio"] = negative_ratio_list
    X["most_negative_score"] = most_negative_score_list
    X["word_num"] = word_num_list
    X["sentence_num"] = sentence_num_list
    X["num_of_question_mark"] = num_of_question_mark_list
    X["num_of_exclaimation_mark"] = num_of_exclaimation_mark_list
    #X["num_of_uppercase_word"] = num_of_uppercase_word_list

    return X


def form_feature_data(complaints):
    """
    Combine some complaint information in the data
    :param complaints: complaints data frame
    :return: a dataframe containing sentiment metrics and [company_response, dispute, Complaint ID]
    """

    narratives = complaints["Consumer complaint narrative"]

    X = generate_sentiment_metric(narratives)

    # Add company response in
    X["company_response"] = complaints["Company response to consumer"].reset_index(drop=True)

    # Add the label in
    X["dispute"] = transfer_label_column(complaints["Consumer disputed?"]).reset_index(drop=True)

    # Add the complaint ID
    X["Complaint ID"] = complaints["Complaint ID"].reset_index(drop=True)

    return X


def dump_feature_to_csv(data, output_file):
    data.to_csv(output_file, index=False)


def get_complaints_with_sentiment(complaints, sentiment_metrics):
    complaints_with_sentiment = pd.merge(complaints.reset_index(drop=True),
                                   sentiment_metrics.reset_index(drop=True),
                                   how='inner',
                                   on=['Complaint ID', 'Complaint ID'])
    return complaints_with_sentiment


def main():
    complaints_file = "data/complaints-2019-05-16_13_17.clean.csv"
    complaints = load_complaints_data(complaints_file)

    print(complaints["Company response to consumer"].value_counts())

    X = form_feature_data(complaints)

    # Save complaint ID and sentiment_metrics
    output_file = "data/complaints.sentiment_metric.csv"
    dump_feature_to_csv(X, output_file)

    # Save complaints together with sentiment metrics
    complaints_with_sentiment = get_complaints_with_sentiment(complaints, X)
    output_file = "data/complaints_with_sentiment_metric.csv"
    dump_feature_to_csv(complaints_with_sentiment, output_file)


#main()


