import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd

# Two flags to decide whether to keep stop word with emotion based on tasks
ESCALATION_PREDICTION = 0
PRODUCT_CATEGORIZE = 1


class TextPreprocesser:
    def __init__(self, complaint_file):
        self.complaint_file = complaint_file

    def extract_complaints_with_narrative(self, output_file):
        complaints = pd.read_csv(self.complaint_file)
        complaints_with_narratives = complaints.dropna(subset=["Consumer complaint narrative"])
        print("There are {} complaints with narrative.".format(len(complaints_with_narratives)))

        complaints_with_narratives.to_csv(output_file, index=False)

    def merge_stop_word(self, task_flag):
        """
        Merge stop words from two libraries and customized words.
        :param task_flag: a flag to denote the task type.  If the task is for escalation prediction,
                        stop_word should remove those containing sentiment, otherwise not.
        :return: a list of stop_words
        """

        all_stopwords = set(stopwords.words('english'))
        all_stopwords.union(STOP_WORDS)
        for word in ENGLISH_STOP_WORDS:
            all_stopwords.add(word)

        if task_flag == ESCALATION_PREDICTION:
            words_with_emotion = ['never', 'aren', 'neither', 'cannot', 'nobody', 'why', 'not', "don't", 'nor', 'whatever',
                                  "aren't", 'should', 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                                  "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                                  'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                                  "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "against"]

            for x in words_with_emotion:
                all_stopwords.remove(x)

        return list(all_stopwords)

    def convert_pos_tag(self, tag):
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

    def pre_process_one_narrative(self, narrative, task_flag):
        """
        Tokenize, lemmetize, remove stop_words from a complaint narrative
        :param narrative: one complaint narrative
        :param task_flag: The flag to denote task between escalation prediction or produce categorize
        :return: a list of pre-processed tokens
        """
        # substitute digits in the narrative into digits. They are not useful for text classification
        # narrative = re.sub(r"\d+", "DIGITS", narrative)
        narrative = re.sub(r"\d+", "", narrative)
        # Remove XXXX which is substitute by US govenment to protect privacy
        narrative = re.sub(r"XXXX", "", narrative)
        narrative = re.sub(r"XX", "", narrative)

        # Tokenize and Lemmetize using NLTK
        # Tokenize the narrative
        tokens = [word.lower() for word in nltk.word_tokenize(narrative) if word.isalpha()]
        tokens_pos = nltk.pos_tag(tokens)

        # Lemmetize the tokens
        lmtzr = nltk.WordNetLemmatizer()
        tokens_lemmatized = [lmtzr.lemmatize(x[0], self.convert_pos_tag(x[1])) for x in tokens_pos]

        """
        # Tokenize and Lemmatize using spaCy,  it's too slow!  
        sp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
        
        sentences = sp(narrative)
        tokens_lemmatized = []
        for word in sentences:
            # Remove Punctuation
            if word.lemma_ not in string.punctuation:
                # print(word.lemma_)
                tokens_lemmatized.append(word.lemma_)
        """

        # Prepare stop-words list
        all_stopwords = self.merge_stop_word(task_flag)
        min_word_len_thresh = 2

        # Remove stop-words
        tokens_lemmatized_nostop = [token for token in tokens_lemmatized if
                                    (len(token) > min_word_len_thresh) and (token not in all_stopwords)]
        return tokens_lemmatized_nostop

    def pre_process(self, complaints, task_flag):
        """
        Pre-process each narrative in complaints dataframe.
        :param complaints: a dataframe containing a column of narratives
        :param task_flag: the task type [ESCALATION_PREDICTION, PRODUCT_CATEGRIZE]
        :return: complaints with a new column "processed narrative" storing a list of tokens
        """
        narratives = complaints["Consumer complaint narrative"]
        processed_narratives = []

        i = 0
        for narrative in narratives:
            if i % 1000 == 0:
                print("Pre processing the {}th complaint narrative!".format(i))

            # Pre-process each narrative
            processed_narratives.append(self.pre_process_one_narrative(narrative, task_flag))
            i += 1

        complaints["processed_narrative"] = processed_narratives

    def export_processed_narratives(self, complaints_narrative, output_file):
        """
        Export complaints with processed_narrative in a file "data/narrative_preprocessed.csv" file with four columns
        [Complaint ID,Consumer complaint narrative,Consumer disputed?,processed_narrative]
        """
        complaints_narrative.to_csv(output_file, index=False)

    def text_preprocess(self, preprocessed_file_for_escalation, preprocessed_file_for_product):
        """Pre-process all complaint narratives. Tokenize, Remove stop words, Lemmatization."""

        # complaints = pd.read_csv("data/complaints-2019-05-16_13_17.clean.csv")
        complaints = pd.read_csv("data/complaints-2019-05-16_13_17.with_narrative.csv")
        complaints_narrative = complaints.loc[:, ["Complaint ID", "Consumer complaint narrative", "Consumer disputed?"]]

        task_flag = ESCALATION_PREDICTION
        # Load in complaint file and generate a new file containing preprocessed narratives
        self.pre_process(complaints_narrative, task_flag)
        self.export_processed_narratives(complaints_narrative, preprocessed_file_for_escalation)

        task_flag = PRODUCT_CATEGORIZE
        self.pre_process(complaints_narrative, task_flag)
        self.export_processed_narratives(complaints_narrative, preprocessed_file_for_product)
