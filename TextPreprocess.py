from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from spacy.lang.en import stop_words


def merge_stop_word():
    print(type(stop_words))
    print(type(stopwords))
    print(type(ENGLISH_STOP_WORDS))
    # nltk_stop_words = Set(stop_words)


def pre_process_narrative(narrative):
    # subtitute digits in the narrative into digits. There are not useful for text classification
    narrative = re.sub(r"\d+", "DIGITS", narrative)

    # remove stop-words


    # Tokenize the narrative




def pre_process_narrative(complaints):
    narratives = complaints["Consumer complaint narrative"]
    cleaned_narratives = []

    for narrative in narratives:
        # Remove digits in the narrative. There are not useful for text classification
        narrative = re.sub(r"\d+", "", narrative)
        # Remove XXXX which is substitute by US govenment to protect privacy
        narrative = re.sub(r"XXXX", "", narrative)
        cleaned_narratives.append(narrative)

    print(complaints.shape)
    print(len(cleaned_narratives))

    complaints["cleaned_narrative"] = cleaned_narratives

merge_stop_word()