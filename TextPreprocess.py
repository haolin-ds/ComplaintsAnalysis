from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
# from spacy.lang.en.stop_words import STOP_WORDS
import nltk


def merge_stop_word():
    """Merge stop words from three library"""
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
    Input: Parts-of-speech tag in Penn TreeBank format.
    Function: convert the first letter of Penn pos-tag to a form NLTK can use
    Output: NLTK-compatible pos-tag.
    """
    pos = {'N': 'n', 'V': 'v', 'J': 'a', 'S': 's', 'R': 'r'}
    if tag in pos.keys():
        return pos[tag]
    else:
        return 'n'  # everything else = noun.


def pre_process_narrative(narrative):
    # subtitute digits in the narrative into digits. There are not useful for text classification
    narrative = re.sub(r"\d+", "DIGITS", narrative)

    # Prepare stop-words list
    all_stopwords = merge_stop_word()

    # Tokenize the narrative
    tokens = [word.lower() for word in nltk.word_tokenize(narrative) if word.isalpha()]
    tokens_pos = nltk.pos_tag(tokens)

    # Lemmetize the tokens
    lmtzr = nltk.WordNetLemmatizer()
    tokens_lemmarized = [lmtzr.lemmatize(x[0], convert_pos_tag(x[1])) for x in tokens_pos]

    # Remove stop-words
    tokens_lemmarized_nostop = [token for token in tokens_lemmarized if
                                (len(token) > 2) and (token not in all_stopwords)]

    print(tokens_lemmarized_nostop)


def pre_process(complaints):
    narratives = complaints["Consumer complaint narrative"]
    cleaned_narratives = []

    for narrative in narratives:
        # Remove digits in the narrative. There are not useful for text classification
        narrative = re.sub(r"\d+", "", narrative)
        # Remove XXXX which is substitute by US govenment to protect privacy
        narrative = re.sub(r"XXXX", "", narrative)
        cleaned_narratives.append(narrative)

        pre_process_narrative(narrative)

    print(narratives.shape)
    print(len(cleaned_narratives))

    # complaints["cleaned_narrative"] = cleaned_narratives


