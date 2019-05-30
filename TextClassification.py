from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
#from spacy.lang.en import stop_words
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def merge_stop_words():
    print(len(stopwords))
    #spacy_stopwords = stop_words.STOP_WORDS
    # print(len(spacy_stopwords))

def LDA(text):
    vect = CountVectorizer(max_features=10000, max_df=0.15)
    #vect = CountVectorizer(ngram_range=(1,3), min_df=5, stop_words="english")
    X = vect.fit_transform(text)
    #X = vect.fit_transform(text_train)

    lda = LatentDirichletAllocation(n_topics=5, learning_method="batch",
                                    max_iter=25, random_state=0)
    # We build the model and transform the data in one step
    # Computing transform makes some time,
    document_topics = lda.fit_transform(X)

    print(lda.components_.shape)

    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names())
    return lda, vect


def concatenate_text(complaints_narrative):
    text_train = "\n".join(complaints_narrative)
    return text_train

def main():
    """
    #LDA
    text_train = concatenate_text(complaints_narrative[0:1000])
    #LDA(text_train)
    LDA(complaints_narrative)

"""




