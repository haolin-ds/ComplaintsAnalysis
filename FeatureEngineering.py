from SentimentMetricGenerator import load_complaints_data
from TextPreprocess import pre_process


def load_data():
    complaints = load_complaints_data("data/complaints-2019-05-16_13_17.csv")

    pre_process(complaints)

load_data()
