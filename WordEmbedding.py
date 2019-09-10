import pandas as pd
import numpy as np


def extract_complaints_with_narrative():
    complaints = pd.read_csv("data/complaints-2019-05-16_13_17.csv")
    complaints_with_narratives = complaints.dropna(subset=["Consumer complaint narrative"])
    print("There are {} complaints with narrative.".format(len(complaints_with_narratives)))

    output_file = "data/complaints-2019-05-16_13_17.with_narrative.csv"
    complaints_with_narratives.to_csv(output_file, index=False)

