import pandas as pd


def clean_data():
    complaints = pd.read_csv("data/complaints-2019-05-16_13_17.csv")

    # Remove data with no dispute label
    complaints = complaints.dropna(subset=["Consumer disputed?"])

    # Remove data with company response "None", "Untimely response"
    complaints = complaints[complaints["Company response to consumer"] != "Untimely response"]
    complaints = complaints[complaints["Company response to consumer"] != "None"]

    # Remove data which belongs to "virtual money" (16)
    complaints = complaints[complaints["Product"] != "Virtual currency"]

    print("After cleaning, there are {} complaints".format(len(complaints)))
    complaints.to_csv("data/complaints-2019-05-16_13_17.clean.csv", index=False)



clean_data()


