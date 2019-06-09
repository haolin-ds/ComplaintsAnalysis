from joblib import dump, load
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler

VALIDATION_SIZE = 1000

PRODUCT_LABELS = ['Bank account or service',
                  'Credit reporting',
                  'Mortgage',
                  'Debt collection',
                  'Credit card',
                  'Student loan',
                  'Payday loan',
                  'Consumer Loan',
                  'Money transfers',
                  'Prepaid card',
                  'Other financial service']


def save_model(model, output_file):
    # save the model as final_model
    dump(model, open(output_file, "wb"))


def load_model(saved_model_file):
    with open(saved_model_file, 'rb') as f:
        temp_model = load(f)
    return temp_model


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train.loc[:, ["word_num", "sentence_num"]])

    save_model(scaler, "trained_models/scaler.joblib")

    X_train.loc[:, ["word_num", "sentence_num"]] = scaler.transform(X_train.loc[:, ["word_num", "sentence_num"]])
    X_test.loc[:, ["word_num", "sentence_num"]] = scaler.transform(X_test.loc[:, ["word_num", "sentence_num"]])

    return X_train, X_test


def get_response_types():
    response_column_names_file = "trained_models/company_corresponse_variable_names.csv"

    with open(response_column_names_file, "r") as fobj:
        line = fobj.readline()
        response_types = line.rstrip().split(",")
        chopped_response_types = []
        for response in response_types:
            response = response.split("_")[-1]
            response = re.sub(r"Closed with ", "", response).capitalize()
            chopped_response_types.append(response)

    return chopped_response_types



def load_models(clf_product_file, clf_escalation_file, tf_idf_vectorizer_file, scaler_file):
    clf_product = load_model(clf_product_file)
    clf_escalation = load_model(clf_escalation_file)
    tf_idf_vectorizer = load_model(tf_idf_vectorizer_file)
    scaler = load_model(scaler_file)

    return clf_product, clf_escalation, tf_idf_vectorizer, scaler


def draw_roc_curve(title, save_file, fpr_list, tpr_list, roc_auc_list, label_name_list, draw_micro=False):
    """
    Draw mutiple roc_curve in one figure.
    :param title:
    :param save_file:
    :param fpr_list:
    :param tpr_list:
    :param roc_auc_list:
    :param label_name_list: The label name of each roc curve
    :param draw_micro: True when it's results from multi-class classifier
    :return:
    """
    # Plot all ROC curves
    plt.figure(figsize=(9, 8))

    n_classes = len(label_name_list)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lw = 2 # line width

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr_list[i], tpr_list[i], color=color, lw = lw,
                 label='{0} (area = {1:0.2f})'
                       ''.format(label_name_list[i], roc_auc_list[i]))

    if draw_micro:
        plt.plot(fpr_list["micro"], tpr_list["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_list["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], color='black', lw=lw, label="Chance", linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(save_file)
