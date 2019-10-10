import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline

from ImbalancedDataSampling import smote_over_sampling
from Utilities import load_model, save_model, draw_roc_curve

LOGISTIC_REGRESSION_L2 = 0
RANDOM_FOREST = 1
GRADIENT_BOOSTING = 2
LOGISTIC_REGRESSION_L1 = 3

MODEL_NAMES = ["lgreg_L2", "rf", "grb", "lgreg_L1"]


class EscalationClassifier():
    def __init__(self, tf_idf_model_file, complaints_with_sentiment_file, narrative_preprocessed_file):
        print("Loading tf_idf_model...")
        self.tf_idf_vectorizer = load_model(tf_idf_model_file)
        self.complaints_with_sentiment_file = complaints_with_sentiment_file
        self.narrative_preprocessed_file = narrative_preprocessed_file

    def feature_engineering(self):
        """Generate features by combining vectorized narratives, company response and sentiment metrics.
        1. Load the narrative and sentiment metrics csv file.
        2. Merge them according to shared complaint ID.
        3. Hot code Company response.
        4. Vectorize narratives by tf-idf vectorizer.
        :Output X, y.  X is features. y is labels whether dispute (1) or not (0).
        """
        # Load the complaints with sentiment metrics file
        complaints = pd.read_csv(self.complaints_with_sentiment_file)
        complaints = complaints[
            ["Complaint ID", "Timely response?", "Product", "corpus_score_sum", "corpus_score_ave", "negative_ratio",
             "most_negative_score", "word_num", "sentence_num", "num_of_question_mark", "num_of_exclaimation_mark",
             "company_response"]]
        print("Loaded {} complaints with sentiment metrics.".format(len(complaints)))

        # Load the preprocessed narratives
        narrative_processed = pd.read_csv(self.narrative_preprocessed_file)
        print("Loaded {} preprocessed narratives.".format(len(narrative_processed)))

        # One hot coding the column "company_response", store the column name of dummy columns
        old_column_num = len(complaints.columns)
        complaints = pd.get_dummies(complaints, columns=["company_response"])
        new_column_names = []
        for i in np.arange(old_column_num - 1, len(complaints.columns)):
            new_column_names.append(complaints.columns[i])

        # Save the new column names to a file for prediction
        column_name_file = "trained_models/company_corresponse_variable_names.csv"
        with open(column_name_file, "w") as fobj:
            fobj.write(",".join(new_column_names))

        # Merge the sentiment metrics and narratives according to their Complaint ID.
        # Those narratives without label do not have sentiment metrics.  These records will be discarded.
        complaints_features = pd.merge(complaints.reset_index(drop=True), narrative_processed.reset_index(drop=True),
                                       how='inner', on=['Complaint ID', 'Complaint ID'])

        print("Loaded {} complaints ".format(complaints_features.shape))

        # Vectorize narratives
        not_to_vectorized_columns = complaints_features.loc[:, "corpus_score_sum":"company_response_Closed with non-monetary relief"]
        print("Tf-idf vectorizing narratives processed for escalation classifier...")
        vectorized_columns = self.tf_idf_vectorizer.transform(complaints_features["processed_narrative"])
        print(vectorized_columns.shape)

        X = hstack((vectorized_columns, np.array(not_to_vectorized_columns)))
        y = [0 if x == "No" else 1 for x in complaints_features["Consumer disputed?"]]

        column_names = self.tf_idf_vectorizer.get_feature_names()
        column_names.extend(not_to_vectorized_columns.columns)
        print(len(column_names))
        X.columns = column_names

        return X, y

    def build_classifier(self, model_flag, X, y):
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=42)

        X_trainval.columns = X.columns

        if model_flag == RANDOM_FOREST:
            is_rf = True
        else:
            is_rf = False

        parameter_grid = self.get_parameter_list(model_flag)
        best_parameter = self.select_model_best_parameters(model_flag, X_trainval, y_trainval, parameter_grid, is_rf)

        # Applying the model with best parameters on all training data and evaluate the model on test data
        X_trainval, X_test = self.scale_features(X_trainval, X_test)
        X_trainval_res, y_trainval_res = smote_over_sampling(X_trainval, y_trainval)
        model = self.set_model(model_flag, best_parameter)
        model_auc, model_precision_recall, preds = self.fit_classifier_model(model,
                                                                             X_trainval_res,
                                                                             X_test,
                                                                             y_trainval_res,
                                                                             y_test,
                                                                             is_rf)
        print("The auc score of the model is {}, precision_recall: {} ".format(model_auc, model_precision_recall))
        fpr, tpr, thresholds = roc_curve(y_test, preds)
        title = "ROC curve for escalate classifier (AUC={:.3f})".format(model_auc)
        save_file = "figs/roc_escalation_classifier_" + MODEL_NAMES[model_flag] + ".png"
        draw_roc_curve(title, save_file, [fpr], [tpr], [model_auc], MODEL_NAMES[model_flag])

        return model, best_parameter

    def fit_classifier_model(self, model, X_train, X_test, y_train, y_test, is_rf):
        """Fit the classifier model based on the data and params. Return the auc score as metric"""
        print(model)
        model.fit(X_train, y_train)

        if is_rf == True:
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.decision_function(X_test)

        model_auc = roc_auc_score(y_test, preds)
        model_precision_recall = average_precision_score(y_test, preds)

        print("auc: {}, precision_recall: {}".format(model_auc, model_precision_recall))

        return model_auc, model_precision_recall, preds

    def select_model_best_parameters(self, model_flag, X_trainval, y_trainval, parameter_list, is_rf):
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=42)

        # Scale sentiment metric features
        X_train, X_val = self.scale_features(X_train, X_val)

        # Oversampling
        X_train_res, y_train_res = smote_over_sampling(X_train, y_train)

        best_parameter = None
        best_auc = 0

        for param_val in parameter_list:
            model = self.set_model(model_flag, param_val)
            model_auc, model_precision_recall, _ = self.fit_classifier_model(model,
                                                                             X_train_res,
                                                                             X_val,
                                                                             y_train_res,
                                                                             y_val,
                                                                             is_rf)
            if model_auc > best_auc:
                best_auc = model_auc
                best_parameter = param_val

        return best_parameter

    def get_parameter_list(self, model_flag):
        if model_flag == LOGISTIC_REGRESSION_L1 or model_flag == LOGISTIC_REGRESSION_L2:
            parameter_grid = [0.01, 0.1, 1, 10, 100]
        elif model_flag == RANDOM_FOREST:
            parameter_grid = [50, 100, 150]
        elif model_flag == GRADIENT_BOOSTING:
            parameter_grid = [1, 2, 3]
        return parameter_grid

    def set_model(self, model_flag, param_val):
        """According to model_flag, choose the right model and parameter set"""
        if model_flag == LOGISTIC_REGRESSION_L2:
            return LogisticRegression(C=param_val, penalty='l2', solver='lbfgs', max_iter=2000)
        elif model_flag == LOGISTIC_REGRESSION_L1:
            return LogisticRegression(C=param_val, penalty='l1', solver='lbfgs', max_iter=2000)
        elif model_flag == RANDOM_FOREST:
            return RandomForestClassifier(n_estimators=param_val, random_state=0)
        elif model_flag == GRADIENT_BOOSTING:
            return GradientBoostingClassifier(max_depth=param_val, random_state=0)
        else:
            print("No valid model selected.")
            return None

    def scale_features(self, X_train, X_test):
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """
        scaler = MinMaxScaler()
        print(X_train.columns)
        scaler.fit(X_train.loc[:, ["word_num", "sentence_num"]])

        X_train.loc[:, ["word_num", "sentence_num"]] = scaler.transform(X_train.loc[:, ["word_num", "sentence_num"]])
        X_test.loc[:, ["word_num", "sentence_num"]] = scaler.transform(X_test.loc[:, ["word_num", "sentence_num"]])
        """

        return X_train, X_test

    def escalation_classifier(self, model_flag):
        print("Feature engineering...")
        X, y = self.feature_engineering()

        print("Building the escalation classifier...")
        escalation_classifier, best_parameter = self.build_classifier(model_flag, X, y)

        print("Saving the product_classifier...")
        model_name = MODEL_NAMES[model_flag]
        model_export_file = "trained_models/escalation_classifier_" + model_name + "." \
                            + str(best_parameter) + ".sav"
        save_model(escalation_classifier, model_export_file)

