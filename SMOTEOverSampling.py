from imblearn.over_sampling import SMOTE


def smote_over_sampling(X_train, y_train, random_state=0):
    """use SMOTE to oversampling X, y"""
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, y_train_res