from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def smote_over_sampling(X_train, y_train, random_state=0):
    """use SMOTE to oversampling X, y"""
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, y_train_res


def under_sampling(X_train, y_train):
    sampler = RandomUnderSampler(sampling_strategy='majority',
                                 random_state=0)
    X_train_under, y_train_under = sampler.fit_sample(X_train, y_train)
    return X_train_under, y_train_under