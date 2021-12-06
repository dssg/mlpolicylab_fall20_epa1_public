from model.sklearnModel import sklearnModel
from sklearn.linear_model import LogisticRegression
from utils.modelUtils import parse_optional_param
import numpy as np
import os
import pickle


class logisticRegressionModel(sklearnModel):
    def __init__(self, name='lr', logger=None, train_set=None, **kwargs):
        """
        Initialize LR model
        :param kwargs: see full list of parameters here:https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_log_proba
        """
        super().__init__()
        default_params = {'penalty': 'l2', 'tol': 1e-4, 'C': 1.0, 'fit_intercept': True,
                          'random_state': 0, 'max_iter': 100, 'verbose': 0,
                          'solver': 'lbfgs', 'l1_ratio': 0.0}
        params_dict = parse_optional_param(default_params, kwargs)
        self.name = name
        self.model_type = "Logistic Regression"
        self.LOGGER = logger
        self.train_set = train_set

        fit_intercept = params_dict['fit_intercept']
        if isinstance(fit_intercept, str):
            fit_intercept = eval(fit_intercept)

        penalty = params_dict['penalty']
        if penalty == 'l1' or penalty == 'l2':
            params_dict['solver'] = 'liblinear'
            self.model = LogisticRegression(penalty=params_dict['penalty'],
                                            tol=float(params_dict['tol']),
                                            C=params_dict['C'],
                                            fit_intercept=fit_intercept,
                                            random_state=params_dict['random_state'],
                                            max_iter=params_dict['max_iter'],
                                            verbose=params_dict['verbose'],
                                            solver=params_dict['solver'],
                                            n_jobs=12)
        elif penalty == 'elasticnet':
            params_dict['solver'] = 'saga'
            self.model = LogisticRegression(penalty=params_dict['penalty'],
                                            tol=float(params_dict['tol']),
                                            C=params_dict['C'],
                                            fit_intercept=fit_intercept,
                                            random_state=params_dict['random_state'],
                                            max_iter=params_dict['max_iter'],
                                            verbose=params_dict['verbose'],
                                            solver=params_dict['solver'],
                                            l1_ratio=float(params_dict['l1_ratio']),
                                            n_jobs=12)
        if self.LOGGER:
            self.LOGGER.debug("initialized LR model with param dict:{}".format(params_dict))
