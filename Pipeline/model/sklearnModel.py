import os
import pickle

import numpy as np

from model.baseModel import baseModel
from utils.modelUtils import parse_optional_param
from data.dataTransform import normalize_data


class sklearnModel(baseModel):

    def __init__(self):
        super().__init__()
        self.LOGGER = None
        self.model = None
        self.name = None
        self.model_type = None
        self.train_set = None
        self.feature_names = []

    def fit(self, data, **kwargs):
        """
        Fit scikit-learn model.
        :param data: list of pd.Dataframe: [training_features, training_labels] (contains the ID column)
        :param kwargs:
        """
        default_params = {'label_name': 'label', 'id_name': 'id'}
        params_dict = parse_optional_param(default_params, kwargs)

        features, label, feature_names = baseModel.split_feature_label(data,
                                                                       label_name=params_dict['label_name'],
                                                                       id_name=params_dict['id_name'])
        self.feature_names = feature_names
        if self.LOGGER:
            self.LOGGER.debug("fitting {} model on {} features".format(self.model_type, len(self.feature_names)))
        if self.model_type == "Logistic Regression":
            features = normalize_data(features)
        self.model.fit(features, label)
        self.LOGGER.debug("finished fitting model.")

    def predict(self, data, **kwargs):
        """
        Use trained scikit-learn  model to predict.
        :param data: pd.Dataframe: eval_features (contains the ID column)
        :param kwargs:
        :return: numpy array of scores : [ID, Scores]
        """
        default_params = {'label_name': 'label', 'id_name': 'id',
                          'base_margin': None, 'ntree_limit': None,
                          'validate_features': True}
        params_dict = parse_optional_param(default_params, kwargs)

        features, IDs, feature_names = baseModel.split_feature_label(data,
                                                                     id_name=params_dict['id_name'],
                                                                     label_name=params_dict['label_name'],
                                                                     eval=True)
        if self.model_type != "XGBClassifier":
            pred = self.model.predict_log_proba(features)
        else:
            pred = self.model.predict_proba(data=features,
                                            ntree_limit=params_dict['ntree_limit'],
                                            validate_features=params_dict['validate_features'],
                                            base_margin=params_dict['base_margin'])

        score = np.expand_dims(np.array([x[1] for x in pred]), 1)  # getting P(label = 1)
        score_with_ID = np.hstack([np.expand_dims(IDs.to_numpy(), 1), score])

        return score_with_ID

    def export_model(self, path, **kwargs):
        """
        Export scikit-learn  model as pickle
        :param path: base directory for outputs
        :param kwargs: additional optional parameters
        """
        model_path = os.path.join(path, self.name)  # e.g., experiments/DTModel_20201008111111
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # TODO: customizable name
        file_path = os.path.join(model_path, 'model_{}_{}.pkl'.format(self.train_set[0], self.train_set[1]))
        # e.g., experiments/DTModel_20201008111111/model.pkl
        pickle.dump(self.model, open(file_path, "wb"))
        if self.LOGGER:
            self.LOGGER.debug("finished fitting, saved {} model to file: {}".format(self.model_type, file_path))

    def load_model(self, path, **kwargs):
        """
        Load existing scikit-learn  model from path
        :param path: model save path
        :param kwargs: additional optional parameters
        """
        split_path = path.split("/")
        self.name = split_path[-2]
        # TODO: parse train set
        # self.train_set = split_path[-1].split("_")
        try:
            self.model = pickle.load(open(path, "rb"))
            if self.LOGGER:
                self.LOGGER.info("successfully loaded {} model from file:{}".format(self.model_type, path))
        except FileNotFoundError:
            if self.LOGGER:
                self.LOGGER.exception("failed to load model")
