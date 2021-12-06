import os

import numpy as np
import xgboost

from model.sklearnModel import sklearnModel
from utils.modelUtils import parse_optional_param


class xgbClassifier(sklearnModel):

    def __init__(self, name='xgbClassifier', logger=None, train_set=None, **kwargs):
        """

        :param log_base_dir: base directory for logging
        """
        super().__init__()
        self.name = name
        self.LOGGER = logger
        self.feature_names = []
        self.model_type = "XGBClassifier"
        self.train_set = train_set
        self.fmap = None

        default_params = {'n_estimators': 10, 'objective': 'reg:squarederror',
                          'learning_rate': 0.3, 'gamma': 0, 'max_depth': 6,
                          'min_child_weight': 1, 'sampling_method': 'uniform',
                          'subsample': 1, 'reg_lambda': 1, 'tree_method': 'auto',
                          'reg_alpha': 0.0, 'colsample_bytree': 1.0,
                          'colsample_bylevel': 1.0, 'colsample_bynode': 1.0,
                          'n_jobs': 1, 'random_state': 0, 'booster': 'gbtree'}

        params_dict = parse_optional_param(default_params, kwargs)
        self.model = xgboost.XGBClassifier(objective=params_dict['objective'],
                                           booster=params_dict['booster'],
                                           n_estimators=params_dict['n_estimators'],
                                           tree_method=params_dict['tree_method'],
                                           n_jobs=params_dict['n_jobs'],
                                           max_depth=params_dict['max_depth'],
                                           learning_rate=params_dict['learning_rate'],
                                           gamma=params_dict['gamma'],
                                           reg_alpha=params_dict['reg_alpha'],
                                           reg_lambda=params_dict['reg_lambda'],
                                           min_child_weight=params_dict['min_child_weight'],
                                           subsample=params_dict['subsample'],
                                           colsample_bytree=params_dict['colsample_bytree'],
                                           colsample_bylevel=params_dict['colsample_bylevel'],
                                           colsample_bynode=params_dict['colsample_bynode'],
                                           random_state=params_dict['random_state'])

        if self.LOGGER:
            self.LOGGER.debug("initialized XGBClassifier with param dict:{}".format(self.model.get_params()))

    def export_model(self, path, **kwargs):
        """
        :param path: base directory
        """

        model_path = os.path.join(path, self.name)  # e.g., experiments/DTModel_20201008111111
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # TODO: customizable name
        file_path = os.path.join(model_path, 'model_{}_{}.xbg'.format(self.train_set[0], self.train_set[1]))
        # e.g., experiments/DTModel_20201008111111/model.pkl
        self.model.save_model(file_path)
        if self.LOGGER:
            self.LOGGER.debug("finished fitting, saved {} model to file: {}".format(self.model_type, file_path))

    def load_model(self, path, **kwargs):
        """
        load existing model from path
        :param path: path to file (e.g. experiments/xgboostModel_20201008111111/model.save)
        :param kwargs:
        """
        split_path = path.split("/")
        self.name = split_path[-2]
        # TODO: parse train set
        # self.train_set = split_path[-1].split("_")
        try:
            self.model.load_model(path)
            if self.LOGGER:
                self.LOGGER.info("successfully loaded {} model from file:{}".format(self.model_type, path))
        except FileNotFoundError:
            if self.LOGGER:
                self.LOGGER.exception("failed to load model")
