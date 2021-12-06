import os

import numpy as np
import xgboost

from model.baseModel import baseModel
from utils.modelUtils import parse_optional_param


class xgboostModel(baseModel):

    def __init__(self, name='xbg', logger=None, train_set=None, **kwargs):
        """
        :param log_base_dir: base directory for logging
        """
        super().__init__()
        self.name = name
        self.LOGGER = logger
        self.model = xgboost.Booster()
        self.feature_names = []
        self.model_type = "xgboost"
        self.train_set = train_set

        default_params = {'num_boost_round': 10, 'save_mode': 'save', 'objective': 'reg:squarederror',
                          'eta': 0.3, 'gamma': 0, 'max_depth': 6,
                          'min_child_weight': 1, 'sampling_method': 'uniform',
                          'subsample': 1, 'lambda': 1, 'tree_method': 'auto',
                          'alpha': 0.0, 'colsample_bytree': 1.0,
                          'colsample_bylevel': 1.0, 'colsample_bynode': 1.0,
                          'nthread': 12, 'seed': 0}

        params_dict = parse_optional_param(default_params, kwargs)
        self.num_boost_round = params_dict.pop('num_boost_round')
        self.save_mode = params_dict.pop('save_mode')
        self.training_params = params_dict

    def fit(self, data, **kwargs):
        """
        train Booster for some iterations (no validation)
        :param data: pd.DataFrame that contains a "label" column (or tuple of [training_data, label])
        :param params (dict): Booster params (see https://xgboost.readthedocs.io/en/latest/parameter.html)
        :param num_boost_round (int): Number of boosting iterations.
        """
        # processing input data

        default_params = {'label_name': 'label', 'id_name': 'id'}
        params_dict = parse_optional_param(default_params, kwargs)
        features, label, feature_names = baseModel.split_feature_label(data,
                                                                       label_name=params_dict['label_name'],
                                                                       id_name=params_dict['id_name'])
        self.feature_names = feature_names

        if self.feature_names and self.LOGGER:
            self.LOGGER.debug("fitting {} model on {} features".format(self.model_type, len(self.feature_names)))

        trainDMatrix = xgboost.DMatrix(data=features,
                                       label=label,
                                       feature_names=self.feature_names,
                                       nthread=2)

        # training
        if self.LOGGER:
            self.LOGGER.debug("begin training with parameters: {}".format(self.training_params))

        self.model = xgboost.train(params=self.training_params,
                                   dtrain=trainDMatrix,
                                   num_boost_round=self.num_boost_round)

    def predict(self, data, **kwargs) -> np.ndarray:
        """
        generate prediction scores for given data
        :param data: pd.DataFrame (require same column orderings as training data)
        :param output_margin (bool) : Whether to output the raw untransformed margin value
        :param ntree_limit (int) : Limit number of trees in the prediction; defaults to 0 (use all trees)
        :param pred_contribs (bool) : When this is True the output will be a matrix of size (nsample, nfeats + 1)
                                      with each record indicating the feature contributions (SHAP values) for that
                                      prediction. The sum of all feature contributions is equal to the raw
                                untransformed margin value of the prediction. Note the final column is the bias term.
        :param approx_contribs (bool) : Approximate the contributions of each feature
        :param pred_interactions (bool) : When this is True the output will be a matrix of size
                                         (nsample, nfeats + 1, nfeats + 1) indicating the SHAP
                                         interaction values for each pair of features.
                                         The sum of each row (or column) of the interaction values equals the
                                         corresponding SHAP value (from pred_contribs), and the sum of the entire
                                         matrix equals the raw untransformed margin value of the prediction.
                                         Note the last row and column correspond to the bias term.
        :param validate_features (bool) : When this is True, validate that the Booster’s and data’s feature_names are
                                          identical. Otherwise, it is assumed that the feature_names are the same.
        : param training (bool) : Whether the prediction value is used for training. This can effect dart booster,
                                  which performs dropouts during training iterations.
        """
        # processing input params
        default_params = {'output_margin': False, 'ntree_limit': 0, 'pred_leaf': False,
                          'approx_contribs': False, 'pred_interactions': False,
                          'validate_features': True, 'training': False,
                          'label_name': 'label', 'id_name': 'id'}
        params_dict = parse_optional_param(default_params, kwargs)

        features, IDs, feature_names = baseModel.split_feature_label(data,
                                                                     id_name=params_dict['id_name'],
                                                                     label_name=params_dict['label_name'],
                                                                     eval=True)

        # processing input data
        predictDMatrix = xgboost.DMatrix(data=features,
                                         feature_names=feature_names,
                                         nthread=-1)

        # predicting
        pred = self.model.predict(data=predictDMatrix,
                                  output_margin=params_dict['output_margin'],
                                  ntree_limit=params_dict['ntree_limit'],
                                  pred_leaf=params_dict['pred_leaf'],
                                  approx_contribs=params_dict['approx_contribs'],
                                  pred_interactions=params_dict['pred_interactions'],
                                  validate_features=params_dict['validate_features'],
                                  training=params_dict['training'])

        score = np.expand_dims(np.array(pred), 1)
        score_with_ID = np.hstack([np.expand_dims(IDs.to_numpy(), 1), score])
        return score_with_ID

    def export_model(self, path, **kwargs):
        """
        save_model: Save the model to a file.
        dump_model: Dump model into a text or JSON file. Unlike save_model, the output format is primarily used
        for visualization or interpretation, hence it’s more human readable but cannot
        be loaded back to XGBoost.
        :param path: base directory to export model
        :param save_mode: method of saving (save/dump). See XGBoost documentation.
        :param fmap: (for dump_model, string) Name of the file containing feature map names.
        :param with_stats: (for dump_model, bool) Controls whether the split statistics are output.
        :param dump_format: (for dump_model, ) Format of model dump file. Can be ‘text’ or ‘json’.
        """

        model_path = os.path.join(path, self.name)  # e.g., experiments/xgboostModel_20201008111111
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if self.save_mode == 'dump':
            default_params = {'fmap': 'fmap', 'with_stats': False, 'dump_format': 'text'}
            params_dict = parse_optional_param(default_params, kwargs)

            file_extension = 'txt' if params_dict['dump_format'] == 'text' else 'json'
            file_path = os.path.join(model_path, 'model.' + file_extension)
            fmap_name = os.path.join(model_path, params_dict['fmap'] + '.' + file_extension)
            # e.g., saved_model/xgboostModel_20201008111111/model.txt and fmap.txt

            self.model.dump_model(fout=file_path,
                                  fmap=fmap_name,
                                  with_stats=params_dict['with_stats'],
                                  dump_format=params_dict['dump_format'])
            if self.LOGGER:
                self.LOGGER.debug("dumped model to file:{}".format(file_path))

        elif self.save_mode == 'save':
            file_path = os.path.join(model_path, 'xgboost_{}_{}.model'.format(self.train_set[0], self.train_set[1]))
            # e.g., experiments/xgboostModel_20201008111111/model.save
            self.model.save_model(fname=file_path)
            if self.LOGGER:
                self.LOGGER.debug("saved model to file:{}".format(file_path))
        else:
            if self.LOGGER:
                self.LOGGER.error("save_mode must be dump or save!")
            raise NotImplementedError('save_mode must be dump or save!')

    def load_model(self, path, **kwargs):
        """
        load existing model from path
        :param path: path to file (e.g. experiments/xgboostModel_20201008111111/model.save)
        :param kwargs:
        """
        dir_split = path.split("/")
        try:
            assert '.txt' not in dir_split[-1] and '.json' not in dir_split[-1]
            self.model.load_model(fname=path)
            self.name = dir_split[-2]
        except AssertionError:
            if self.LOGGER:
                self.LOGGER.exception('cannot load dumped model!')
        except FileNotFoundError:
            if self.LOGGER:
                self.LOGGER.exception('failed to load model')