import abc
import pandas as pd
import numpy as np


class baseModel(metaclass=abc.ABCMeta):

    def __init__(self):
        self.model_type = None
        self.model = None
        self.feature_names = []

    @staticmethod
    def split_feature_label(data, label_name='label', id_name='id', eval=False):
        """
        split entire data matrix into feature columns and label column
        :param label_name: name of label column (e.g., 'formal')
        :param id_name: name of ID column (e.g., 'handler_id')
        :param data: pd.DataFrame that contains a "label" column
        :param eval: if False get the label column; else return the ID column
        :return: features, label, and feature_names as separate things
        """
        if isinstance(data, pd.DataFrame):
            if not eval:
                label = np.array(data[label_name])
                features = data.drop(label_name, axis=1)
                feature_names = list(data.columns).remove(label_name)
            else:
                feature_names = list(data.columns)
                label = None
                if id_name in feature_names:
                    label = data[id_name]
                    features = data.drop(id_name, axis=1)
                    feature_names.remove(id_name)
        elif isinstance(data, list) or isinstance(data, tuple):
            assert len(data) == 2, "must contain feature and label df"
            features_with_ID, labels_with_ID = data
            feature_names = list(features_with_ID.columns)

            if id_name in feature_names:
                features = features_with_ID.drop(id_name, axis=1)
                feature_names.remove(id_name)
            label = labels_with_ID[label_name]

        else:
            raise NotImplementedError("Doesn't support this type of data for now!")

        return features, label, feature_names

    @abc.abstractmethod
    def fit(self, data, **kwargs):
        """
        abstract method for fitting the model with data
        :param data: training data
        :param kwargs: additional arguments
        """
        pass

    @abc.abstractmethod
    def predict(self, data, **kwargs):
        """
        abstract method for predicting on the given data
        :param data: data to eval
        :param kwargs: additional arguments
        """
        pass

    @abc.abstractmethod
    def export_model(self, path, **kwargs):
        """
        abstract method for exporting the model
        :param path: path to save model
        :param kwargs: additional arguments
        """
        pass

    @abc.abstractmethod
    def load_model(self, path, **kwargs):
        """
        abstract method for loading the model
        :param path: path to load model
        :param kwargs: additional arguments
        """
        pass

    @property
    def feature_importance(self):
        """
        Model feature importance and feature names
        :return: tuple feature importance (ordered) and feature names
        """
        if self.model_type in ["Decision Tree", "Random Forest"]:
            base_feature_importance = -self.model.feature_importances_
        elif self.model_type == "XGBClassifier":
            feat_importance_dict = self.model.get_booster().get_score(importance_type='gain')
            self.f_map = dict(list(zip(list(map(lambda x: 'f{}'.format(x),
                                                range(len(self.feature_names)))), self.feature_names)))
            base_feature_importance = [feat_importance_dict.get(f, 0.) for f in self.f_map]
            base_feature_importance = np.array(base_feature_importance, dtype=np.float32)
            base_feature_importance = -(base_feature_importance / base_feature_importance.sum())
        elif self.model_type == "xgboost":
            feat_importance_dict = self.model.get_score(importance_type='gain')
            self.f_map = dict(list(zip(list(map(lambda x: 'f{}'.format(x),
                                                range(len(self.feature_names)))), self.feature_names)))
            base_feature_importance = [feat_importance_dict.get(f, 0.) for f in self.f_map]
            base_feature_importance = np.array(base_feature_importance, dtype=np.float32)
            base_feature_importance = -(base_feature_importance / base_feature_importance.sum())
        elif self.model_type == 'Logistic Regression':
            base_feature_importance = self.model.coef_[0]
        else:
            raise NotImplementedError

        feature_importance_order = np.argsort(base_feature_importance)
        feature_importance = base_feature_importance[feature_importance_order]
        if self.model_type != 'Logistic Regression':
            feature_importance = -feature_importance
        feature_names = np.array(self.feature_names)[feature_importance_order]
        nonzero_idx = feature_importance != 0

        return np.array(feature_names)[nonzero_idx], feature_importance[nonzero_idx]

    def load_feature_names(self, data_path):
        """
        Load training feature names from training/validation data.
        :param data_path: path to data (csv file)
        :return:
        """
        try:
            data = pd.read_csv(data_path)
        except Exception:
            raise Exception
        feature_names = list(data.columns)[1:]
        feature_names.remove('handler_id')
        self.feature_names = feature_names
        print("successfully loaded {} feature names".format(len(self.feature_names)))
