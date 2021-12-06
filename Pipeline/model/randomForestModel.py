from model.sklearnModel import sklearnModel
from sklearn.ensemble import RandomForestClassifier
from utils.modelUtils import parse_optional_param


class randomForestModel(sklearnModel):
    def __init__(self, name='rf', logger=None, train_set=None, **kwargs):
        super().__init__()
        default_params = {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None,
                          'min_samples_split': 2, 'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0, 'max_features': None,
                          'random_state': None, 'max_leaf_nodes': None,
                          'min_impurity_decrease': None, 'class_weight': None,
                          'bootstrap': True, 'max_samples': None, 'verbose': 0, 'n_jobs': 16}
        params_dict = parse_optional_param(default_params, kwargs)
        self.name = name
        self.model_type = "Random Forest"
        self.LOGGER = logger
        self.train_set = train_set

        bootstrap = params_dict['bootstrap']
        if isinstance(bootstrap, str):
            bootstrap = eval(bootstrap)

        self.model = RandomForestClassifier(n_estimators=params_dict['n_estimators'],
                                            criterion=params_dict['criterion'],
                                            max_depth=params_dict['max_depth'],
                                            min_samples_split=params_dict['min_samples_split'],
                                            min_samples_leaf=params_dict['min_samples_leaf'],
                                            min_weight_fraction_leaf=params_dict['min_weight_fraction_leaf'],
                                            max_features=params_dict['max_features'],
                                            random_state=params_dict['random_state'],
                                            max_leaf_nodes=params_dict['max_leaf_nodes'],
                                            min_impurity_decrease=params_dict['min_impurity_decrease'],
                                            class_weight=params_dict['class_weight'],
                                            bootstrap=bootstrap,
                                            verbose=params_dict['verbose'],
                                            n_jobs=params_dict['n_jobs'])
        if self.LOGGER:
            self.LOGGER.debug("initialized Decision Tree model with param dict:{}".format(self.model.get_params()))

