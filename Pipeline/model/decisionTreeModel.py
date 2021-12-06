from model.sklearnModel import sklearnModel
from sklearn.tree import DecisionTreeClassifier
from utils.modelUtils import parse_optional_param


class decisionTreeModel(sklearnModel):
    def __init__(self, name='dt', logger=None, train_set=None, **kwargs):
        super().__init__()
        default_params = {'criterion': 'gini', 'splitter': 'best', 'max_depth': None,
                          'min_samples_split': 2, 'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0, 'max_features': None,
                          'random_state': None, 'max_leaf_nodes': None,
                          'min_impurity_decrease': None, 'class_weight': None,
                          'ccp_alpha': 0.0}
        params_dict = parse_optional_param(default_params, kwargs)
        self.name = name
        self.model_type = "Decision Tree"
        self.LOGGER = logger
        self.train_set = train_set
        self.model = DecisionTreeClassifier(criterion=params_dict['criterion'],
                                            splitter=params_dict['splitter'],
                                            max_depth=params_dict['max_depth'],
                                            min_samples_split=params_dict['min_samples_split'],
                                            min_samples_leaf=params_dict['min_samples_leaf'],
                                            min_weight_fraction_leaf=params_dict['min_weight_fraction_leaf'],
                                            max_features=params_dict['max_features'],
                                            random_state=params_dict['random_state'],
                                            max_leaf_nodes=params_dict['max_leaf_nodes'],
                                            min_impurity_decrease=params_dict['min_impurity_decrease'],
                                            class_weight=params_dict['class_weight'])
        if self.LOGGER:
            self.LOGGER.debug("initialized Decision Tree model with param dict:{}".format(self.model.get_params()))
