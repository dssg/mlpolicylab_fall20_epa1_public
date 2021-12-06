## Models

#### File Structure
```
├── model/
│   ├── baseModel.py
│   ├── decisionTreeModel.py
│   ├── logisticRegressionModel.py
│   ├── randomForestModel.py
│   ├── sklearnModel.py
│   ├── xgbClassifier.py
│   └── xgboostModel.py
```

- `baseModel.py` : abstract class that defines the functionalities of a model 
- `sklearnModel.py`: base model for `scikit-learn` models
    - `decisionTreeModel.py` implements wrapper for decision tree classifier
    - `logisticRegressionModel.py` implements wrapper for logistic regression classifier
    - `randomForestModel.py` implements wrapper for random forests classifier
    - `xgbClassifier.py` implements wrapper for `xgboost` API for `scikit-learn`
- `xgboostModel.py` implements wrapper for `xgboost` model