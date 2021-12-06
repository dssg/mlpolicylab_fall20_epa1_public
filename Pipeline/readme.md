# Model Pipeline

## Running the Pipeline

### Q: How to run a specific model config file?

_New in the latest version_: you would have to specify the model config file, otherwise the we would use `experiment_config/sampleConfig.yaml` for training!

Simply run 

```
python execute.py --mc experiment_config/[CONFIG FILE NAME]
```

where `--mc`/`-model_config` flag is for the directory of the file that contains the model config file in `yaml`


### Q: How to create model config file from model training outputs?

To automatically generate the model config file, simply run

```
python utils/modelUtils.py --mp experiments/grid_autogen_201115_1504/with_grid_result.csv --co experiment_config/[FILE NAME] --save [SAVE OPTIONS]
```

where 
  - `--mp`/`-model_param` flag is for the directory of the file that contains the model specifications. This could be the default output file in `.csv` file, or a `.csv` or `.txt` file created by you (note that it **must** contains the columnms `model_type` and `model_params`)
  - `--co`/`-config_out` flag is for the directory of the output config file
  - `--save`/`-save_model` flag is used to control the saving behavior of generated model config file (if `true`, then every model trained using this config would be saved).
  - `--prk`/`--plot_prk` flag is used to control the plotting behavior (if `true`, then plot the precision-recall-k curve for all validation sets)
  - `--save_valid`/`-save_valid` flag is used to control whether to save the validation dataset or not (can be useful if you want to look at the cross-tabs) 
  - `--k`, `-topk` is used to specify the population percent k for the evaluator (takes two arguments, the first for with history cohort, the second for without history cohort; default is 0.05 for both.)

  
##### Example `model_param` file format

```
model_type,model_param
xgboost,'objective=binary:logitraw;num_boost_round=10;eta=0.1;gamma=0;sampling_method=uniform'
xgboost,'objective=binary:logitraw;num_boost_round=10;eta=0.1;gamma=0;sampling_method=uniform'
```
where `model_param` is a string (wrapped between single quotation marks) where the hyperparameters are separated by `;` and the values are specified by `=`.

