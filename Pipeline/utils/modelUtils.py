import argparse
import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
import itertools

import yaml

SKLEARN_MODELS = ['lr', 'dt', 'rf', 'xgbClassifier']

MODELS_MAP = {'lr': 'logisticRegressionModel',
              'dt': 'decisionTreeModel',
              'rf': 'randomForestModel',
              'xgboost': 'xgboostModel',
              'xgbClassifier': 'xgbClassifier'}

DEFAULT_MODEL_CONFIG = "experiment_config/defaultConfig.yaml"


def str2bool(s):
    lowered = s.lower()
    first_cap = lowered[0].upper()
    return eval(first_cap + lowered[1:])


def parse_optional_param(default_params, kwargs_dict):
    """
    parse optional parameters. Use user input keyword argument's
    values if possible, otherwise use the default value
    :param default_params: dictionary of params with default values
    :param kwargs_dict: dictionary of input keyword arguments with values
    :return: dictionary with correct values for each keyword
    """
    result = dict()
    for key in default_params:
        if key in kwargs_dict:
            val = kwargs_dict[key]
            if val == "None":
                val = None
            result[key] = val
        else:
            result[key] = default_params[key]
    return result


def parse_hyperparams(model_config, direct_import):
    """
    Parse model params from config and build model grid
    :param direct_import: whether to directly import model grid or automatically compute grid
    :param model_config: model config for a model
    :return: list of model param dicts
    """

    if not direct_import:
        param_combo, base_param = list(), dict()
        iter_params, param_order = [], []
        for param in model_config:
            spec = model_config[param]
            if spec is None:
                result = None
            elif isinstance(spec, list):
                result = spec
            elif isinstance(spec, str):
                try:
                    result = eval(spec)
                except Exception:
                    raise Exception
            else:
                result = [spec]

            if result is None:  # the case when no hyperparam was specified
                continue
            elif len(result) == 1:  # only one value, add to the base param dict
                base_param[param] = result[0]
            else:
                iter_params.append(list(result))
                param_order.append(param)

        if len(iter_params) == 0:
            param_grid = []
        elif len(iter_params) == 1:
            param_grid = list(map(lambda x: [x], iter_params[0]))
        else:
            param_grid = itertools.product(iter_params[0], iter_params[1])
            for i in range(2, len(iter_params)):
                param_grid = itertools.product(param_grid, iter_params[i])
            param_grid = list(param_grid)

        num_iter_param = len(param_grid)
        if num_iter_param == 0:
            param_combo = [base_param]
        else:
            for i in range(len(param_grid)):
                base_param_copy = copy.copy(base_param)
                param_order_copy = copy.copy(param_order)
                param_name = param_order_copy.pop(-1)

                base_param_copy[param_name] = param_grid[i][-1]
                rest = param_grid[i][0]
                while len(param_order_copy) > 0:
                    param_name = param_order_copy.pop(-1)
                    if len(param_order_copy) == 0:
                        base_param_copy[param_name] = rest
                    else:
                        base_param_copy[param_name] = rest[-1]
                        rest = rest[0]
                param_combo.append(base_param_copy)
    else:
        param_combo = list()
        param_lst_len = None
        keys_to_pop = []
        for param in model_config:
            spec = model_config[param]
            if isinstance(spec, list):
                if not param_lst_len:
                    param_lst_len = len(spec)
                else:
                    assert param_lst_len == len(spec), "list of parameters must have same length"
            elif isinstance(spec, str):
                continue
            elif spec is None:
                keys_to_pop.append(param)
            else:
                raise Exception("bad input:{}".format(spec))

        for i in range(param_lst_len):
            param_dict = dict()
            for param in model_config:
                if param in keys_to_pop:
                    continue
                spec = model_config[param]
                if isinstance(spec, list):
                    param_dict[param] = spec[i]
                else:
                    param_dict[param] = spec
            param_combo.append(param_dict)
    return param_combo


def save_model_config(base_dir, which_model_config):
    """
    Write model param to text file for reference.
    :param base_dir: base saving directory
    :param which_model_config: model param dict
    :return:
    """
    file_path = os.path.join(base_dir, "model_param.txt")
    with open(file_path, "w") as f:
        f.write("{}".format(which_model_config))
    f.close()


def save_grid_search_result(base_dir, out_file, output):
    """
    Write grid search result to csv file.
    :param base_dir: base directory for the output file.
    :param out_file: output file name
    :param output: output from grid search routine
    :return: csv file with fields [model_type, model_name, precision_overtime, model_param]
             note that the values in precision_overtime and model_param is separated by ";"
             (use sep=";" to split and parse if needed)
    """
    out_file = os.path.join(base_dir, out_file)
    with open(out_file, "w") as f:
        header = "{},{},{},{},{}\n".format('model_type', 'model_name', 'precision_overtime',
                                           'support_overtime', 'model_param')
        f.write(header)
        for model_type in output:
            model_result = output[model_type]
            precision_overtime = model_result['precision_overtime']
            support_overtime = model_result['support_overtime']
            model_name = model_result['model_names']
            model_params = model_result['which_model_params']
            for i in range(len(model_name)):
                precision_overtime_i = list(map(lambda x: str(x), list(precision_overtime[i, :])))
                support_overtime_i = list(map(lambda x: str(x), list(support_overtime[i, :])))
                model_params_i = ["{}={}".format(param, model_params[i][param]) for param in model_params[i]]
                message = "{},{},{},{},'{}'\n".format(model_type, model_name[i],
                                                      ";".join(precision_overtime_i),
                                                      ";".join(support_overtime_i),
                                                      ";".join(model_params_i))
                f.write(message)
        f.close()


def generate_model_config(model_param_filename, config_output_filename, save_model, plot_prk, save_valid, top_k):
    """
    Generate model config file from model params file.
    :param model_param_filename: name of file containing model hyperparameters
                                (comma separated format, must include header and 'model_type' and 'model_param' column)
    :param config_output_filename:
    :param save_model:
    :return:
    """
    print("importing model params from: {}".format(model_param_filename))
    file_extension = model_param_filename.split(".")[-1]
    assert file_extension in 'txt' or file_extension in 'csv'
    base_path = "experiment_config"

    if config_output_filename is None:
        config_output_filename = ["modelConfig" + "_" + datetime.now().strftime("%y%m%d_%H%M") + ".yaml"]

    config_output_filename = config_output_filename[0]

    with open(model_param_filename, "r") as f:
        contents = f.readlines()
        for content in contents:
            content = content.strip()
            headers = content.split(",")
            assert 'model_param' in headers and 'model_type' in headers and 'model_name' in headers
            model_result = pd.read_csv(model_param_filename)
            if 'with_history' in headers:
                for i, hist_option in enumerate([True, False]):
                    history = 'with_history' if hist_option else 'without_history'
                    file_name = history + "_" + config_output_filename
                    file_dir = os.path.join(base_path, file_name)
                    idx = model_result.with_history == (hist_option)
                    model_result_with_hist_option = model_result[idx]
                    if (model_result_with_hist_option.shape[0]) > 0:
                        generate_model_config_history(model_result_with_hist_option, file_dir, save_model, hist_option,
                                                      plot_prk, save_valid, top_k[i])
            else:
                file_dir = os.path.join(base_path, config_output_filename)
                generate_model_config_history(model_result, file_dir, save_model, None,
                                              plot_prk, save_valid)
            break


def generate_model_config_history(model_result, config_output_filename, save_model, with_history,
                                  plot_prk, save_valid, top_k):
    """
    Helper for handling history when generating model config.
    :param with_history:
    :param save_model:
    :param config_output_filename:
    :param model_param_filename:
    :return:
    """

    def process(param, s):
        if param in ['num_boost_round', 'gamma', 'min_child_weight', 'nthread', 'n_estimators',
                     'min_samples_split', 'min_samples_leaf', 'verbose', 'random_state', 'seed',
                     'max_iter', 'n_jobs']:
            return int(s)
        elif param in ['eta', 'subsample', 'lambda', 'alpha', 'max_samples',
                       'colsample_bytree', 'colsample_bylevel', 'colsample_bynode',
                       'min_weight_fraction_leaf', 'min_impurity_decrease',
                       'tol', 'C', 'l1_ratio', 'learning_rate', 'reg_alpha', 'reg_lambda']:
            return float(s)
        elif param == 'max_depth':
            return None if s.lower() == "none" else int(s)
        elif param in ["bootstrap", 'fit_intercept', 'with_history']:
            return str2bool(s)
        else:
            return s

    with open(os.path.abspath(DEFAULT_MODEL_CONFIG), "r") as model_config_f:
        raw_config = yaml.safe_load(model_config_f.read())
        new_model_config = raw_config['model']
        model_config_f.close()

    first = True
    # model_param_idx, model_type_idx = None, None
    optional_param_idxs = []
    optional_param = []
    history_idx = None

    headers = list(model_result.columns)
    model_type_idx = headers.index("model_type")
    model_param_idx = headers.index("model_param")

    optional_param.append('model_name')
    optional_param_idxs.append(headers.index("model_name"))

    if with_history is not None:
        history_idx = headers.index('with_history')

    for row in model_result.to_numpy():
        content = ",".join([str(row[i]) for i in range(len(row))])
        vals = content.split(",")
        if not (with_history is None or str2bool(vals[history_idx]) == with_history):
            continue

        model_params = vals[model_param_idx]
        which_model = vals[model_type_idx]
        if not new_model_config['which_model']:
            new_model_config['which_model'] = [which_model]
        elif which_model not in new_model_config['which_model']:
            new_model_config['which_model'].append(which_model)
        model_params_list = map(lambda x: x.split("="), model_params.strip("'").split(";"))
        model_params_dict = dict(model_params_list)
        for key in model_params_dict:
            val = process(key, model_params_dict[key])
            if not new_model_config[which_model][key]:
                new_model_config[which_model][key] = [val]
            else:
                new_model_config[which_model][key].append(val)

        for p_idx, p in enumerate(optional_param):
            val = process(p, vals[optional_param_idxs[p_idx]])
            if p not in new_model_config[which_model].keys():
                new_model_config[which_model][p] = [val]
            else:
                new_model_config[which_model][p].append(val)

    new_model_config['mode'] = 'grid_search_import'
    raw_config['model'] = new_model_config
    raw_config['model']['save_model'] = save_model
    raw_config['with_history'] = [True, False] if with_history is None else with_history

    raw_config['model']['plot_prk'] = plot_prk
    raw_config['model']['save_valid'] = save_valid
    raw_config['top_k'] = top_k

    stream = open(os.path.abspath(config_output_filename), "w")
    yaml.dump(raw_config, stream)
    print("dumped model experiment config to file: {}".format(config_output_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model utilities.')
    parser.add_argument('-model_param', '--mp',
                        dest='model_param',
                        metavar='model_param_file', type=str, nargs=1,
                        help='path to file containing model hyperparamters')
    parser.add_argument('-config_out', '--co',
                        dest='config_out',
                        metavar='config_output_file', type=str, nargs=1,
                        help='path to output model config file')
    parser.add_argument('-save_model', '--save',
                        dest='save_model', action='store_true', default=False,
                        help='to save model or not')
    parser.add_argument('-plot_prk', '--prk',
                        dest='plot_prk', action='store_true', default=False,
                        help='to model prk plot or not')
    parser.add_argument('-save_valid', '--save_valid',
                        dest='save_valid', action='store_true', default=False,
                        help='to save last validation set or not')
    parser.add_argument('-topk', '--k',
                        dest='k',
                        metavar='model_param_file', type=float, nargs=2,
                        help='top k percent for eval metric')
    args = parser.parse_args()

    if args.model_param is not None:
        generate_model_config(args.model_param[0], args.config_out, args.save_model, args.plot_prk, args.save_valid,
                              args.k)
