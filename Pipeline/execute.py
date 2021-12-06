import importlib
import time

import pandas as pd

import data.dataPrep as dataPrep
import data.dataTransform as dataTransform
from utils.loggingUtils import custom_logger, shutdown_logger
from utils.modelUtils import *
from validation.eval import Evaluator

LOG_FREQ_PERCENT = 0.25
DEFAULT_MODEL_CONFIG = "experiment_config/sampleConfig.yaml"


def load_config(model_config_path):
    """
    Load all config files.
    :return: database config, data config, model config, sql template config, meta information config
    """
    with open('experiment_config/dataConfig.yaml', 'r') as data_config_f:
        raw_config = yaml.safe_load(data_config_f.read())
        db_config = raw_config['database']
        data_config = raw_config['data']
        data_config_f.close()

    with open('data/templates.yaml', 'r') as f2:
        sql_templates_config = yaml.safe_load(f2.read())
        f2.close()

    model_configs, meta_configs, history_configs, top_k_configs = [], [], [], []
    if model_config_path is None:
        model_config_path = [DEFAULT_MODEL_CONFIG]
    for i in range(len(model_config_path)):
        with open(os.path.abspath(model_config_path[i]), "r") as model_config_f:
            raw_config = yaml.safe_load(model_config_f.read())
            print("loaded model config from: {}".format(model_config_path[i]))
            model_configs.append(raw_config['model'])
            meta_configs.append(raw_config['meta'])
            history_configs.append(raw_config['with_history'])
            top_k_configs.append(raw_config['top_k'])
            model_config_f.close()

    return db_config, data_config, model_configs, sql_templates_config, meta_configs, history_configs, top_k_configs


def fit_one_model(which_model, which_model_config, model_name, train_features, train_labels, train_set, logger):
    """
    Fit one model given specifications
    :param which_model: model type
    :param which_model_config: model config for which_model
    :param model_name: name of the model
    :param train_features: training features
    :param train_labels: training labels
    :param train_set: time range of training set [training set label start, label end]
    :param logger: global logger
    :return:
    """
    if which_model in SKLEARN_MODELS or which_model == "xgboost":
        class_name = MODELS_MAP[which_model]
        model_class = getattr(importlib.import_module("model.{}".format(class_name)), "{}".format(class_name))
        model = model_class(name=model_name,
                            logger=logger,
                            train_set=train_set,
                            **which_model_config)
        model.fit(data=[train_features, train_labels],
                  label_name="formal",
                  id_name="handler_id")
    else:
        logger.error("sorry we don't support {}!".format(which_model))
        raise NotImplementedError()
    return model


def eval_one_model(model, plot, save_path, test_features, test_labels, logger, eval_year, top_k):
    """
    Evaluate one model on one validation set
    :param save_path: path to save eval results
    :param model: trained model
    :param test_features: validation/testing features
    :param test_labels: validation/testing labels
    :param logger: global logger
    :param eval_year: validation label year
    :return: precision & support at top 5% # TODO: make this variable for different cohort
    """
    score = model.predict(data=test_features,
                          label_name="formal",
                          id_name="handler_id")
    score_df = pd.DataFrame(data=score, columns=['handler_id', 'score'])
    label_df = test_labels[['handler_id', 'formal']]
    logger.debug("evaluating on test set from year {} with {} rows".format(eval_year, label_df.shape[0]))
    eva = Evaluator(scores=score_df,
                    y_true=label_df,
                    save_path=save_path,
                    model_name=model.name,
                    eval_year=eval_year,
                    logger=logger)
    prk = eva.precision_recall_k()
    if plot:
        eva.graph_prk(prk)
        if eval_year == 2014:
            dataPrep.export2csv(os.path.join(save_path, model.name, 'scores_{}.csv'.format(eval_year)), score_df)
    logger.debug("finished evaluation")
    precision_at_k, support_at_k, recall_at_k = eva.precision_support_recall_at_k(top_k)
    return precision_at_k, support_at_k


def save_one_model(which_model, save_path, which_model_config, model):
    """
    Export one model
    :param save_path: directory to save model
    :param which_model: model type
    :param which_model_config: model config for model type
    :param model: trained model
    :return: None
    """
    if which_model in SKLEARN_MODELS:
        model.export_model(path=save_path)
    elif which_model == 'xgboost':
        model.export_model(path=save_path,
                           save_mode=which_model_config['model_save_mode'])
    else:
        raise NotImplementedError


def train_routine(which_model, model_name, db_config, data_config, model_config, sql_templates_config, top_k, logger):
    """
    Training routine (train only one model)
    :param which_model: model type
    :param model_name: name of the model
    :param logger: global logger
    :param db_config: database config
    :param data_config: data setting config
    :param model_config: model config
    :param sql_templates_config: sql template config
    :return:
    """
    # create engine
    db_engine = dataPrep.getEngine(filePath=db_config['db_secret_directory'])
    logger.debug("successfully connected to database engine")

    # get dates
    data_config = dataPrep.inferDates(data_config, logger)
    logger.debug("successfully inferred dates from config")

    logger.info("start preparing train and test/validation features and labels")

    # prepare train features/labels for multiple years and val features/labels for one year
    train_features_all, train_labels, \
    test_features_all, test_labels = dataPrep.prepDataOvertime(data_config, sql_templates_config, db_engine, logger)

    logger.info("Total # training instance:{}, "
                "#testing/val instance:{}".format(train_features_all.shape[0],
                                                  test_features_all.shape[0]))

    train_set = [data_config['years']['train_label_start'][0], data_config['years']['train_label_start'][-1]]
    model = fit_one_model(which_model=which_model,
                          which_model_config=model_config[which_model],
                          model_name=model_name,
                          train_features=train_features_all,
                          train_labels=train_labels,
                          train_set=train_set,
                          logger=logger)

    if model_config['save_model']:
        save_one_model(which_model=which_model,
                       save_path=model_config['base_dir'],
                       which_model_config=model_config[which_model],
                       model=model)

    if model_config['eval_model']:
        _ = eval_one_model(model=model,
                           save_path=model_config['base_dir'],
                           plot=model_config['plot_prk'],
                           test_features=test_features_all,
                           test_labels=test_labels,
                           logger=logger,
                           eval_year=data_config['years']['val_label_start'][0],
                           top_k=top_k)


def cv_routine_multiple_model(which_model, which_model_grid, db_config, data_config,
                              model_config, sql_templates_config, top_k, logger):
    """
    Temporal cross-validation routine with model grid search.
    :param which_model: type of model running running
    :param which_model_grid: model grid for which_model
    :param db_config: database config
    :param data_config: global data config
    :param model_config: global model config
    :param sql_templates_config: sql feature templates
    :param logger: global logger
    :return: prk matrix
    """
    db_engine = dataPrep.getEngine(filePath=db_config['db_secret_directory'])
    logger.debug("successfully connected to database engine")

    data_config = dataPrep.inferDates(data_config, logger)
    logger.debug("successfully inferred dates from config")

    logger.info("start preparing train and test/validation features and labels")

    train_features_all, test_features_all = None, None
    train_labels, test_labels = None, None
    train_label_start = data_config['years']['train_label_start']

    precision_matrix = np.zeros(shape=(len(which_model_grid), len(train_label_start)))
    support_matrix = np.zeros(shape=(len(which_model_grid), len(train_label_start)))
    which_model_names = []
    which_model_params = []

    save_model = model_config['save_model']
    eval_model = model_config['eval_model']

    start_start_time = time.time()
    for i in range(len(train_label_start)):
        val_label_start = data_config['years']['val_label_start'][i]
        train_set = [data_config['years']['train_label_start'][0], data_config['years']['train_label_start'][i]]

        logger.debug("start preparing train features with labels in {} and "
                     "test/validation with labels in {}".format(train_label_start[i], val_label_start))

        train_handlers_table_name = dataPrep.prepCohortOneYear(data_config=data_config,
                                                               sql_template_config=sql_templates_config,
                                                               year_index=i,
                                                               mode="train",
                                                               db_engine=db_engine,
                                                               logger=logger)

        test_handlers_table_name = dataPrep.prepCohortOneYear(data_config=data_config,
                                                              sql_template_config=sql_templates_config,
                                                              year_index=i,
                                                              mode="val",
                                                              db_engine=db_engine,
                                                              logger=logger)

        logger.info("finished generating active cohort tables (train:{}-{}, val:{})".format(train_label_start[0],
                                                                                            train_label_start[i],
                                                                                            val_label_start))
        logger.debug("start preparing training and testing labels")

        train_labels_cur = dataPrep.prepLabelsOneYear(data_config=data_config,
                                                      sql_template_config=sql_templates_config,
                                                      year_index=i,
                                                      mode="train",
                                                      handlers_table_name=train_handlers_table_name,
                                                      db_engine=db_engine)

        test_labels_cur = dataPrep.prepLabelsOneYear(data_config=data_config,
                                                     sql_template_config=sql_templates_config,
                                                     year_index=i,
                                                     mode="val",
                                                     handlers_table_name=test_handlers_table_name,
                                                     db_engine=db_engine)

        num_train, num_test = len(train_labels_cur), len(test_labels_cur)
        logger.info("#training instance:{}, #testing/val instance:{}".format(num_train, num_test))

        train_features_all_cur = dataPrep.prepFeaturesOneYear(data_config=data_config,
                                                              sql_template_config=sql_templates_config,
                                                              year_index=i,
                                                              mode="train",
                                                              handlers_table_name=train_handlers_table_name,
                                                              num_data=num_train,
                                                              db_engine=db_engine,
                                                              logger=logger)

        test_features_all_cur = dataPrep.prepFeaturesOneYear(data_config=data_config,
                                                             sql_template_config=sql_templates_config,
                                                             year_index=i,
                                                             mode="val",
                                                             handlers_table_name=test_handlers_table_name,
                                                             num_data=num_test,
                                                             db_engine=db_engine,
                                                             logger=logger)

        if train_features_all is None:
            train_features_all, train_labels = train_features_all_cur, train_labels_cur
            test_features_all, test_labels = test_features_all_cur, test_labels_cur
        else:
            assert train_features_all.shape[1] == train_features_all_cur.shape[1]
            assert test_features_all.shape[1] == test_features_all_cur.shape[1]

            prev_train_len = train_features_all.shape[0]
            # train features/labels are cumulative
            train_features_all = pd.concat([train_features_all, train_features_all_cur], ignore_index=True)
            train_labels = pd.concat([train_labels, train_labels_cur], ignore_index=True)
            assert train_features_all.shape[0] == train_labels.shape[0] == prev_train_len + num_train

            # test/val features/label are different each time since we're only using one year
            test_features_all, test_labels = test_features_all_cur, test_labels_cur

        processed_train_features, train_onehot = dataTransform.impute_transform(train_features_all, data_config, logger)
        processed_test_features, _ = dataTransform.impute_transform(test_features_all, data_config, logger,
                                                                    train_onehot)
        processed_train_features, processed_test_features = dataTransform.consistent_transform(processed_train_features,
                                                                                               processed_test_features)
        assert processed_train_features.shape[1] == processed_test_features.shape[1]
        logger.info("Total # training instance for this fold:{} [{}, {}], "
                    "#testing/val instance for this fold:{} [{}], "
                    "Total # of features: {}".format(processed_train_features.shape[0],
                                                     train_set[0], train_set[1],
                                                     processed_test_features.shape[0],
                                                     val_label_start,
                                                     processed_train_features.shape[1]))
        curr_best_prk = None
        N = len(which_model_grid)
        log_N = 1 if N * LOG_FREQ_PERCENT < 1 else int(N * LOG_FREQ_PERCENT)
        start_time = time.time()
        for j in range(N):
            model_grid_j = which_model_grid[j]
            model_name_j = which_model + "_" + str(j)
            if 'model_name' in which_model_grid[j].keys():
                model_name_j = which_model_grid[j]['model_name']

            model = fit_one_model(which_model=which_model,
                                  which_model_config=model_grid_j,
                                  model_name=model_name_j,
                                  train_features=processed_train_features,
                                  train_labels=train_labels,
                                  train_set=train_set,
                                  logger=logger)

            if save_model:
                save_one_model(which_model=which_model,
                               save_path=model_config['base_dir'],
                               which_model_config=model_config[which_model],
                               model=model)

            if eval_model:
                precision_at_k, support_at_k = eval_one_model(model=model,
                                                              plot=model_config['plot_prk'],
                                                              save_path=model_config['base_dir'],
                                                              test_features=processed_test_features,
                                                              test_labels=test_labels,
                                                              eval_year=val_label_start,
                                                              logger=logger,
                                                              top_k=top_k)
                precision_matrix[j, i] = precision_at_k
                support_matrix[j, i] = support_at_k

                if curr_best_prk is None or precision_at_k > curr_best_prk:
                    curr_best_prk = precision_at_k
                    logger.info(
                        "current best prk@top {}%:{} with support@top {}%: {}".format(int(top_k*100),
                                                                                      curr_best_prk,
                                                                                      int(top_k*100),
                                                                                      support_at_k,
                                                                                      ))

                if i == len(train_label_start) - 1 and model_config['save_valid']:
                    base_path = model_config['base_dir']
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)
                    out_path = os.path.join(base_path, "valid_features_{}.csv".format(val_label_start))
                    dataPrep.export2csv(out_path, processed_test_features)

                    out_path = os.path.join(base_path, 'valid_labels_{}.csv'.format(val_label_start))
                    dataPrep.export2csv(out_path, test_labels)
                    logger.debug("exported last validation set to {}".format(out_path))

            if (j + 1) % log_N == 0:
                percent = (j + 1) / N
                past_time = time.time() - start_time
                minute_since_start = time.time() - start_start_time
                logger.info("{:.1f}% completed for this fold. "
                            "Avg speed: {:2.2f}s/iteration. "
                            "Total time: {:2.2f} min.".format(percent * 100,
                                                              past_time / log_N,
                                                              minute_since_start / 60))
                start_time = time.time()

            if i == 0:
                which_model_names.append(model_name_j)
                which_model_params.append(model_grid_j)
                # save_model_config(os.path.join(model_config['base_dir'], model_name_j), model_grid_j)

    return precision_matrix, support_matrix, which_model_names, which_model_params


def grid_search_routine(model_name, db_config, data_config, model_config, sql_templates_config, top_k, logger):
    """
    Model grid search routine.
    :param model_name: name of the model
    :param db_config: database config
    :param data_config:
    :param model_config:
    :param sql_templates_config:
    :param logger:
    :return:
    """
    # TODO: move history setting to model config
    model_config_copy = copy.copy(model_config)
    final_output = dict()
    new_base_dir = os.path.join(model_config_copy['base_dir'], model_name)
    direct_import = model_config_copy['mode'] == "grid_search_import"
    for which_model in model_config_copy['which_model']:
        final_output[which_model] = dict()
        model_grid = parse_hyperparams(model_config_copy[which_model], direct_import=direct_import)
        logger.info("starting grid search for model:{}".format(which_model))
        history = "with" if data_config['with_history'] else "without"
        model_config_copy['base_dir'] = os.path.join(new_base_dir, which_model + "_" + history)

        precision_matrix, support_matrix, which_model_names, which_model_params = \
            cv_routine_multiple_model(which_model=which_model,
                                      which_model_grid=model_grid,
                                      db_config=db_config,
                                      data_config=data_config,
                                      model_config=model_config_copy,
                                      sql_templates_config=sql_templates_config,
                                      top_k=top_k,
                                      logger=logger)

        final_output[which_model]['precision_overtime'] = precision_matrix
        final_output[which_model]['support_overtime'] = support_matrix
        final_output[which_model]['model_names'] = which_model_names
        final_output[which_model]['which_model_params'] = which_model_params

    return final_output


def main(model_config_names, db_config, data_config, sql_templates_config, model_configs, meta_configs,
         history_configs, top_k_configs):
    """
    Main routine
    :param history_configs: list of history configs
    :param meta_configs: list of meta configs
    :param model_configs: list of model configs
    :param model_config_names: list of name of model configs
    :param db_config: database config
    :param data_config: data related config
    :param sql_templates_config: feature templates config
    :return:
    """
    time_for_filename = datetime.now().strftime("%y%m%d_%H%M")
    if model_config_names is None:
        model_config_names = [DEFAULT_MODEL_CONFIG]
    for config_idx in range(len(model_config_names)):
        model_config_name = model_config_names[config_idx].split("/")[-1][0:-len(".yaml")]
        meta_config = meta_configs[config_idx]
        model_config = model_configs[config_idx]
        history_config = history_configs[config_idx]
        top_k_config = top_k_configs[config_idx]

        np.random.seed(meta_config['random_seed'])
        mode = model_config["mode"]
        if 'grid_search' in mode:
            appendix = 'autogen' if mode == "grid_search_autogen" else 'import'
            name = "grid" + "_" + appendix
            dir_name = name + "_" + time_for_filename + "_" + model_config_name
            base_dir = os.path.join(model_config['base_dir'], dir_name)
            LOGGER = custom_logger(name=name,
                                   dir=base_dir,
                                   save=meta_config['dump_log'],
                                   level=meta_config['log_level'])

            if not isinstance(history_config, list):
                history_config = [history_config]
                top_k_config = [top_k_config]

            for idx, cur_history in enumerate(history_config):
                data_config['with_history'] = cur_history
                top_k = top_k_config[idx]
                history_str = 'with' if cur_history else 'without'
                if mode == "grid_search_autogen":
                    LOGGER.info("start grid search on cohort {} history "
                                "using config {}".format(history_str, model_config_name))
                else:
                    LOGGER.info("start running model grid on cohort {} history with "
                                "imported params from config {}".format(history_str, model_config_name))
                final_output = grid_search_routine(dir_name, db_config, data_config, model_config,
                                                   sql_templates_config, top_k, LOGGER)
                save_grid_search_result(base_dir, "{}_grid_result_top{}%.csv".format(history_str,
                                                                                     int(100*top_k)), final_output)
            LOGGER.info("finished grid search routine!")
        else:
            # if not grid search, only fit the zeroth model type
            which_model = model_config['which_model'][0]
            model_name = which_model + "_" + datetime.now().strftime("%y%m%d-%H%M")

            # if not grid search mode, only fit one type of model
            if isinstance(history_config, list):
                history_config = history_config[0]
                top_k = top_k_config[0]

            history = 'with' if history_config else 'without'
            data_config['with_history'] = history_config

            LOGGER = custom_logger(name=which_model,
                                   dir=os.path.join(model_config['base_dir'], model_name),
                                   save=meta_config['dump_log'],
                                   level=meta_config['log_level'])
            if mode == 'cv':
                which_model_config = model_config[which_model]
                LOGGER.info("start cross validation of {} model on cohort {} history "
                            "using config {}".format(model_name, history, model_config_name))
                precision_lst = cv_routine_multiple_model(which_model, [which_model_config], db_config, data_config,
                                                          model_config, sql_templates_config, top_k, LOGGER)
                LOGGER.info("Precision%5 for cv folds:{}".format(precision_lst))
            elif mode == 'train':
                LOGGER.info("start training {} model on cohort {} history "
                            "using config {}".format(model_name, history, model_config_name))
                train_routine(which_model, model_name, db_config,
                              data_config, model_config, sql_templates_config, top_k, LOGGER)
            else:
                raise NotImplementedError("unrecognized mode")
        shutdown_logger(LOGGER)
        del LOGGER


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute pipeline.')
    parser.add_argument('-model_config', '--mc',
                        dest='model_config_filenames',
                        metavar='model_config_filenames', type=str, nargs='+',
                        help='paths to model config file')
    args = parser.parse_args()

    db_config_main, data_config_main, model_configs_main, \
        sql_templates_config_main, meta_configs_main, \
        history_configs_main, top_k_configs_main = load_config(args.model_config_filenames)

    main(args.model_config_filenames, db_config_main, data_config_main, sql_templates_config_main,
         model_configs_main, meta_configs_main, history_configs_main, top_k_configs_main)
