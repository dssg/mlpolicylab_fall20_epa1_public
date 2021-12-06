import copy

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from deprecated import deprecated

import data.dataTransform as dataTransform


def getEngine(filePath):
    """
    Starts the engine used to query the database.
    :param filePath: path to .yaml file with database parameters
    Returns a database engine.
    """
    with open(filePath, 'r') as f:
        secrets = yaml.safe_load(f)
    db_params = secrets
    engine = create_engine('postgres://{user}:{password}@{host}:{port}/{dbname}'.format(
        host=db_params['host'],
        port=db_params['port'],
        dbname=db_params['dbname'],
        user=db_params['user'],
        password=db_params['password']
    ))
    return engine


@deprecated(reason="use the new inferDates function")
def generateSplitDates(years):
    """
    Generates train/test split dates.
    :param years: tuple, (start year, end year, prediction year), (YYYY, YYYY, YYYY)
    Returns tuples, (train start, train end), (test start, test end). All dates formatted as YYYY-MM-DD.
    """
    train_dates = ('{}-01-01'.format(years[0]), '{}-12-31'.format(years[1]))
    test_dates = ('{}-01-01'.format(years[2]), '{}-12-31'.format(years[2]))
    return train_dates, test_dates


def inferDates(data_config, logger):
    """
    Automatically compute relevant dates for training
    :param data_config: data config from yaml
    :param logger: global logger
    :return: data_config with fields overwritten
    """
    train_label_range = data_config['years']['train_label_range']
    val_label_range = data_config['years']['val_label_range']
    assert train_label_range[0] <= train_label_range[1]
    assert val_label_range[0] <= val_label_range[1]

    train_label_start = list(range(train_label_range[0], train_label_range[1] + 1))
    logger.debug("training label year from {} to {} (inclusive)".format(train_label_start[0], train_label_start[-1]))
    data_config['years']['train_label_start'] = train_label_start

    train_cohort_start = list(map(lambda x: x - 1, train_label_start))
    logger.debug("training cohort year from {} to {} (inclusive)".format(train_cohort_start[0], train_cohort_start[-1]))
    data_config['years']['train_cohort_start'] = train_cohort_start

    train_feature_end = list(map(lambda x: x - 0, train_cohort_start))
    train_feature_start = list(map(lambda x: x - 4, train_feature_end))
    logger.debug("training feature year ranges: {}".format(list(zip(train_feature_start, train_feature_end))))
    data_config['years']['train_feature_end'] = train_feature_end
    data_config['years']['train_feature_start'] = train_feature_start

    val_label_start = list(range(val_label_range[0], val_label_range[1] + 1))
    logger.debug("testing/val label year from {} to {} (inclusive)".format(val_label_start[0], val_label_start[-1]))
    data_config['years']['val_label_start'] = val_label_start

    val_cohort_start = list(map(lambda x: x - 1, val_label_start))
    logger.debug("testing/val cohort year from {} to {} (inclusive)".format(val_cohort_start[0], val_cohort_start[-1]))
    data_config['years']['val_cohort_start'] = val_cohort_start

    val_feature_end = list(map(lambda x: x - 0, val_cohort_start))
    val_feature_start = list(map(lambda x: x - 4, val_feature_end))
    logger.debug("testing/val feature year ranges: {}".format(list(zip(val_feature_start, val_feature_end))))
    data_config['years']['val_feature_end'] = val_feature_end
    data_config['years']['val_feature_start'] = val_feature_start

    return data_config


def getHandlers(template, cohort_start_year, cohort_label_start, cohort_name, engine):
    """
    Queries database for active handlers within the given train dates.
    :param template: string, SQL template from config file
    :param cohort_start_year: starting year for cohort
    :param cohort_label_start: starting year for labels
    :param cohort_name: string, cohort name from config
    :param engine: database engine
    Returns name of table that contains handlers in cohort
    """
    conn = engine.connect()

    assert cohort_start_year in range(2006, 2015, 1)
    assert cohort_label_start in range(2006, 2015, 1)

    sql_template = template.format(cohort_name, cohort_start_year, cohort_label_start)

    conn.execute(text(sql_template))
    conn.close()
    return cohort_name


def getValidationHandlers(template, cohort_start_year, val_feature_start, val_feature_end, cohort_name, engine):
    """
    Queries database for active handlers within the given train/test dates.
    :param template: string, SQL template from config file
    :param cohort_start_year: starting year for cohort
    :param cohort_label_start: starting year for labels
    :param cohort_name: string, cohort name from config
    :param engine: database engine
    Returns name of table that contains handlers in cohort
    """
    conn = engine.connect()
    sql_template = template.format(cohort_name, cohort_start_year, val_feature_start, val_feature_end)
    conn.execute(text(sql_template))
    conn.close()
    return cohort_name

def getInspectedHandlers(template, handlers_table_name, inspection_start_year, inspection_end_year, cohort_name,
                         engine):
    """
    Queries database for handlers inspected within the given dates.
    :param template: string, SQL template from config file
    :param handlers_table_name: table name of active handlers cohort
    :param inspection_start_year: start year of inspection
    :param inspection_end_year: end year of inspection
    :param cohort_name: string, cohort name from config
    :param engine: database engine
    Returns dataframe with inspected handler ID and latest evaluation start date. 
    """
    conn = engine.connect()
    sql_template = template.format(cohort_name, handlers_table_name, inspection_start_year, inspection_end_year)
    conn.execute(text(sql_template))
    conn.close()
    return cohort_name


def getLabels(template, handlers_table_name, inspection_year, engine):
    """
    Queries database for labels.
    :param template: string, SQL template from config file
    :param handlers_table_name: table name of cohort
    :param inspection_year: starting year of label period
    :param engine: database engine
    Returns dataframe with handler ID and label column.
    """
    sql_template = template.format(handlers_table_name, inspection_year)
    label_df = pd.read_sql(sql_template, engine)
    return label_df


def getFeatures(template, handlers_table, feature_start_year, feature_end_year, engine):
    """
    Queries database for features. Calls specified feature function.
    :param template: string, SQL template from config file
    :param handlers_table: table name of cohort
    :param feature_start_year: start year of feature
    :param feature_end_year: end year of feature
    :param engine: database engine
    Returns dataframe with handler ID and features columns.
    """
    sql_template = template.format(handlers_table, feature_start_year, feature_end_year)
    feature_df = pd.read_sql(sql_template, engine)
    return feature_df


def getFeaturesAcrossYears(template_name, template, handlers_table, feature_start_year, feature_end_year, engine,
                           logger):
    """
    Queries database for features with different levels of aggregation across time.
    :param template_name: name of the feature template
    :param template: dictionary, feature template from config file, contains a SQL template and list of years
    :param handlers_table: table name of cohort
    :param feature_start_year: start year of feature
    :param feature_end_year: end year of feature
    :param engine: database engine
    :param logger: global logger
    :return: one data frame that contains one set of features with different levels of aggregation across time.
    """
    base_format_config = {'year': None,
                          'table': handlers_table,
                          'start': feature_start_year,
                          'end': feature_end_year,
                          }

    def get_new_start_year(start_year, end_year, offset):
        if offset == 'total':
            new_start_year = start_year
        elif offset == 'three_year':
            new_start_year = end_year - 3 + 1  # inclusive
        elif offset == 'five_year':
            new_start_year = end_year - 5 + 1  # inclusive
        elif offset == 'most_recent':
            new_start_year = end_year
        else:
            raise NotImplementedError()
        return new_start_year

    sql_template = template['template']
    years = template['years']

    features_all_year = None

    for y in years:
        format_config = copy.copy(base_format_config)
        format_config['year'] = y
        format_config['start'] = get_new_start_year(format_config['start'], format_config['end'], y)
        formatted_template = sql_template.format(**format_config)

        logger.debug("feature: {}, aggregation: {} time:[{},{}]".format(template_name, y,
                                                                        format_config['start'],
                                                                        format_config['end']))

        label_df = pd.read_sql(formatted_template, engine)

        if features_all_year is None:
            features_all_year = label_df
        else:
            # joining the same feature but with different years
            features_all_year = features_all_year.merge(label_df, on='handler_id')

    return features_all_year


def export2csv(file_path, df):
    """
    Saves training and test datasets to file.
    :param file_path:
    :param df: dataframe
    """
    df.to_csv(file_path)


def prepCohortOneYear(data_config, sql_template_config, year_index, mode, db_engine, logger):
    """
    Prepare the cohort table for training or validation data
    :param data_config: data config from yaml
    :param sql_template_config: sql template config from yaml
    :param year_index: the index of cohort, label, feature start
    :param mode: train or val
    :param db_engine: database engine
    :param logger: global logger
    :return: train_handlers_table_name, test_handlers_table_name
    """

    cohort_start = data_config['years']['{}_cohort_start'.format(mode)][year_index]
    label_start = data_config['years']['{}_label_start'.format(mode)][year_index]
    if (mode == "val"):
        # active handlers with/without inspection history
        if (data_config['with_history']):
            cohort_template = sql_template_config['cohort_templates']['validation_with_history_cohort_template']
            cohort_name = data_config['cohort_name']['val_handlers_with_hist_table_name'] + "_" + str(cohort_start)
        else:
            cohort_template = sql_template_config['cohort_templates']['validation_without_history_cohort_template']
            cohort_name = data_config['cohort_name']['val_handlers_without_hist_table_name'] + "_" + str(cohort_start)
        val_feature_start = data_config['years']['val_feature_start'][year_index]
        val_feature_end = data_config['years']['val_feature_end'][year_index]
        handlers_table_name = getValidationHandlers(template=cohort_template,
                                                   cohort_start_year=cohort_start, 
                                                   val_feature_start=val_feature_start, 
                                                   val_feature_end=val_feature_end,
                                                   cohort_name=cohort_name,
                                                   engine=db_engine)
        logger.debug("generating active cohort tables in validation...")
        return handlers_table_name
    else:
        # active and inspected handlers
        cohort_name = data_config['cohort_name']['train_handlers_table_name']
        cohort_name = cohort_name + "_" + str(cohort_start)
        cohort_template = sql_template_config['cohort_templates']['handlers_template']
        handlers_table_name = getHandlers(template=cohort_template,
                                          cohort_start_year=cohort_start,
                                          cohort_label_start=label_start,
                                          cohort_name=cohort_name,
                                          engine=db_engine)

    logger.debug("generating active cohort tables...")

    assert mode == 'train'
    if data_config['with_history']:
        inspected_handlers_template = sql_template_config['cohort_templates']['inspected_handler_template']

        feature_start = data_config['years']['{}_feature_start'.format(mode)][year_index]
        feature_end = data_config['years']['{}_feature_end'.format(mode)][year_index]
        cohort_name_hist = data_config['cohort_name']['train_handlers_with_hist_table_name']

        cohort_name_hist = cohort_name_hist + '_' + str(cohort_start)

        handlers_table_name = getInspectedHandlers(template=inspected_handlers_template,
                                                   handlers_table_name=handlers_table_name,
                                                   inspection_start_year=feature_start,
                                                   inspection_end_year=feature_end,
                                                   cohort_name=cohort_name_hist,
                                                   engine=db_engine)

        logger.debug("generating active cohort tables for handlers with inspection history...")

    return handlers_table_name


def prepLabelsOneYear(data_config, sql_template_config, year_index, mode, handlers_table_name, db_engine):
    """
    Prepare one year labels for training and evaluation.
    :param data_config: data config from yaml
    :param sql_template_config: sql template config from yaml
    :param year_index: the index of label_start
    :param handlers_table_name: name of table to training cohort
    :param mode: train or val
    :param db_engine: database engine
    :return: training labels, test/val labels
    """
    labels = getLabels(template=sql_template_config['label_template'],
                       handlers_table_name=handlers_table_name,
                       inspection_year=data_config['years']['{}_label_start'.format(mode)][year_index],
                       engine=db_engine)

    return labels


def prepFeaturesOneYear(data_config, sql_template_config, year_index, db_engine,
                        handlers_table_name, num_data, mode, logger):
    """
    Prepare one year of features for training or evaluation.
    :param sql_template_config: config for sql templates
    :param data_config: data config from yaml
    :param year_index: the index to label/cohort/feature start
    :param db_engine: database engine
    :param handlers_table_name: name of cohort table
    :param num_data: number of examples (used for sanity check)
    :param mode: train or val
    :param logger: global logger
    :return: dataframes of train and test/val features (WITH IDs)
    """
    feature_list = data_config['features_with_history'] if data_config['with_history'] \
        else data_config['features_without_history']
    logger.debug("start preparing features:{}".format(feature_list))

    feature_start = data_config['years']['{}_feature_start'.format(mode)][year_index]
    feature_end = data_config['years']['{}_feature_end'.format(mode)][year_index]

    features_all = None

    for i in range(len(feature_list)):
        # loop through the feature template list in config
        feature_name = feature_list[i]
        # get the feature template
        feature_template = sql_template_config['feature_templates'][feature_name]

        features = getFeaturesAcrossYears(feature_name, feature_template, handlers_table_name,
                                          feature_start, feature_end,
                                          db_engine, logger)

        assert features.shape[0] == num_data

        logger.debug("successfully got {} features {} from template: {} from {} to {}".format(mode, feature_name,
                                                                                              handlers_table_name,
                                                                                              feature_start,
                                                                                              feature_end))
        if features_all is None:
            features_all = features
        else:
            features_all = features_all.merge(features, on='handler_id')

    return features_all


def prepDataOvertime(data_config, sql_templates_config, db_engine, logger):
    """
    Prepare data for training routine (not to be used with cross validation mode)
    :param data_config: data setting config
    :param sql_templates_config: sql template config
    :param db_engine: database engine
    :param logger: global logger
    :return:
    """
    train_features_all, test_features_all = None, None
    train_labels, test_labels = None, None
    # assume we have one or more years of data in our training big table
    train_label_start = data_config['years']['train_label_start']
    # assume we are only using one year's of data to validate
    val_label_start = data_config['years']['val_label_start']

    for i in range(len(train_label_start)):
        logger.debug("start preparing train features with labels in {} and "
                     "test/validation with labels in {}".format(train_label_start[i], val_label_start[0]))

        train_handlers_table_name = prepCohortOneYear(data_config=data_config,
                                                      sql_template_config=sql_templates_config,
                                                      year_index=i,
                                                      mode="train",
                                                      db_engine=db_engine,
                                                      logger=logger)

        train_labels_cur = prepLabelsOneYear(data_config=data_config,
                                             sql_template_config=sql_templates_config,
                                             year_index=i,
                                             mode="train",
                                             handlers_table_name=train_handlers_table_name,
                                             db_engine=db_engine)

        num_train = len(train_labels_cur)

        train_features_all_cur = prepFeaturesOneYear(data_config=data_config,
                                                     sql_template_config=sql_templates_config,
                                                     year_index=i,
                                                     mode="train",
                                                     db_engine=db_engine,
                                                     handlers_table_name=train_handlers_table_name,
                                                     num_data=num_train,
                                                     logger=logger)

        if i == 0:
            test_handlers_table_name = prepCohortOneYear(data_config=data_config,
                                                         sql_template_config=sql_templates_config,
                                                         year_index=0,
                                                         mode="val",
                                                         db_engine=db_engine,
                                                         logger=logger)

            logger.debug("finished generating active cohort tables (train:{}, val:{})".format(train_label_start[i],
                                                                                              val_label_start[0]))
            test_labels = prepLabelsOneYear(data_config=data_config,
                                            sql_template_config=sql_templates_config,
                                            year_index=0,
                                            mode="val",
                                            handlers_table_name=test_handlers_table_name,
                                            db_engine=db_engine)

            num_test = len(test_labels)
            logger.debug("#training instance:{}, #testing/val instance:{}".format(num_train, num_test))

            test_features_all = prepFeaturesOneYear(data_config=data_config,
                                                    sql_template_config=sql_templates_config,
                                                    year_index=0,
                                                    mode="val",
                                                    db_engine=db_engine,
                                                    handlers_table_name=test_handlers_table_name,
                                                    num_data=num_test,
                                                    logger=logger)

        else:
            logger.debug("finished generating active cohort tables (train:{})".format(train_label_start[i]))
            logger.debug("#training instance:{}".format(num_train))

        if train_features_all is None:
            train_features_all, train_labels = train_features_all_cur, train_labels_cur
        else:
            assert train_features_all.shape[1] == train_features_all_cur.shape[1]
            assert train_labels.shape[1] == train_labels_cur.shape[1]

            prev_train_len = train_features_all.shape[0]
            train_features_all = pd.concat([train_features_all, train_features_all_cur], ignore_index=True)
            train_labels = pd.concat([train_labels, train_labels_cur], ignore_index=True)
            assert train_features_all.shape[0] == train_labels.shape[0] == prev_train_len + num_train

    processed_train_features, train_onehot = dataTransform.impute_transform(train_features_all, data_config, logger)
    processed_test_features, _ = dataTransform.impute_transform(test_features_all, data_config, logger, train_onehot)

    return processed_train_features, train_labels, processed_test_features, test_labels
