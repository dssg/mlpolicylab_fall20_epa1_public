from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

year_extensions = ["", "_three_year", "_five_year", "_most_recent", "_total"]


def flagMissing(feature_df, features, logger):
    """
    Generates a binary flag column for each feature (if NA, 1; else, 0).
    :param feature_df: dataframe with feature columns
    :param features: list of strings, feature (column) names to impute
    :param logger: global logger
    Returns updated feature_df.
    """
    temp_data = feature_df.copy()
    existing_features = list(temp_data.columns)
    # TODO: fill values other than zero?
    imp = SimpleImputer(missing_values=np.nan, fill_value=0, strategy="constant")
    imp.fit([[0.0]])
    for feature in features:
        for year_extension in year_extensions:
            # add extension because in config we only need to specify the feature name without the year extension
            feature_with_time = feature.lower() + year_extension
            if feature_with_time in existing_features:
                num_nans = temp_data[feature_with_time].isna().to_numpy().sum()
                if num_nans > 0:
                    col_name = feature_with_time + '_missing_flag'
                    temp_data[col_name] = temp_data[feature_with_time].isna().astype(int)
                    temp_data[feature_with_time] = \
                        imp.transform((temp_data[feature_with_time].to_numpy()).reshape(-1, 1))
                    # logger.debug("created new missing flag column for feature:{}".format(col_name))

    return temp_data


class oneHotEncoding(object):

    def __init__(self):
        self.oneHot_dict = dict()

    def fit(self, feature_df, features):
        temp_data = feature_df.copy()
        existing_features = list(temp_data.columns)

        for feature in features:
            for year_extension in year_extensions:
                feature_with_time = feature.lower() + year_extension
                if feature_with_time in existing_features:
                    # strip white spaces
                    temp_data[feature_with_time] = temp_data[feature_with_time].str.strip()
                    temp_column = temp_data[feature_with_time]
                    col_name = feature_with_time + '_onehot'

                    temp_column_dummy = pd.get_dummies(temp_column, prefix=col_name)
                    dummy_columns_names = list(temp_column_dummy.columns)

                    self.oneHot_dict[feature_with_time] = dummy_columns_names
                    temp_data = pd.concat([temp_data, temp_column_dummy], axis=1)
                    temp_data = temp_data.drop(columns=[feature_with_time])
        return temp_data

    def transform(self, feature_df, features):
        temp_data = feature_df.copy()
        existing_features = list(temp_data.columns)

        for feature in features:
            for year_extension in year_extensions:
                feature_with_time = feature.lower() + year_extension
                if feature_with_time in existing_features:
                    # strip white spaces
                    temp_data[feature_with_time] = temp_data[feature_with_time].str.strip()
                    temp_column = temp_data[feature_with_time]
                    col_name = feature_with_time + '_onehot'

                    temp_column_dummy = pd.get_dummies(temp_column, prefix=col_name)
                    dummy_columns_names = list(temp_column_dummy.columns)

                    # dropping onehot columns unseen during training
                    for val_onehot_column in dummy_columns_names:
                        if val_onehot_column not in self.oneHot_dict[feature_with_time]:
                            temp_column_dummy = temp_column_dummy.drop(columns=[val_onehot_column])

                    # adding onehot columns unseen during validation
                    val_columns = list(temp_column_dummy.columns)
                    for train_onehot_column in self.oneHot_dict[feature_with_time]:
                        if train_onehot_column not in val_columns:
                            temp_column_dummy[train_onehot_column] = 0

                    # reorder data alphabetically and append to dataframe to ensure order
                    temp_column_dummy = temp_column_dummy.reindex(sorted(temp_column_dummy.columns), axis=1)
                    temp_data = pd.concat([temp_data, temp_column_dummy], axis=1)
                    temp_data = temp_data.drop(columns=[feature_with_time])
        return temp_data


def impute_transform(feature_df, data_config, logger, train_onehot: oneHotEncoding = None):
    """
    Perform imputation and transformation.
    :param train_onehot: object of class OneHotEncoding
    :param feature_df: Input feature_df
    :param data_config: data setting config
    :param logger: global logger
    :return: New feature_df
    """
    # TODO: add support for more methods (don't hard code)
    imputed_feature_df = flagMissing(feature_df, data_config['impute_methods']['flag'], logger)
    if train_onehot is not None:    # test features
        transformed_feature_df = train_onehot.transform(feature_df=imputed_feature_df,
                                                        features=data_config['transform_methods']['onehot'])
    else:   # training features
        train_onehot = oneHotEncoding()
        transformed_feature_df = train_onehot.fit(feature_df=imputed_feature_df,
                                                  features=data_config['transform_methods']['onehot'])

    return transformed_feature_df, train_onehot


def consistent_transform(df1, df2):
    """
    assume df1 column number <= df2 column number
    want to keep features between two feature df consistent
    :param df1: dataframe of features
    :param df2: dataframe of features
    :return: new df1, new df2 where two df has the same set of columns/features
    """
    if (df1.shape[1] == df2.shape[1]):
        return df1, df2
    df1_columns = set(df1.columns.to_list())
    df2_columns = set(df2.columns.to_list())
    for feature in df2_columns:
        if feature not in df1_columns:
            # feature should be a missing_flag, 0 indicates it's not missing
            df1[feature] = 0
    df1 = df1.reindex(sorted(df1.columns), axis=1)
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    assert df1.shape[1] == df2.shape[1]
    return df1, df2


def normalize_data(df):
    """
    Normalize the data using equation x_i - min(x_i.)/ max(x_i.) - min(x_i.)
    :param df: data
    :return:
    """
    df_min = df.min(axis=0, skipna=True)
    df_max = df.max(axis=0, skipna=True)
    diff = df_max - df_min
    min_equal_max_idx = diff == 0
    diff[min_equal_max_idx] = 1

    return (df - df_min) / diff
