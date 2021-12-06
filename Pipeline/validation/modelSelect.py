import pandas as pd
import numpy as np


class modelSelector(object):
    """
    Model Selector class
    """

    def __init__(self):
        self.model_result = None

    def read_result_from_file(self, result_filepath):
        """
        Read model training results from file.
        :param result_filepath: path to model training result csv file
        :return:
        """
        if self.model_result is None:
            self.model_result = pd.read_csv(result_filepath)
        else:
            result = pd.read_csv(result_filepath)
            self.model_result = pd.concat([self.model_result, result], axis=0, ignore_index=True)

    @staticmethod
    def parse_str_column_as_float(df, column_idx, sep, new_column_names=None):
        """
        Parse a column of string into multiple columns of float
        :param df: original dataframe object to parse
        :param column_idx: index to target column
        :param sep: separating pattern in string (e.g., ";" if string is "0.5;0.5"
        :param new_column_names: optional names for newly generated columns
        :return:
        """
        temp_df = df.iloc[:, column_idx].str.split(pat=sep, expand=True)
        temp_npy = temp_df.to_numpy().astype(np.float32)
        temp_df = pd.DataFrame(temp_npy)
        if new_column_names:
            temp_df = temp_df.rename(columns=dict(zip(list(range(0, len(new_column_names))),
                                                      new_column_names)))
        return temp_df

    def parse_columns(self, df, name):
        """
        Parse precision and compute average precision
        :param name: name of the target column (i.e., precision or support)
        :param df: dataframe object containing 'precision_overtime' column
        :return:
        """
        column_names = list(df.columns)
        new_column_names = list(map(lambda x: '{}_'.format(name) + str(x), list(range(2009, 2015))))
        precision_df = self.parse_str_column_as_float(df,
                                                      column_names.index('{}_overtime'.format(name)),
                                                      ';',
                                                      new_column_names)
        precision_df['avg_{}'.format(name)] = precision_df.mean(axis=1, skipna=False)
        precision_df = pd.concat([df, precision_df], axis=1)
        precision_df.drop(axis=1, columns='{}_overtime'.format(name))
        return precision_df

    def parse_precision_support(self):
        self.model_result = self.parse_columns(self.model_result, "support")
        self.model_result = self.parse_columns(self.model_result, "precision")

    def filter_models(self, method="avg_support_overtime", cutoff=0.5):
        """
        Filter models by method using the given cutoff.
        :param method: method name
        :param cutoff: cutoff to threshold models
        :return:
        """
        column_names = list(self.model_result.columns)
        old_N = self.model_result.shape[0]

        if method == "avg_support_overtime":
            assert 'support_overtime' in column_names
            support_higher_than_threshold = self.model_result['avg_support'] >= cutoff

            self.model_result = self.model_result[support_higher_than_threshold]
        else:
            raise NotImplementedError

        new_N = self.model_result.shape[0]
        print("filtered out {} models".format(old_N - new_N))

    def rank_models(self, method="avg_precision_overtime"):
        """
        Rank models by specified method.
        :param method: method name
        :return:
        """
        column_names = list(self.model_result.columns)
        ordered = None

        if method == "avg_precision_overtime":
            assert 'precision_overtime' in column_names
            ordered = self.model_result.sort_values(by=['avg_precision',
                                                        'precision_2014',
                                                        'precision_2013',
                                                        'precision_2012',
                                                        'precision_2011',
                                                        'precision_2010',
                                                        'precision_2009'], ascending=False)
        elif method == 'twice_as_important_overtime':
            sum_weights = (1 + 1 / 2 + 1 / 4 + 1 / 8 + 1 / 16 + 1 / 32)
            self.model_result['discounted_precision'] = self.model_result.precision_2014 * 1 / sum_weights + \
                                         self.model_result.precision_2013 * (1 / 2) / sum_weights + \
                                         self.model_result.precision_2012 * (1 / 4) / sum_weights + \
                                         self.model_result.precision_2011 * (1 / 8) / sum_weights + \
                                         self.model_result.precision_2010 * (1 / 16) / sum_weights + \
                                         self.model_result.precision_2009 * (1 / 32) / sum_weights
            ordered = self.model_result.sort_values(by='discounted_precision', ascending=False)
        else:
            raise NotImplementedError

        return ordered

    @staticmethod
    def take_N_and_export(with_history: bool, ordered_df: pd.DataFrame, n: int, file_path: str, append: bool = False):
        """
        Export top N model configuration to file.
        :param with_history: specify to fit on with or without history cohort
        :param ordered_df: dataframe of ranked models
        :param n: number of models to export
        :param file_path: name of file to export
        :return:
        """
        n_df = ordered_df.head(n)
        n_df.loc[:, 'with_history'] = with_history
        columns = list(n_df.columns)
        index_to_take = list(
            map(lambda x: columns.index(x), ['model_type', 'model_name', 'model_param', 'with_history']))
        if append:
            n_df.iloc[:, index_to_take].to_csv(file_path, index=False, mode='a')
        else:
            n_df.iloc[:, index_to_take].to_csv(file_path, index=False)
        print("successfully exported top {} model configs to {}".format(n, file_path))
