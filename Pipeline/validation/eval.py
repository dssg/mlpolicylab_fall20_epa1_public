from collections import namedtuple
import os
import pickle

import numpy as np
import pandas as pd
import sklearn.metrics as metric
import matplotlib.pyplot as plt

from utils.modelUtils import MODELS_MAP
from utils.databaseUtils import *

RUN = False  # change this to True if want to test with Main


class Evaluator(object):

    def __init__(self, scores, y_true, save_path=None, model_name=None, eval_year=None, logger=None):
        """
        :param scores : dataframe, the raw prediction scores from model (e.g., logit for logistics regression)
                        column: [handler_id, score] e.g., [[0, 0.5],
                                                   [1, 0.6],
                                                   [2, 0.7]]
        :param y_true : dataframe, test/validation labels
                        column: [handler_id, formal] e.g., [[0, 1],
                                                   [1, 0],
                                                   [2, 1]]
        :param save_path: base path to save all eval result
        :param model_name: name of the model being evaluated
        :param eval_year: validation/testing label year
        :param logger: global logger
        """
        self.scores = scores
        self.y_true = y_true
        self.save_path = save_path
        self.model_name = model_name
        self.eval_year = eval_year
        self.LOGGER = logger

        res = self.y_true.merge(self.scores, on='handler_id')
        self.sorted_res = res.sort_values(by='score', ascending=False)

    def gen_dir(self):
        """
        Generate saving directory if needed.
        """
        if not os.path.exists(os.path.join(self.save_path, self.model_name)):
            os.makedirs(os.path.join(self.save_path, self.model_name))

    def threshold_classifier(self, threshold=0.5) -> np.ndarray:
        return (self.scores[:, 1] > threshold).astype(np.float)

    def confusion_matrix(self, y_pred, labels=None) -> np.ndarray:
        """
        Compute confusion matrix to evaluate the accuracy of a classification.
        (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
        :param y_pred: Estimated targets as returned by a classifier.
        :param labels: List of labels to index the matrix (optional).
        :return: confusion matrix
        """
        assert self.y_true.shape == y_pred.shape, "ytrue and ypred must have the same length!"
        return metric.confusion_matrix(y_true=self.y_true[:, 1],
                                       y_pred=y_pred[:, 1],
                                       labels=labels)

    def num_labeled_at_k(self, k):
        num_scores = self.scores.shape[0]
        i = int(k * num_scores)
        return self.sorted_res.iloc[:i, :].dropna().shape[0]

    def recall_disparity_fdr_at_k_by_cat(self, k, metric_name, ref_group, comp_groups):

        def cal_ratio(val1, val2):
            if val2 == 0.0:
                if val1 == 0.0:
                    ratio = 1.0
                else:
                    ratio = np.inf
            else:
                ratio = val1 / val2
            return ratio

        num_scores = self.sorted_res.shape[0]
        i = int(k * num_scores)

        top_k_percent = self.sorted_res.iloc[:i, :]
        top_k_labeled = top_k_percent.dropna()
        total_relevant = self.sorted_res[['formal']].sum()[0]

        top_k_labeled_ref = top_k_labeled[top_k_labeled.loc[:, metric_name] == ref_group]
        ref_top_k_N = top_k_labeled_ref.shape[0]

        recall_disparity = []
        fdr_ratio = []
        top_k_N = []

        ref_fdr = np.nan if ref_top_k_N == 0 else (top_k_labeled_ref.loc[:, 'formal'] == 0.0).sum() / ref_top_k_N
        ref_recall = np.nan if total_relevant == 0 else ref_top_k_N / total_relevant

        for c in comp_groups:
            top_k_labeled_c = top_k_labeled[top_k_labeled.loc[:, metric_name] == c]
            c_top_k_N = top_k_labeled_c.shape[0]
            c_fdr = np.nan if c_top_k_N == 0 else (top_k_labeled_c.loc[:, 'formal'] == 0.0).sum() / c_top_k_N
            c_recall = np.nan if total_relevant == 0 else c_top_k_N / total_relevant

            fdr_ratio.append(cal_ratio(c_fdr, ref_fdr))
            recall_disparity.append(cal_ratio(c_recall, ref_recall))
            top_k_N.append(c_top_k_N)

        return recall_disparity, fdr_ratio, top_k_N, ref_top_k_N

    def precision_support_recall_at_k(self, k):
        """
        Calculate precision at k%.
        :param k: percentage of population
        :return: (precision value, support value)
        """
        num_scores = self.scores.shape[0]
        i = int(k * num_scores)

        top_k_percent = self.sorted_res.iloc[:i, :]
        top_k_labeled = top_k_percent.dropna()

        support_at_k = float(top_k_labeled.shape[0]) / i
        y_true_at_k = top_k_labeled[['formal']].to_numpy().reshape(-1)
        relevant_at_k = np.ones(y_true_at_k.shape[0]) * y_true_at_k

        # (# of recommended items that are relevant @k)/(# of recommended items at k)
        if top_k_labeled.shape[0] == 0:
            precision_at_k = np.nan
        else:
            precision_at_k = sum(relevant_at_k) / top_k_labeled.shape[0]

        total_num_relevant = (self.y_true.dropna())[['formal']].sum(axis=0)[0]
        recall_at_k = sum(relevant_at_k) / total_num_relevant

        return precision_at_k, support_at_k, recall_at_k

    def false_discovery_rate_at_k(self, k):
        """
        Calculate FDR at k%
        :param k: percentage of population
        :return: FDR = (labeled 0 in top k with label / top k with label)
        """
        num_scores = self.scores.shape[0]
        i = int(k * num_scores)

        top_k_percent = self.sorted_res.iloc[:i, :]
        top_k_labeled = top_k_percent.dropna()

        y_true_at_k = top_k_labeled[['formal']].to_numpy().reshape(-1)
        relevant_at_k = np.ones(y_true_at_k.shape[0]) * y_true_at_k

        if top_k_labeled.shape[0] == 0:
            return np.nan
        fdr_at_k = (top_k_labeled.shape[0] - sum(relevant_at_k)) / top_k_labeled.shape[0]
        return fdr_at_k

    def precision_recall_k(self):
        """
        Calculate the Precision-Recall @ Top K% metric
        :return: namedtuple with k, precision, recall fields
        """
        result = namedtuple('PRK', ['k', 'precision', 'recall'])

        num_scores = self.scores.shape[0]
        total_num_relevant = (self.y_true.dropna())[['formal']].sum(axis=0)[0]

        k_tmp, precision_tmp, recall_tmp = [], [], []
        for i in range(1, num_scores + 1):
            # TODO: maybe not increment by one example at a time, could be slow if we have tons of examples...
            k = i / num_scores

            top_k_percent = self.sorted_res.iloc[:i, :]
            top_k_labeled = top_k_percent.dropna()
            if top_k_labeled.shape[0] == 0:
                precision_at_k = np.nan
                recall_at_k = np.nan
            else:
                y_true_at_k = top_k_labeled[['formal']].to_numpy().reshape(-1)
                relevant_at_k = np.ones(y_true_at_k.shape[0]) * y_true_at_k

                # (# of recommended items that are relevant @k)/(# of recommended items at k)
                precision_at_k = sum(relevant_at_k) / float(top_k_labeled.shape[0])
                # (# of recommended items that are relevant @k)/(total # of relevant items)
                recall_at_k = sum(relevant_at_k) / total_num_relevant

            k_tmp.append(k)
            precision_tmp.append(precision_at_k)
            recall_tmp.append(recall_at_k)

        result.k, result.precision, result.recall = np.array(k_tmp), np.array(precision_tmp), np.array(recall_tmp)
        five_percent_idx = round(num_scores * 0.05)

        if self.LOGGER:
            self.LOGGER.debug("Precision@5% : {} "
                              "Recall@5% : {}".format(result.precision[five_percent_idx],
                                                      result.recall[five_percent_idx]))

        return result

    def graph_prk(self, result):
        """
        Displays Precision-Recall @ Top K% curve
        :param result: namedtuple with k, precision, recall fields
        """
        fig, host = plt.subplots(figsize=(7, 5))
        par1 = host.twinx()

        p1, = host.plot(result.k, result.precision, "b-")
        p2, = par1.plot(result.k, result.recall, "r-")

        host.set_xlabel("percent of population")
        host.set_ylabel("precision")
        par1.set_ylabel("recall")

        host.set_xlim(0, 1)
        host.set_ylim(0, 1)
        par1.set_ylim(0, 1)

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())

        # TODO: customizable name
        self.gen_dir()
        fig_name = "val_{}_prk.png".format(self.eval_year)
        save_file = os.path.join(self.save_path, self.model_name, fig_name)
        if self.LOGGER:
            self.LOGGER.debug("plotting prk and saving to {}".format(save_file))
        plt.savefig(save_file, dpi=400)
        plt.close()


class BiasFairness(object):
    def __init__(self, with_history: bool, metric_name: str, ref_group, base_dir: str):
        """
        Initialize bias and fairness metric generator.
        :param with_history: for the with history cohort or without history cohort
        :param metric_name: bias and fairness metric name (same as the one in the table)
        :param ref_group: reference group name
        :param base_dir: base directory of the models
        """
        self.with_history = with_history
        self.base_dir = base_dir
        self.metric_table = None
        self.metric_name = metric_name  # only support single metric
        self.ref_group = ref_group

    def get_bias_fairness_table(self, metric_template, db_secrets_path='/data/groups/epa1/wenyu/wenyu_db_secrets.yaml'):
        """
        Get the table with metric category from database.
        :param metric_template: sql template for metric of interest
        :param db_secrets_path: path to database secret
        :return:
        """
        secrets = getSecrets(db_secrets_path)
        engine = startEngine(secrets)
        self.metric_table = pd.read_sql(metric_template, engine)

    def generate_bias_fairness_metric(self, k):
        """
        Generate bias and fairness metric table
        :param k: top k percent for calculating precision, recall etc.
        :return:
        """
        model_types = list(MODELS_MAP.keys())
        history_extension = '_with' if self.with_history else '_without'

        metric_categories = list(self.metric_table.loc[:, self.metric_name].unique())
        metric_categories.remove(self.ref_group)

        group_names = ['{}_vs_{}'.format(cat, self.ref_group) for cat in metric_categories]
        model_name_lst = []
        model_type_lst = []
        precision_lst = []
        top_k_num_labeled_ref_lst = []
        top_k_num_labeled_lst = [[] for i in range(len(metric_categories))]
        recall_disparity_lst = [[] for i in range(len(metric_categories))]
        fdr_ratio_lst = [[] for i in range(len(metric_categories))]

        for model in model_types:

            model_paths = os.path.join(self.base_dir, model + history_extension)
            if not os.path.exists(model_paths):
                continue

            all_models = os.listdir(model_paths)

            val_label_path = os.path.join(self.base_dir, model + history_extension, 'valid_labels_2014.csv')
            val_labels_df = pd.read_csv(val_label_path)

            for m in all_models:
                if '.csv' not in m:
                    score_df_path = os.path.join(self.base_dir, model + history_extension, m, 'scores_2014.csv')
                    scores_df = pd.read_csv(score_df_path)
                    merged_df = pd.merge(scores_df, self.metric_table, 'inner', on='handler_id')

                    e_total = Evaluator(scores=scores_df, y_true=val_labels_df)
                    precision_k, _, recall_k = e_total.precision_support_recall_at_k(k)
                    precision_lst.append(precision_k)
                    model_type_lst.append(model)
                    model_name_lst.append(m)

                    e_metric = Evaluator(scores=merged_df, y_true=val_labels_df)
                    r_d, fdr_ratio, num_labeled, ref_top_k_N = \
                        e_metric.recall_disparity_fdr_at_k_by_cat(k=k,
                                                                 metric_name=self.metric_name,
                                                                 ref_group=self.ref_group,
                                                                 comp_groups=metric_categories)

                    top_k_num_labeled_ref_lst.append(ref_top_k_N)

                    for idx, cat in enumerate(metric_categories):
                        top_k_num_labeled_lst[idx].append(num_labeled[idx])
                        recall_disparity_lst[idx].append(r_d[idx])
                        fdr_ratio_lst[idx].append(fdr_ratio[idx])

        for i in range(len(metric_categories)):
            d = {'model_name': model_name_lst,
                 'model_type': model_type_lst,
                 'precision_k': precision_lst,
                 'recall_disparity': recall_disparity_lst[i],
                 'fdr_ratio': fdr_ratio_lst[i],
                 'num_labeled_k': top_k_num_labeled_lst[i],
                 'num_labeled_ref_k': top_k_num_labeled_ref_lst}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(self.base_dir, "bias_fairness_top{}%_{}_{}.csv".format(int(k * 100),
                                                                                          self.metric_name,
                                                                                          group_names[i])),
                      index=False)


if __name__ == '__main__':
    if RUN:
        # example from
        # https://queirozf.com/entries/evaluation-metrics-for-ranking-problems-introduction-and-examples
        scores_example = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                   [0.63, 0.24, 0.36, 0.85, 0.47, 0.71, 0.90, 0.16]]).T
        labels_example = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                   [1, 0, 1, 0, 0, 1, 1, 0]]).T

        e = Evaluator(scores=scores_example, y_true=labels_example)
        PRK = e.precision_recall_k()
        print("K:", list(PRK.k), "\nprecision:", list(PRK.precision), "\nrecall:", list(PRK.recall))

        e.graph_prk(PRK, "foo.png")
