import copy
import sys

import numpy as np
import model_settings
import settings as s
import utils.utils as ut
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import model_settings as ms
import evaluation.metrics_base as eb
import json
from pathlib import Path
import warnings
class EvaluateFeatures():
    def __init__(self):
        # get data from means-end prompting
        self.metrics = eb.MetricsBase()
        self.counter_none = 0
    def do_evaluation(self, data_instances_dict,description=""):
        if description == ms.MEANS_END_PATH_ENCODER or description == ms.MEANS_END_PATH_DECODER:
            print("Switching to Evaluation of means-end path classification")
            return self.evaluate_ask_scheme_paths(data_instances_dict)
        return self.evaluate_scheme_metrics(data_instances_dict)

    # evaluation is done for a specific model
    def evaluate_scheme_metrics(self,data_instances_dict):

        pred_list = []
        ground_list = []

        # go through scheme dict
        for scheme,data_instance in data_instances_dict.items():
            for arg in data_instance:
                pred = arg[ms.PREDICTION]
                grd = arg[ms.SCHEME]
                pred_list.append(pred)
                ground_list.append(grd)

        scheme_names_list = sorted(list(set(ground_list)))

        nbr_examples_total = len(ground_list)
        macro_data_dict = self.metrics.eval_metrics(ground_list, pred_list, average="macro")
        micro_data_dict = self.metrics.eval_metrics(ground_list, pred_list, average="micro")
        weighted_data_dict = self.metrics.eval_metrics(ground_list, pred_list, average="weighted")

        all_schemes_metric_dict = {
            ms.MACRO : macro_data_dict,
            ms.MICRO : micro_data_dict,
            ms.WEIGHTED : weighted_data_dict,
            ms.NBR : nbr_examples_total,

        }

        single_schemes_metric_dict = dict()
        for scheme in scheme_names_list:
            single_metric_dict = self.calc_metrics_single_scheme(ground=ground_list,pred=pred_list,scheme_name=scheme)
            single_schemes_metric_dict.update(single_metric_dict)

        return {ms.METRICS_SINGLE_SCHEMES : single_schemes_metric_dict, ms.METRICS_ALL_SCHEMES : all_schemes_metric_dict}

    def calc_metrics_single_scheme(self,ground=None,pred=None,scheme_name=None):

        # use the scheme name to create a binary mask of ground list and pred list
        ground_list_binary = [1 if item == scheme_name else 0 for item in ground]
        pred_list_binary = [1 if item == scheme_name else 0 for item in pred]

        assert len(ground_list_binary) == len(pred_list_binary)

        metrics = self.metrics.eval_metrics(ground=ground_list_binary, pred=pred_list_binary)  # evaluate classification of nodes for this scheme
        return {scheme_name : metrics}

    def evaluate_ask_scheme_paths_single_scheme(self, scheme_argument_list):

        """Evaluate Performance of the single used features, we do it separate for every scheme"""
        scheme_node_pred_dict = defaultdict(list) # store predicted values
        scheme_node_ground_dict = defaultdict(list) # store ground truth values

        for argument in scheme_argument_list:
            classification_paths_list = argument[ms.PREDICTION]

            # go through each possible classification path and notate the models decisions
            for path in classification_paths_list:
                for single_decision_node in path:
                    node = single_decision_node[ms.NODE]
                    node_correct = single_decision_node[ms.SCHEME]
                    node_pred = single_decision_node[ms.PREDICTION]
                    if node_pred is None:
                        self.counter_none += 1
                        print("None prediction - ignoring")
                        sys.exit(1) # this should not happen, but if it does, we stop the evaluation
                        continue
                    scheme_node_pred_dict[node].append(node_pred)
                    scheme_node_ground_dict[node].append(node_correct)

        # some schemes enable multiple ways, which have to be filtered out
        contradictory_nodes = []
        for node,ground_truth in scheme_node_ground_dict.items():
            if len(set(ground_truth)) > 1:
                contradictory_nodes.append(node)
        for x in contradictory_nodes:
            print(f"Contradictory node: {x} is removed")
            del scheme_node_pred_dict[x]
            del scheme_node_ground_dict[x]

        scheme_nodes_evaluated = dict() # dictionary to store the corresponding evaluations
        for node in scheme_node_ground_dict.keys():
            ground_labels = scheme_node_ground_dict[node]
            pred_labels = scheme_node_pred_dict[node]
            assert len(ground_labels) == len(pred_labels)

            metrics = self.metrics.eval_metrics(ground=ground_labels, pred=pred_labels) # evaluate classification of nodes for this scheme
            scheme_nodes_evaluated.update({node : metrics})

        # calculate metrics for nodes used for the scheme
        scheme_question_nodes_data_dict = {ms.PREDICTION : scheme_node_pred_dict, ms.GROUND : scheme_node_ground_dict,
                                           ms.UNDEFINED : contradictory_nodes, ms.EVALUATION : scheme_nodes_evaluated}
        return scheme_question_nodes_data_dict


    # assume full evaluation has been done
    def evaluate_ask_scheme_paths(self, schemes_dict):
        # is of type dict(x) ([Any])
        schemes_nodes_data_dict = dict()

        # go through all schemes and evaluate the corresponding nodes, which are needed for the scheme evaluation
        for scheme,data_instances in schemes_dict.items():
            all_features_dict = self.evaluate_ask_scheme_paths_single_scheme(data_instances)
            schemes_nodes_data_dict.update({scheme : all_features_dict})

        prediction_dict_list = []
        ground_dict_list = []

        # perform corresponding evaluations for all features across all schemes
        for feature_name, single_feature_schemes_dict in schemes_nodes_data_dict.items():
            prediction_dict = single_feature_schemes_dict[ms.PREDICTION]
            ground_dict = single_feature_schemes_dict[ms.GROUND]
            prediction_dict_list.append(prediction_dict)
            ground_dict_list.append(ground_dict)

        # join the dict with ground und pred of each node to get prediction and ground for all schemes for each node
        total_pred_dict = ut.join_dicts(prediction_dict_list)
        total_ground_dict = ut.join_dicts(ground_dict_list)

        total_evaluated_nodes_metrics_dict = dict()
        # calculate corresponding ground metrics
        for feature_name, ground_list in total_ground_dict.items():
            pred_list = total_pred_dict[feature_name]

            total_zeros = ground_list.count(0)
            total_ones = ground_list.count(1)
            total_ground_unique_vals = len(set(ground_list))
            nbr_examples_total = len(ground_list)

            macro_data_dict = self.metrics.eval_metrics(ground_list, pred_list, average="macro")
            micro_data_dict = self.metrics.eval_metrics(ground_list, pred_list, average="micro")
            weighted_data_dict = self.metrics.eval_metrics(ground_list, pred_list, average="weighted")

            total_zeros_metrics = self.metrics.eval_metrics(ground_list, pred_list, average="binary", pos_label=0)
            total_ones_metrics = self.metrics.eval_metrics(ground_list, pred_list, average="binary", pos_label=1)

            share_answer_0 = total_zeros / nbr_examples_total
            share_answer_1 = total_ones / nbr_examples_total

            all_features_data_dict = {
                                    ms.MACRO : macro_data_dict,
                                    ms.MICRO : micro_data_dict,
                                    ms.WEIGHTED : weighted_data_dict,
                                    ms.NBR : nbr_examples_total,
                                    ms.TOTAL_0 : total_zeros, # total numbers
                                    ms.TOTAL_1 : total_ones,
                                    ms.SHARE_ANSWER_0 : share_answer_0,
                                    ms.SHARE_ANSWER_1 : share_answer_1,
                                    ms.TOTAL_0_METRICS : total_zeros_metrics, # calculated scores
                                    ms.TOTAL_1_METRICS : total_ones_metrics,
                                    ms.GROUND_VALS_NBR : total_ground_unique_vals,
                                     }

            total_evaluated_nodes_metrics_dict.update({feature_name : all_features_data_dict})
        print("None predictions: ",self.counter_none)
        # create final data for combination
        return {ms.ASK_SCHEME_NODE_DATA_DICT :schemes_nodes_data_dict, ms.ASK_TOTAL_EVAL_DICT : total_evaluated_nodes_metrics_dict}







