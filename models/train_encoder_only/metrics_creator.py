import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import model_settings as ms
from collections import defaultdict
import evaluation.metrics_base as mb
import json


class EvaluateFeatures():

    def __init__(self, number_to_scheme_translation_dict = None):
        # get data from means-end prompting
        self.metrics = mb.MetricsBase()
        self.multiple_samples = False
        self.counter_none = 0
        self.number_to_scheme_dict = number_to_scheme_translation_dict

    def calc_metrics_single_scheme(self,ground=None,pred=None,scheme_name=None):

        # use the scheme name to create a binary mask of ground list and pred list
        ground_list_binary = [1 if item == scheme_name else 0 for item in ground]
        pred_list_binary = [1 if item == scheme_name else 0 for item in pred]

        assert len(ground_list_binary) == len(pred_list_binary)

        metrics = self.metrics.eval_metrics(ground=ground_list_binary, pred=pred_list_binary)  # evaluate classification of nodes for this scheme
        return {scheme_name : metrics}


    # central functions, serves to compute variety of metrics for (multi-label) classification
    def compute_metrics(self, p) :

        if hasattr(p, 'label_ids') and hasattr(p, 'predictions') :
            ground_list = p.label_ids
            pred_list = p.predictions
        else:
            ground_list = p[0]
            pred_list = p[1]

        if self.multiple_samples:
            pred_list = torch.sigmoid(torch.tensor(pred_list)) > 0.5  # Multi-label thresholding
            pred_list = np.asarray(pred_list, dtype=int)
            labels_to_use = [x for x in range(len(true_labels[0]))]

        else:
            labels_to_use = np.unique(ground_list)
            pred_list = np.array(pred_list)
            if len(pred_list.shape) != 1 :
                pred_list = np.argmax(pred_list, axis=1) # select the index of the highest value

        if self.number_to_scheme_dict is not None: # work with names instead of numbers
            ground_list = [self.number_to_scheme_dict[item] for item in ground_list]
            pred_list = [self.number_to_scheme_dict[item] for item in pred_list]
            labels_to_use = list(set(ground_list))

        nbr_examples_total = len(ground_list)
        macro_data_dict = self.metrics.eval_metrics(ground=ground_list, pred=pred_list, average="macro")
        micro_data_dict = self.metrics.eval_metrics(ground=ground_list, pred=pred_list, average="micro")
        weighted_data_dict = self.metrics.eval_metrics(ground=ground_list, pred=pred_list, average="weighted")

        all_schemes_metric_dict = {
            ms.MACRO : macro_data_dict,
            ms.MICRO : micro_data_dict,
            ms.WEIGHTED : weighted_data_dict,
            ms.NBR : nbr_examples_total,
        }

        single_schemes_metric_dict = dict()
        for label in labels_to_use:
            single_metric_dict = self.calc_metrics_single_scheme(ground=ground_list,pred=pred_list,scheme_name=label)
            single_schemes_metric_dict.update(single_metric_dict)

        if self.multiple_samples:
            precision_samples = precision_score(y_true=true_labels, y_pred=preds, average='samples')
            recall_samples = recall_score(y_true=true_labels, y_pred=preds, average='samples')
            f1_samples = f1_score(y_true=true_labels, y_pred=preds, average='samples')

            all_multiple_samples_metric_dict = {
                    'Precision (samples)' : precision_samples,
                    'Recall (samples)' : recall_samples,
                    'F1 Score (samples)' : f1_samples,
                }


        return {ms.METRICS_SINGLE_SCHEMES : single_schemes_metric_dict, ms.METRICS_ALL_SCHEMES : all_schemes_metric_dict, ms.PREDICTION : pred_list, ms.GROUND : ground_list}










if __name__ == "__main__":
    true_labels = np.array([
        [1, 0, 1, 0],  # Sample 1: Label 1 and Label 3 are true
        [0, 1, 0, 1],  # Sample 2: Label 2 and Label 4 are true
        [1, 1, 0, 1]  # Sample 3: Label 1, Label 2, and Label 4 are true
    ])

    preds = np.array([
        [1, 0, 1, 0],  # Sample 1: Predicted correctly for Label 1 and Label 3
        [0, 0, 1, 1],  # Sample 2: Predicted incorrectly for Label 2, correctly for Label 4
        [1, 1, 1, 1]  # Sample 3: Predicted correctly for all labels
    ])

    #true_labels = np.array([1, 0, 1, 1, 0])  # Actual labels: [Positive, Negative, Positive, Positive, Negative]
    #preds = np.array([1, 0, 0, 1, 0])  # Predicted labels: [Positive, Negative, Negative, Positive, Negative]

    compute_metrics([true_labels, preds],multiple_samples=True)