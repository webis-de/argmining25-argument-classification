from pathlib import Path
import model_settings as mset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score,fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import numpy as np

class MetricsBase():

    def eval_metrics(self, ground=None, pred=None, average="binary", labels=None, pos_label=1):
        accuracy_undefined = False
        with warnings.catch_warnings(record=True) as w :
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always", category=UndefinedMetricWarning)
            accuracy_single = accuracy_score(y_true=ground, y_pred=pred)
            if len(w) > 0 :
                accuracy_undefined = True

        precision_undefined = False
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always", category=UndefinedMetricWarning)
            precision_single = precision_score(y_true=ground, y_pred=pred, average=average,labels=labels,pos_label=pos_label)
            if len(w) > 0:
                precision_undefined = True

        recall_undefined = False
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always", category=UndefinedMetricWarning)
            recall_single = recall_score(y_true=ground,y_pred=pred, average=average,labels=labels,pos_label=pos_label)
            if len(w) > 0:
                recall_undefined = True

        f1_undefined = False
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always", category=UndefinedMetricWarning)
            precision, recall, f1, support = precision_recall_fscore_support(y_true=ground, y_pred=pred, average=average,labels=labels,pos_label=pos_label)
            if len(w) > 0:
                f1_undefined = True

        if isinstance(precision,np.ndarray):
            assert precision.all() == precision_single.all()
            assert recall.all() == recall_single.all()

            precision = precision.tolist()
            recall = recall.tolist()
            f1 = f1.tolist()

        else:
            assert precision == precision_single
            assert recall == recall_single

        if accuracy_undefined or precision_undefined or recall_undefined or f1_undefined:
            print("Accuracy undefined: ",accuracy_undefined)
            print("Precision undefined: ",precision_undefined)
            print("Recall undefined: ",recall_undefined)
            print("F1 undefined: ",f1_undefined)

        return_dict = {
            mset.ACCURACY : accuracy_single,
            mset.PRECISION : precision,
            mset.PRECISION_UNDEF : precision_undefined,
            mset.RECALL: recall,
            mset.RECALL_UNDEF : recall_undefined,
            mset.F1 : f1,
            mset.F1_UNDEF : f1_undefined }

        return return_dict


    def eval_beta(self,ground,pred,average="binary"):
        fbeta_undefined = False
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always", category=UndefinedMetricWarning)
            fbeta = fbeta_score(y_true=ground, y_pred=pred, beta=0.5,average=average,pos_label=1)
            if len(w) > 0:
                fbeta_undefined = True

        return_dict = {
            mset.F_BETA : fbeta,
            mset.F1_UNDEF : fbeta_undefined }

        return return_dict
