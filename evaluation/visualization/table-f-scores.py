
import model_settings as ms
import copy
import settings as s
import pickle
from tabulate import tabulate
import evaluation.visualization.helper as uh
import pandas as pd
import evaluation.visualization.viz_keys as k
import evaluation.visualization.helper as h
from collections import OrderedDict
import re
# import evaluation.visualization.unified_argument_keys as uk
import model_settings as ms
import evaluation.eval_base_class as eb
import models.train_encoder_only.encoder_only_utils as eut
import evaluation.eval_filter as ef
import random
random.seed(42)
import evaluation.visualization.all_definition_configs as adc

ARGUMENT_IDS_TO_CONSIDER = adc.arguments_to_consider()


METRICS_TO_SHOW = [(ms.MACRO,ms.F1), (ms.MICRO,ms.F1), (ms.WEIGHTED,ms.F1)]
ORDER_OF_SCHEMES = ['example', 'values', 'cause to effect', 'consequences']

table_columns = []

def get_headers(approaches,desc=""):
    header_columns = ["metric" + desc]
    for dataset in adc.DATASETS_TO_USE:
        for experiment_approach in approaches:
            experiment_name = experiment_approach[ms.EXPERIMENT_DESCRIPTION]
            model_name = experiment_approach[ms.MODEL_NAME]
            dataset_name = dataset[ms.DATASET_NAME]
            header = f"{dataset_name}_{experiment_name}_{model_name}"
            header_columns.append(header)
    return header_columns

def get_fone_scores(approaches):
    table_body = []
    for metric in METRICS_TO_SHOW:
        current_row = [str(metric)]
        for dataset in adc.DATASETS_TO_USE:
            for experiment_approach in approaches:
                dataset_copy = copy.deepcopy(dataset)
                experiment_approach_copy = copy.deepcopy(experiment_approach)
                # get desired data needed for the classification
                config = {**dataset_copy, **experiment_approach_copy}
                argument_ids_to_consider_list = ARGUMENT_IDS_TO_CONSIDER[dataset_copy[ms.DATASET_NAME]]
                requested_scheme_dict = ef.get_specified_args_from_database(config,
                                                                            argument_ids_to_get_list=argument_ids_to_consider_list)  # narrow it down to a maximum of 10 arguments
                Eval = eb.EvaluateFeatures()
                DATA_TO_USE = Eval.do_evaluation(requested_scheme_dict, description=config[ms.EXPERIMENT_DESCRIPTION])
                data_metric = DATA_TO_USE[ms.METRICS_ALL_SCHEMES]
                for sub_metric in metric:
                    data_metric = data_metric[sub_metric]
                current_row.append(data_metric)
        table_body.append(current_row)
    return table_body


multi_class_data = get_fone_scores(adc.MULTI_CLASS_APPROACHES)
multi_class_data_headers = get_headers(adc.MULTI_CLASS_APPROACHES,desc="_multi_class")
multi_class_data_df = pd.DataFrame(multi_class_data)
multi_class_data_df.columns = multi_class_data_headers

means_end_data = get_fone_scores(adc.MEANS_END_APPROACHES)
means_end_data_headers = get_headers(adc.MEANS_END_APPROACHES,desc="_means_end")
means_end_data_df = pd.DataFrame(means_end_data)
means_end_data_df.columns = means_end_data_headers


fscore_df = pd.concat([multi_class_data_df, means_end_data_df], axis=1)
fscore_df = fscore_df.drop(fscore_df.columns[[5]], axis=1)


pd_df = pd.DataFrame(fscore_df).copy()
print(tabulate(pd_df, floatfmt=".2f", tablefmt="simple",showindex=False,intfmt=",",headers=pd_df.columns))

df_format = pd_df.copy()

f1_cols_bert = [("*",x) for x in pd_df.columns if s.BERT_MODEL.lower() in x.lower() and "metric" not in x.lower()]
f1_cols = [("*",x) for x in pd_df.columns if s.BERT_MODEL.lower() not in x.lower() and "metric" not in x.lower()]

nbr_to_str_func = lambda x: str(round(float(x) * 100))
CELL_REPLACE_FSCORE = lambda x : r"\fone{" + nbr_to_str_func(x) + "}"
CELL_REPLACE_FSCORE_BERT = lambda x : r"\fone{" + nbr_to_str_func(x) + "}&"

formatting_dict_fscore = {
    k.CELL_REPLACE : CELL_REPLACE_FSCORE,
    k.INCLUDE : f1_cols
}
df_format = h.apply_function_with_exclusions(df_format, data_configuration=formatting_dict_fscore)

formatting_dict_fscore_bert = {
    k.CELL_REPLACE : CELL_REPLACE_FSCORE_BERT,
    k.INCLUDE : f1_cols_bert
}
df_format = h.apply_function_with_exclusions(df_format, data_configuration=formatting_dict_fscore_bert)


# CREATE FINAL DICT AND PRINT IT
formatting_dict = {
    k.ROW_INDEX_TRANSLATION : k.SCHEME_TRANSLATION,
}
h.print_pretty_table(df_format,config_dict=formatting_dict)

