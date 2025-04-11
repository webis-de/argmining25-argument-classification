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
import evaluation.eval_filter as ef
import settings as s
import random
random.seed(42)
import evaluation.visualization.all_definition_configs as adc


ARGUMENT_IDS_TO_CONSIDER = adc.arguments_to_consider()

metrics_to_show = [ms.PRECISION, ms.RECALL, ms.F1]

#order_of_schemes = skt.SchemeGroupingInfo().groups_in_use_list
# determine order of schemes by the list
ORDER_OF_SCHEMES = ['example', 'values', 'cause to effect', 'consequences']



collect_data_list = []
header_table_creator = []
# function to go through the specified header files

column_names = ["scheme"]
for dataset in adc.DATASETS_TO_USE:
    dataset_name = dataset[ms.DATASET_NAME]
    for experiment_approach in adc.MULTI_CLASS_APPROACHES:
        model = experiment_approach[ms.MODEL_NAME]
        experiment_description = experiment_approach[ms.EXPERIMENT_DESCRIPTION]
        for metric in metrics_to_show :
            column_names.append(dataset_name + "_" + experiment_description + "_" + model + "_" + metric)

table_body = []

for scheme in ORDER_OF_SCHEMES: #
    current_row = [scheme]
    for dataset in adc.DATASETS_TO_USE:
        for experiment_approach in adc.MULTI_CLASS_APPROACHES:
            dataset_copy = copy.deepcopy(dataset)
            experiment_approach = copy.deepcopy(experiment_approach)

            # get desired data needed for the classification
            config = {**dataset, **experiment_approach}
            argument_ids_to_consider_list = ARGUMENT_IDS_TO_CONSIDER[dataset_copy[ms.DATASET_NAME]]

            requested_scheme_dict = ef.get_specified_args_from_database(config,argument_ids_to_get_list=argument_ids_to_consider_list)  # narrow it down to a maximum of 10 arguments
            Eval = eb.EvaluateFeatures()
            DATA_TO_USE = Eval.do_evaluation(requested_scheme_dict, description=config[ms.EXPERIMENT_DESCRIPTION])
            data_scheme = DATA_TO_USE[ms.METRICS_SINGLE_SCHEMES][scheme]
            for metric in metrics_to_show:
                current_row.append(data_scheme[metric])
    table_body.append(current_row)

pd_df = pd.DataFrame(table_body, columns=column_names)
print(tabulate(pd_df, floatfmt=".2f", tablefmt="simple",showindex=False,intfmt=",",headers=pd_df.columns))


# do corresponding labeling of the columns
def get_corresponding_columns(df, name): # get corresponding columns which work reliable
    return [("*",col) for col in df.columns if name in col.lower()]

precision_cols = get_corresponding_columns(pd_df, ms.PRECISION)
recall_cols = get_corresponding_columns(pd_df, ms.RECALL)
f1_cols = get_corresponding_columns(pd_df, ms.F1)

df_format = pd_df.copy()

# Cell Replace Functions
nbr_to_str_func = lambda x: str(round(float(x) * 100))
CELL_REPLACE_PREC = lambda x : r"\pre{" + nbr_to_str_func(x) + "}"
CELL_REPLACE_REC = lambda x : r"\rec{" + nbr_to_str_func(x) + "}"
CELL_REPLACE_FSCORE = lambda x : r"\fone{" + nbr_to_str_func(x) + "}&"

formatting_dict_precision = {
    k.CELL_REPLACE : CELL_REPLACE_PREC,
    k.INCLUDE : precision_cols
}
df_format = h.apply_function_with_exclusions(df_format, data_configuration=formatting_dict_precision)

formatting_dict_recall = {
    k.CELL_REPLACE : CELL_REPLACE_REC,
    k.INCLUDE : recall_cols
}
df_format = h.apply_function_with_exclusions(df_format, data_configuration=formatting_dict_recall)

formatting_dict_fscore = {
    k.CELL_REPLACE : CELL_REPLACE_FSCORE,
    k.INCLUDE : f1_cols
}
df_format = h.apply_function_with_exclusions(df_format, data_configuration=formatting_dict_fscore)

# CREATE FINAL DICT AND PRINT IT
formatting_dict = {
    k.ROW_INDEX_TRANSLATION : k.SCHEME_TRANSLATION,
}
df_format.set_index(ms.SCHEME, inplace=True)
h.print_pretty_table(df_format,config_dict=formatting_dict)



mewo = 1
# can be used for corresponding finetuning options

# specify that corresponding fields are being changed according to the pre and rec symbols
# combine the underlying tables






