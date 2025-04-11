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
import evaluation.visualization.unified_argument_keys as uk
import model_settings as ms
import evaluation.eval_base_class as eb
import evaluation.eval_filter as ef
import settings as s
import random
random.seed(42)
# configurable approaches provide information about the index, the used experiment and the dataset under consideration


multi_class_no_def_zeroshot_model = [{ms.COLLECTION : ms.MULTI_CLASS_ENCODER, ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_NO_DEFINITIONS, ms.DATASET_NAME : ms.ETHIX_EVALUATION_TEST, ms.SPLIT : ms.TEST}]
multi_class_all_def_zeroshot_model = [{ms.COLLECTION : ms.MULTI_CLASS_ENCODER, ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS, ms.DATASET_NAME : ms.ETHIX_EVALUATION_TEST, ms.SPLIT : ms.TEST}]
means_end_zeroshot_model = [{ms.COLLECTION : ms.MEANS_END_ENCODER, ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_ENCODER, ms.DATASET_NAME : ms.ETHIX_EVALUATION_TEST, ms.SPLIT : ms.TEST}]

# multi_class_no_def_finetuned_model = [{ms.ARGUMENT_ID : all_argument_ids, ms.MODEL:ms.LLAMA, ms.COLLECTION : ms.MULTI_CLASS, ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS, ms.DATASET : ms.VRAKATSELI_SPLIT, ms.FINETUNED : True,  ms.SPLIT : ms.TEST}]
# requested_data = ef.get_number_of_schemes(multi_class_no_def_finetuned_model[0])

experiment_configs = [ ] + multi_class_no_def_zeroshot_model  # + multi_class_all_def_zeroshot_model + means_end_zeroshot_model

language_models = [s.OLLAMA_MODEL] # [ms.GPT,ms.LLAMA]
metrics_to_show = [ms.PRECISION, ms.RECALL, ms.F1]

#order_of_schemes = skt.SchemeGroupingInfo().groups_in_use_list
# determine order of schemes by the list
order_of_schemes = ['example','analogy','alternatives','values','cause to effect','consequences']
experiment_description = "zero_shot_all_names_all_definitions"

elastic_information = []


collect_data_list = []
header_table_creator = []
# function to go through the specified header files

column_names = ["scheme"]
for experiment_approach in experiment_configs:
    experiment_description = experiment_approach[ms.EXPERIMENT_DESCRIPTION]
    for model in language_models :
        for metric in metrics_to_show :
            column_names.append(experiment_description + "_" + model + "_" + metric)

table_body = []

for scheme in order_of_schemes:
    current_row = [scheme]
    for experiment_approach in experiment_configs:
        for model in language_models:

            experiment_approach = copy.deepcopy(experiment_approach)
            experiment_approach.update({ms.MODEL_NAME : model})

            # get number of data with the implemented constraint that only argument ids from the id list should be retrieved
            requested_scheme_dict = ef.get_specified_args_from_database(experiment_approach, argument_ids_to_get_list=uk.COMMON_ARG_IDS_TO_DISPLAY)
            Eval = eb.EvaluateFeatures()
            DATA_TO_USE = Eval.do_evaluation(requested_scheme_dict, description=experiment_approach[ms.EXPERIMENT_DESCRIPTION])
            data_scheme = DATA_TO_USE[ms.METRICS_SINGLE_SCHEMES][scheme]
            for metric in metrics_to_show:
                current_row.append(data_scheme[metric])
    table_body.append(current_row)

# create final row with micro data
final_row = [ms.MICRO]

for experiment_approach in experiment_configs:
    for model in language_models:
        experiment_approach = copy.deepcopy(experiment_approach)
        experiment_approach.update({ms.MODEL_NAME : model})
        requested_scheme_dict = ef.get_specified_args_from_database(experiment_approach)
        Eval = eb.EvaluateFeatures()
        DATA_TO_USE = Eval.do_evaluation(requested_scheme_dict)
        data_scheme = DATA_TO_USE[ms.METRICS_ALL_SCHEMES]
        for metric in metrics_to_show:
            data = data_scheme[ms.MICRO][metric]
            final_row.append(data)
table_body.append(final_row)


# create final row with macro data
final_row = [ms.MACRO]

for experiment_approach in experiment_configs:
    for model in language_models:
        experiment_approach = copy.deepcopy(experiment_approach)
        experiment_approach.update({ms.MODEL_NAME : model})
        requested_scheme_dict = ef.get_specified_args_from_database(experiment_approach)
        Eval = eb.EvaluateFeatures()
        DATA_TO_USE = Eval.do_evaluation(requested_scheme_dict)
        data_scheme = DATA_TO_USE[ms.METRICS_ALL_SCHEMES]
        for metric in metrics_to_show:
            data = data_scheme[ms.MACRO][metric]
            final_row.append(data)
table_body.append(final_row)


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
nbr_to_str_func = lambda x : str(x * 100).split(".")[0]
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






