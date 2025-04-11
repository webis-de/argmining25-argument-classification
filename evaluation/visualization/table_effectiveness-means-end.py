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
# configurable approaches provide information about the index, the used experiment and the dataset under consideration



ARGUMENT_IDS_TO_CONSIDER = adc.arguments_to_consider()

#order_of_schemes = skt.SchemeGroupingInfo().groups_in_use_list
# determine order of schemes by the list
ORDER_OF_SCHEMES = ['example', 'values', 'cause to effect', 'consequences']

# function to go through the specified header files
METRICS_TO_SHOW = [ms.PRECISION, ms.RECALL, ms.F1]

NODE_NAMES_FOR_ACCURACY = eut.get_available_nodes()

# create the column names for the table
# columns are the node names with the corresponding metrics

def get_headers_accuracy(experiment_approach):
    column_names = ["scheme"]
    experiment_approach = copy.deepcopy(experiment_approach)
    model = experiment_approach[ms.MODEL_NAME]
    experiment_description = experiment_approach[ms.EXPERIMENT_DESCRIPTION]
    for node in NODE_NAMES_FOR_ACCURACY:
        column_names.append(ms.ACCURACY + "_" + experiment_description + "_" + model + "_" + node)
    return column_names

def get_headers_scores(experiment_approach):
    column_names = ["scheme"]
    experiment_approach = copy.deepcopy(experiment_approach)
    model = experiment_approach[ms.MODEL_NAME]
    experiment_description = experiment_approach[ms.EXPERIMENT_DESCRIPTION]
    for metric in METRICS_TO_SHOW :
        column_names.append(experiment_description + "_" + model + "_" + metric)
    return column_names



# obtain the required accuracies  for a designated dataset
def get_accuracy_for_datasets(dataset,experiment_approach):
    table_body = []
    for scheme in ORDER_OF_SCHEMES: #
        current_row = [scheme]
        for node in NODE_NAMES_FOR_ACCURACY:

            dataset_copy = copy.deepcopy(dataset)
            experiment_approach_copy = copy.deepcopy(experiment_approach)

            # get desired data needed for the classification
            config = {**dataset_copy, **experiment_approach_copy}
            argument_ids_to_consider_list = ARGUMENT_IDS_TO_CONSIDER[dataset_copy[ms.DATASET_NAME]]
            requested_scheme_dict = ef.get_specified_args_from_database(config,argument_ids_to_get_list=argument_ids_to_consider_list)  # narrow it down to a maximum of 10 arguments
            Eval = eb.EvaluateFeatures()
            DATA_TO_USE = Eval.do_evaluation(requested_scheme_dict, description=config[ms.EXPERIMENT_DESCRIPTION])
            try:
                data  = DATA_TO_USE[ms.ASK_SCHEME_NODE_DATA_DICT][scheme][ms.EVALUATION][node][ms.ACCURACY]
            except KeyError :
                data = ""
            current_row.append(data)
        table_body.append(current_row)
    return table_body

def get_precision_recall_fone(dataset,experiment_approach):
    table_body = []
    for scheme in ORDER_OF_SCHEMES:
        current_row = [scheme]

        dataset_copy = copy.deepcopy(dataset)
        experiment_approach_copy = copy.deepcopy(experiment_approach)

        # get desired data needed for the classification

        config = {**dataset_copy, **experiment_approach_copy}
        argument_ids_to_consider_list = ARGUMENT_IDS_TO_CONSIDER[dataset_copy[ms.DATASET_NAME]]
        requested_scheme_dict = ef.get_specified_args_from_database(config,argument_ids_to_get_list=argument_ids_to_consider_list)  # narrow it down to a maximum of 10 arguments
        Eval = eb.EvaluateFeatures()
        DATA_TO_USE = Eval.do_evaluation(requested_scheme_dict, description=config[ms.EXPERIMENT_DESCRIPTION])
        for metric in METRICS_TO_SHOW:
            data = DATA_TO_USE[ms.METRICS_SINGLE_SCHEMES][scheme][metric]
            current_row.append(data)
        table_body.append(current_row)
    return table_body


def merge_datasets_on_schemes(df1, df2):
    # Merge the two dataframes on the 'scheme' column
    merged_df = pd.merge(df1, df2, on='scheme', suffixes=('_df1', '_df2'))

    # Check if the merged dataframe has the same number of rows as the original dataframes
    assert merged_df.shape[0] == df1.shape[0] == df2.shape[0], "Merged dataframe does not have the same number of rows as original dataframes"

    return merged_df

def dataframe_for_dataset_and_configs(dataset=None, path_config=None,scores_config = None ):
    accuracy_raw = get_accuracy_for_datasets(dataset, path_config)
    accuracy_df = pd.DataFrame(accuracy_raw)
    header_tmp = get_headers_accuracy(path_config)
    accuracy_df.columns = header_tmp

    classification_raw = get_precision_recall_fone(dataset, scores_config)
    classification_df = pd.DataFrame(classification_raw)
    header_tmp = get_headers_scores(scores_config)
    classification_df.columns = header_tmp

    merged_df = merge_datasets_on_schemes(accuracy_df, classification_df)
    classification_df_test = classification_df.drop("scheme",axis=1)
    final_df = pd.concat([accuracy_df, classification_df_test], axis=1)
    assert final_df.shape == merged_df.shape, f"Dataframes have different lengths: {len(final_df)} vs {len(merged_df)}"
    return merged_df

# ETHIX DATASET
ethix_df_encoder = dataframe_for_dataset_and_configs(adc.ETHIX, path_config=adc.MEANS_END_PATH_ENCODER, scores_config=adc.MEANS_END_ENCODER)
ethix_df_decoder = dataframe_for_dataset_and_configs(adc.ETHIX, path_config=adc.MEANS_END_PATH_DECODER, scores_config=adc.MEANS_END_DECODER)

# ETHIX FULLDATASET
ethix_df = merge_datasets_on_schemes(ethix_df_encoder, ethix_df_decoder)


# USTV Dataset
ustv_df_encoder = dataframe_for_dataset_and_configs(adc.USTV, path_config=adc.MEANS_END_PATH_ENCODER, scores_config=adc.MEANS_END_ENCODER)
ustv_df_decoder = dataframe_for_dataset_and_configs(adc.USTV, path_config=adc.MEANS_END_PATH_DECODER, scores_config=adc.MEANS_END_DECODER)

# USTV FULLDATASET
ustv_df = merge_datasets_on_schemes(ustv_df_encoder, ustv_df_decoder)

pd_df = pd.concat([ustv_df, ethix_df], axis=0)
print(tabulate(pd_df, floatfmt=".2f", tablefmt="simple",showindex=False,intfmt=",",headers=pd_df.columns))


# APPLY FINAL FORMATTING OF THE CORRESPONDING DATAFRAME

def format_dataframe(df_format):
    df_format = df_format.copy()

    # do corresponding labeling of the columns
    def get_corresponding_columns(df, name) :  # get corresponding columns which work reliable
        return [("*", col) for col in df.columns if name in col.lower()]

    accuracy_cols = [("*", col) for col in df_format.columns if ms.ACCURACY in col.lower() and not col.endswith("47")]
    accuracy_cols_end = [("*", col) for col in df_format.columns if ms.ACCURACY in col.lower() and col.endswith("47")]

    precision_cols = get_corresponding_columns(pd_df, ms.PRECISION)
    recall_cols = get_corresponding_columns(pd_df, ms.RECALL)
    f1_cols = get_corresponding_columns(pd_df, ms.F1)

    # Cell Replace Functions
    nbr_to_str_func = lambda x: str(round(float(x) * 100))

    CELL_REPLACE_ACC = lambda x : r"\cell{" + nbr_to_str_func(x) + "}"
    CELL_REPLACE_ACC_END = lambda x : r"\cell{" + nbr_to_str_func(x) + "}&"


    CELL_REPLACE_PREC = lambda x : r"\pre{" + nbr_to_str_func(x) + "}"
    CELL_REPLACE_REC = lambda x : r"\rec{" + nbr_to_str_func(x) + "}"
    CELL_REPLACE_FSCORE = lambda x : r"\fone{" + nbr_to_str_func(x) + "}&"

    formatting_dict_accuracy_end = {
        k.CELL_REPLACE : CELL_REPLACE_ACC_END,
        k.INCLUDE : accuracy_cols_end
    }
    df_format = h.apply_function_with_exclusions(df_format, data_configuration=formatting_dict_accuracy_end)


    formatting_dict_accuracy = {
            k.CELL_REPLACE : CELL_REPLACE_ACC,
            k.INCLUDE : accuracy_cols
        }

    df_format = h.apply_function_with_exclusions(df_format, data_configuration=formatting_dict_accuracy)


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


if __name__ == "__main__":
    # Print the final formatted DataFrame
    print("\n\nETHIX Dataset:")
    format_dataframe(ethix_df)
    print("\n\nUSTV Dataset:")
    format_dataframe(ustv_df)
    mewo = 1



