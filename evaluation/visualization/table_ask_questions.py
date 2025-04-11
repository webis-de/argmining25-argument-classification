import settings as s
import pickle
from tabulate import tabulate
import model_settings as ms
import re
import evaluation.visualization.helper as h
import pandas as pd
from collections import defaultdict

# visualize the corresponding entries in a scheme

import model_settings as ms
import evaluation.eval_base_class as eb
import evaluation.eval_filter as ef


#overview of all model files
# import evaluation.visualization.unified_argument_keys as uk

experiment_description = ms.MEANS_END_PATH_ENCODER


import evaluation.visualization.viz_keys as k # use to set keys for data instances

# data is already grouped by scheme
min_nbr = 10
max_nbr = 10

#test_split_data= ef.get_number_of_schemes({ms.INDEX : ms.MEANS_END_PATH_CORRECT_CLASSIFICATION , ms.DATASET : ms.VRAKATSELI_SPLIT, ms.MODEL : s.OPEN_AI_MODEL_FINETUNED})
#test_split_data= ef.get_number_of_schemes({ms.INDEX : ms.MEANS_END_PATH_CORRECT_CLASSIFICATION , ms.DATASET : ms.VRAKATSELI_SPLIT, ms.MODEL : ms.GPT, ms.FINETUNED : True},min_nbr=min_nbr, max_nbr=max_nbr)
#test_split_data= ef.get_number_of_schemes({ms.INDEX : ms.MEANS_END_PATH_CORRECT_CLASSIFICATION , ms.DATASET : ms.VRAKATSELI_SPLIT, ms.MODEL : ms.GPT, ms.FINETUNED : False},min_nbr=min_nbr, max_nbr=max_nbr)
#test_split_data= ef.get_number_of_schemes({ms.INDEX : ms.MEANS_END_PATH_CORRECT_CLASSIFICATION , ms.DATASET : ms.VRAKATSELI_SPLIT, ms.MODEL : ms.LLAMA, ms.FINETUNED : True, ms.SPLIT : ms.DEV },min_nbr=min_nbr, max_nbr=max_nbr)
#test_split_data= ef.get_number_of_schemes({ms.INDEX : ms.MEANS_END_PATH_CORRECT_CLASSIFICATION , ms.DATASET : ms.VRAKATSELI_SPLIT, ms.MODEL : ms.LLAMA, ms.FINETUNED : True, ms.SPLIT : ms.TEST },min_nbr=min_nbr, max_nbr=max_nbr)
test_split_data = ef.get_specified_args_from_database({ms.COLLECTION : ms.MEANS_END_PATH_ENCODER, ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_PATH_ENCODER, ms.DATASET_NAME : ms.ETHIX_EVALUATION_TEST, ms.SPLIT : ms.TEST, ms.EXPERIMENT_TAG : ms.FEW_SHOT})

Eval = eb.EvaluateFeatures()
DATA_TO_USE = Eval.do_evaluation(test_split_data,description=experiment_description)

columns = sorted([int(x) for x in DATA_TO_USE[ms.ASK_TOTAL_EVAL_DICT].keys()])
order_of_columns_to_search = [str(x) for x in columns]


order_of_classification_approaches = [ms.ASK_SCHEME_NODE_DATA_DICT, ms.ASK_TOTAL_EVAL_DICT]

# use of all columns
order_of_schemes = ['example','analogy','alternatives','values','cause to effect','consequences']
order_of_columns_names = [ms.SCHEME] + order_of_columns_to_search #

# idea: get the underlying data for the table by going through all entries
load_underlying_data_config = True
# create existing data configs
all_data_configs_list = []

# we collect for each dataset and each model

data_to_print = []
# create raw set for the table data
table_data = []
for scheme in order_of_schemes:
    current_row = [scheme]
    for column in order_of_columns_to_search:
        try: # get corresponding data for accuracy
            data = DATA_TO_USE[ms.ASK_SCHEME_NODE_DATA_DICT][scheme][ms.EVALUATION][column][ms.ACCURACY] # add the desired data
        except KeyError:
            data = ""
        current_row.append(data)
    table_data.append(current_row)




## FORMATTING - put structure into certain data fields
nbr_to_str_func = lambda x : str(x * 100).split(".")[0]
EMPTY_REPLACE = lambda x : "-"
CELL_REPLACE = lambda x : r"\cell{" + nbr_to_str_func(x) + "}"

# do formatting of needed cells as required for good visualization
formatting_dict = {
    k.CELL_REPLACE : CELL_REPLACE,
    k.EMPTY_REPLACE : EMPTY_REPLACE,
}

# convert list to dataframe, and perform operation on selected points
table_data_df = pd.DataFrame(table_data, columns=order_of_columns_names)
table_data_df.set_index(ms.SCHEME, inplace=True)
table_df_latex = h.apply_function_with_exclusions(table_data_df, data_configuration=formatting_dict)

# specify the positions in the corresponding data dict
row_order_latex = [(ms.SHARE_ANSWER_0,), (ms.TOTAL_0_METRICS, ms.PRECISION,), (ms.TOTAL_0_METRICS, ms.RECALL,), (ms.TOTAL_1_METRICS, ms.PRECISION,), (ms.TOTAL_1_METRICS, ms.RECALL,), (ms.MACRO,ms.F1)]

# add more calculated metrics
addition_table_data = []
for data_entry in row_order_latex:
    current_row = [str(data_entry)]
    for column in order_of_columns_to_search:
        data = DATA_TO_USE[ms.ASK_TOTAL_EVAL_DICT][column]
        for data_key in data_entry: # go into the nested structure according to specification
            data = data[data_key]
        current_row.append(data)
    addition_table_data.append(current_row)

## FORMATTING OF ADDITIONAL DATA
nbr_to_str_func = lambda x : str(x * 100).split(".")[0]
EMPTY_REPLACE = lambda x : "-"
CELL_REPLACE = lambda x : r"\fone{" + nbr_to_str_func(x) + "}"

# do formatting of needed cells as required for good visualization
formatting_dict = {
    k.CELL_REPLACE : CELL_REPLACE,
    k.EMPTY_REPLACE : EMPTY_REPLACE,
    k.EXCLUDE : [((ms.SHARE_ANSWER_0,),"*")]
}

# convert list to dataframe, and perform operation on selected points
additional_data_df = pd.DataFrame(addition_table_data, columns=order_of_columns_names)
additional_data_df.set_index(ms.SCHEME, inplace=True)
additional_df_latex = h.apply_function_with_exclusions(additional_data_df, data_configuration=formatting_dict)

# do conversion of the index
# do formatting of needed cells as required for good visualization
formatting_dict_percentage = {
    k.CELL_REPLACE : lambda  x : f"{x:.2f}\%",
    k.EMPTY_REPLACE : EMPTY_REPLACE,
    k.INCLUDE : [((ms.SHARE_ANSWER_0,),"*")]
}
additional_df_latex = h.apply_function_with_exclusions(additional_df_latex, data_configuration=formatting_dict_percentage)


# adjust the printing according to the specified data format


translation_dict = { # latex translation parts for the keys
(ms.SHARE_ANSWER_0,) : r"\parbox[t]{4em}{Share}  A",
(ms.TOTAL_0_METRICS, ms.PRECISION,) : r"\parbox[t]{4em}{Precision} A",
(ms.TOTAL_0_METRICS, ms.RECALL,) : r"\parbox[t]{4em}{Recall} A",
(ms.TOTAL_1_METRICS, ms.PRECISION,) : r"\parbox[t]{4em}{Precision} B",
(ms.TOTAL_1_METRICS, ms.RECALL,) : r"\parbox[t]{4em}{Recall} B",
(ms.MACRO,ms.F1) : r"$F_1$ \ w-macro"
}
translation_dict.update(**k.SCHEME_TRANSLATION)
# is used for pretty printing and human readability

first_df = pd.DataFrame(table_data)
second_df = pd.DataFrame(addition_table_data)
pd_df = pd.concat([first_df,second_df],axis=0)
print(tabulate(pd_df, floatfmt=".2f", tablefmt="simple",showindex=False,intfmt=",",headers=pd_df.columns))


row_order_latex =  order_of_schemes + [r"\midrule", (ms.SHARE_ANSWER_0,), r"\addlinespace[3pt]", (ms.TOTAL_0_METRICS, ms.PRECISION,), (ms.TOTAL_0_METRICS, ms.RECALL,), r"\addlinespace[4pt]", (ms.TOTAL_1_METRICS, ms.PRECISION,), (ms.TOTAL_1_METRICS, ms.RECALL,), r"\midrule", (ms.MACRO,ms.F1)]


# is used for creating the latex table
pd_df_latex = pd.concat([table_df_latex,additional_df_latex],axis=0)
formatting_dict = {
    k.ROW_ORDER : row_order_latex,
    k.ROW_INDEX_TRANSLATION : translation_dict,
}

h.print_pretty_table(pd_df_latex,config_dict=formatting_dict)




