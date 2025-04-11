
import data_analysis.analyze_utils as at
import pandas as pd
import data_handling.data_loader as dl
import argu_schemes_definitions.ethix_def.parse_ethix_definition as pw

import argu_class_data.scheme_translation as skt
import ask_decision_tree.viz_analyze_visser as decision_tree_visser
import evaluation.visualization.helper as h
import evaluation.visualization.viz_keys as k
import model_settings as ms
import mongodb.mongo_handler as mdb
import settings as s
import utils.utils as ut
# print data about the underlying vrakataseli dataset
# load data, including information about the ASK paths

WaltonSchemesDefObj = pw.ParseSchemeDefinition()
SchemeGroupingInfo = skt.SchemeFormatGroup()

min_number = None
max_number = None

ethix_data = mdb.get_data_from_mongo(collection_name=ms.ETHIX_SPLIT,filter_dict={ms.SPLIT_IDENTIFIER: ms.SPLIT_SCHEMES_TOPICS})
ethix_data_dict = ut.convert_argument_list_to_schemes_dict(ethix_data)
ethix_df = at.create_pandas_df(ethix_data_dict, name="Ethix")
ethix_df = at.set_min_max_dataframes(ethix_df, min_number=min_number, max_number=max_number)

ustv_data = mdb.get_data_from_mongo(collection_name=ms.USTV2016_SPLIT,filter_dict={ms.SPLIT_IDENTIFIER: ms.SPLIT_SCHEMES})
ustv_data_dict = ut.convert_argument_list_to_schemes_dict(ustv_data)
ustv_df = at.create_pandas_df(ustv_data_dict, name="USTV")
ustv_df = at.set_min_max_dataframes(ustv_df, min_number=min_number, max_number=max_number)

combined_data = ethix_data + ustv_data
combined_data_dict = ut.convert_argument_list_to_schemes_dict(combined_data)
combined_df = at.create_pandas_df(combined_data_dict, name="Combined")
combined_df = at.set_min_max_dataframes(combined_df, min_number=min_number, max_number=max_number)

# we work with corresponding table data heads
schemes_to_display = ethix_df[ms.SCHEME].tolist()
walton_numbering = WaltonSchemesDefObj.walton_nbr_to_scheme_dict

# order how to print schemes
# Convert string values to float with ',' as decimal separator
ordered_data = sorted(walton_numbering.items(), key=lambda x: float(x[1].replace('.', '').replace(',', '.')))
# Create ordered dictionary
scheme_to_display_order_names = list({k: v for k, v in ordered_data}.keys())


data_frames_to_use = [ethix_df, ustv_df, combined_df]

ASK_SCHEME_QUESTIONS_DICT = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_questions.json")
ASK_SCHEME_PATHS_DICT = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_paths.json")


order_of_columns_names = [ms.SCHEME, ms.NBR, "PathLength"]
for df in data_frames_to_use:
    dataset_name = list(set(df[ms.DATASET_NAME].to_list()))[0]
    order_of_columns_names.extend([f"{dataset_name}-Nbr", f"{dataset_name}-Percentage"])

table_body = []
for scheme in scheme_to_display_order_names:
    if scheme not in schemes_to_display:
        print(f"Scheme {scheme} not in schemes to display")
        continue
    current_row = [scheme]
    scheme_walton_nbr = WaltonSchemesDefObj.schemes_definition_dict[scheme][ms.NBR]

    # assert that only path for classification is existing
    scheme_paths = ASK_SCHEME_PATHS_DICT[scheme]
    assert len(scheme_paths) == 1
    scheme_path_length = len(scheme_paths[0])
    current_row.extend([scheme_walton_nbr, scheme_path_length])  # add walton number and path length

    for df in data_frames_to_use:
        arguments_nbr = df[df[ms.SCHEME] == scheme][ms.NBR].values[0]
        arguments_percentage = df[df[ms.SCHEME] == scheme][ms.PERCENTAGE].values[0]
        current_row.extend([arguments_nbr, arguments_percentage])
    table_body.append(current_row)


# add final row, which contains the sum of all arguments frequencies from all considered datasets
last_row = ["$\Sigma$", "", ""]
for df in data_frames_to_use:
    total_argument = df[ms.NBR].sum()
    total_percentage = df[ms.PERCENTAGE].sum()
    last_row.extend([total_argument, ""])

table_body.append(last_row)

def format_nbr(x):
    if isinstance(x, float):
        return f"{x:.1f}"
    return str(x)

# do formatting of needed cells as required for good visualization
formatting_dict = {
    k.CELL_REPLACE : format_nbr,
}

data_df = pd.DataFrame(table_body, columns=order_of_columns_names)
data_df.set_index(ms.SCHEME, inplace=True)
df_latex = h.apply_function_with_exclusions(data_df, data_configuration=formatting_dict)

pretty_table = h.print_pretty_table(df_latex, )