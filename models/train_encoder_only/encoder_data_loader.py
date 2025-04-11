import copy

import model_settings as ms
import models.train_encoder_only.encoder_only_utils as eut
import settings as s

# # load specified data files for analysis of fully used model
# def prepare_binary_ask_dataset(load_config):
#
#     assert ms.NODE in load_config
#     node_name = load_config[ms.NODE]
#     dataset_description = load_config[ms.DATASET_NAME]
#
#     # global experiment_nbr enables to set up a complete new experiment design
#
#     ethix_combined_train_set = mdb.get_data_from_mongo(load_config)
#
#     # get full data from the database
#     ethix_original_data = mdb.get_data_from_mongo(collection_name=ms.ETHIX_ORIGINAL,
#                                                       filter_dict={ms.SPLIT_IDENTIFIER : s.EXPERIMENT_NBR,
#                                                             })
#
#     # split data into train, dev and test
#     ethix_original_data_split = ecu.argument_list_to_train_dev_test_split_dict(ethix_original_data)
#     ethix_original_data_split[ms.TRAIN] = ethix_combined_train_set # attach the designated (synthetic)  arguments for arguments
#
#     # do splitting according to binary data for classification
#     reference_map_dict = ecu.get_decision_tree_labels(node_name) # at each position the schemes are becoming a binary label, dependent on the current node position
#
#     for split,arguments in ethix_original_data_split.items():
#         for argument in arguments:
#             argument[ms.SCHEME] = reference_map_dict.get(argument[ms.SCHEME], ms.UNDEFINED)
#
#     for split, arguments in ethix_original_data_split.items():
#         arguments_cleaned = [argument for argument in arguments if argument[ms.SCHEME] != ms.UNDEFINED]
#         assert len(arguments_cleaned) == len(arguments), f"Error Undefined {ms.UNDEFINED} label still in arguments, {len(arguments_cleaned)} vs {len(arguments)}"
#
#     # flatten and get corresponding translation key
#     ethix_data_train_dev_test_binary_data_flat = ecu.flatten_argument_dict(ethix_original_data_split)
#     reference_dict = ecu.get_schemes_to_numbers_dict(ethix_data_train_dev_test_binary_data_flat) # create the scheme to number dict for the binary data
#
#
#     dataset_to_use = ecu.create_dataset_dict(dataset=ethix_original_data_split,dataset_desc=dataset_description,
#                                              scheme_to_number_dict=reference_dict)
#
#     return dataset_to_use


def prepare_binary_original_data_only_dataset(node,loaded_dataset_list):
    loaded_dataset_list = copy.deepcopy(loaded_dataset_list)  # make a copy of the loaded dataset list to avoid modifying the original data
    reference_map_dict = eut.get_decision_tree_labels(node)  # get the reference map dict for the current node in the decision tree
    for argument in loaded_dataset_list :
        argument[ms.SCHEME] = reference_map_dict.get(argument[ms.SCHEME], ms.UNDEFINED)

    # remove all arguments which are not defined
    dataset_set_cleaned = [argument for argument in loaded_dataset_list if argument[ms.SCHEME] != ms.UNDEFINED]
    scheme_to_index_dict = eut.get_schemes_to_numbers_dict(dataset_set_cleaned)  # create the scheme to number dict for the binary data

    dataset_train_dev_test_split = eut.argument_list_to_train_dev_test_split_dict(dataset_set_cleaned)

    dataset_description_raw = list(set([x[ms.DATASET_NAME] for x in loaded_dataset_list]))
    if len(dataset_description_raw) != 1 :
        raise ValueError(f"More than one dataset description found: {dataset_description_raw}")
    dataset_description = dataset_description_raw[0]

    dataset_to_use = eut.create_dataset_dict(dataset=dataset_train_dev_test_split, dataset_desc=dataset_description,
                                             scheme_to_number_dict=scheme_to_index_dict)

    return dataset_to_use




# load specified data files for analysis of fully used model
def prepare_multiclass_dataset(loaded_dataset_list):
    # global experiment_nbr enables to set up a complete new experiment design
    loaded_dataset_list = copy.deepcopy(loaded_dataset_list)  # make a copy of the loaded dataset list to avoid modifying the original data

    # split data into train, dev and test
    full_data_split = eut.argument_list_to_train_dev_test_split_dict(loaded_dataset_list)

    # flatten and get corresponding translation key
    reference_dict = eut.get_schemes_to_numbers_dict(loaded_dataset_list) # create the scheme to number dict for the binary data

    dataset_description_raw = list(set([x[ms.DATASET_NAME] for x in loaded_dataset_list]))
    if len(dataset_description_raw) > 1:
        raise ValueError(f"More than one dataset description found: {dataset_description_raw}")

    dataset_description = dataset_description_raw[0]
    dataset_to_use = eut.create_dataset_dict(dataset=full_data_split, dataset_desc=dataset_description,
                                             scheme_to_number_dict=reference_dict)
    return dataset_to_use