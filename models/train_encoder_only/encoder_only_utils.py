import data_handling.data_loader as dl
from typing import Optional, List, Dict
import settings as s
import model_settings as ms
from collections import defaultdict

ASK_SCHEME_PATHS = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_paths.json")


# for each binary classification node, we need a dict, which tells which decision as this node should be taken for a correct classification
def get_decision_tree_labels(reference_node_name):
    assert isinstance(reference_node_name, str)

    reference_map_dict = {} # at each position the schemes are becoming a binary label, dependent on the current node position
    for scheme, decision_tree_paths in ASK_SCHEME_PATHS.items():
        assert len(decision_tree_paths) == 1
        for path in decision_tree_paths:
            for node_path,decision in path:
                assert isinstance(node_path, str)
                if node_path == reference_node_name:
                    reference_map_dict[scheme] = str(decision) # we see the decision as a label, therefore it is a string
    return reference_map_dict

def get_available_nodes():
    available_nodes = []
    for scheme, paths in ASK_SCHEME_PATHS.items():
        for path in paths:
            for node, decision in path:
                available_nodes.append(node)
    available_nodes = sorted(list(set(available_nodes)))
    return available_nodes

def reverse_dict(d):
    return {v: k for k, v in d.items()}

def flatten_argument_dict(data_dict): # used to flatten keys and values of the argument dict
    final_data_list = []
    for key, value in data_dict.items():
        final_data_list.extend(value)
    return final_data_list

def argument_list_to_train_dev_test_split_dict(argument_list):
    final_dict = {x : [] for x in [ms.TRAIN, ms.DEV, ms.TEST]}
    for argument in argument_list:
        final_dict[argument[ms.SPLIT]].append(argument)
    return final_dict


def get_schemes_to_numbers_dict(argument_list):
    # create standard sequence for schemes tobe ordered
    used_schemes = set([x[ms.SCHEME] for x in argument_list])
    names_sorted = sorted(used_schemes)
    # translate these scheme names to index numbers
    scheme_to_number_dict = {scheme : i for i, scheme in enumerate(names_sorted)}
    return scheme_to_number_dict


def translate_argument_list(arguments, translation_dict):
    for argument_object in arguments:
            argument_object[ms.SCHEME] = translation_dict[argument_object[ms.SCHEME]]
    return arguments


# abstract class for handling the processed data
# the dataset already consists of a split into train, dev and test
def create_dataset_dict(dataset: Optional[Dict] = None,
                       dataset_desc: Optional[str] = None,
                       scheme_to_number_dict: Optional[Dict] = None):

    final_dataset_dict = {}
    for dataset_part_key, value in dataset.items():
        cleaned_dataset_dict = defaultdict(list)
        for argument in value :
            translated_label = scheme_to_number_dict[argument[ms.SCHEME]]
            cleaned_dataset_dict[ms.LABEL].append(translated_label)
            cleaned_dataset_dict[ms.TEXT].append(argument[ms.ARGUMENT])
            # cleaned_dataset_dict[ms.ARGUMENT_ID].append(argument.get(ms.ARGUMENT_ID,None))

        final_dataset_dict[dataset_part_key] = cleaned_dataset_dict

    # assemble needed dict containing the data for training
    output = {}
    output[ms.DATA] = final_dataset_dict
    output[ms.META] =  {ms.DATASET_NAME : dataset_desc }
    output[ms.NBR_LABELS] = len(scheme_to_number_dict)
    if len(scheme_to_number_dict) == 0:
        raise ValueError(f"Scheme to number dict is empty, no data available for training")

    output[ms.SCHEMES_TO_INDICES_DICT] = scheme_to_number_dict
    output[ms.INDICES_TO_SCHEMES_DICT] = reverse_dict(scheme_to_number_dict) if scheme_to_number_dict else None
    return output

