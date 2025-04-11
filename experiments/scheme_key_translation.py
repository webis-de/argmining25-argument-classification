
import sys
from pathlib import Path

import yaml

import settings as s
import utils.utils as ut


# Load the YAML content into a Python dictionary
def parse_yaml(yaml_file):
    if Path(yaml_file).exists() is False:
        print(f"File {yaml_file} does not exist")
        return None

    with open(yaml_file, 'r') as yl:
        data = yaml.safe_load(yl)
    return data

def get_translation_dict(name, dir=None, style="ref"):
    if name is None:
        return None
    name = name.lower()
    if not name.endswith(".yaml"):
        name = name + ".yaml"
    file_path_full = FILE_PATH / name
    if dir is not None:
        file_path_full = FILE_PATH / dir / name
    data_dict = parse_yaml(file_path_full)[style]
    return data_dict

FILE_PATH = s.PROJECT_ROOT / "key_pair_translation"
REFERENCE_ALL_SCHEMES_LIST = parse_yaml(FILE_PATH / "all_argument_keys_list.yaml")
SCHEMES_TO_EXCLUDE = parse_yaml(FILE_PATH / "schemes_to_exclude.yaml")


# is used to filter out schemes which are not in the reference list
class Prefiltering():
    def __init__(self,dict_with_schemes_to_translate,reference_scheme_file):
        pass
        # GENERAL_GROUPING_DICT_REF = s.GENERAL_GROUPING_DICT_REF
        # check_value_set = s.check_if_set(s.GENERAL_GROUPING_DICT_REF) # check if general grouping reference is set
        # if check_value_set is False:
        #     print("No general grouping reference scheme specified - No Schemes will be grouped")
        #     GENERAL_GROUPING_DICT_REF = None


        # reference_scheme_dict = get_translation_dict(reference_scheme_file, dir="dataset_renaming")
        # self.dict_with_schemes_to_translate = dict_with_schemes_to_translate
        # self.dict_with_general_grouping = None
        
        # if GENERAL_GROUPING_DICT_REF is not None:
        #     self.dict_with_general_grouping = get_translation_dict(GENERAL_GROUPING_DICT_REF, dir="groupings")
        # self.pre_grouping_dict = reference_scheme_dict


    def do_prefiltering(self):
        filtered_out = ut.filter_schemes_out_of_list(self.dict_with_schemes_to_translate, SCHEMES_TO_EXCLUDE)
        renamed = ut.rename_scheme_dict(filtered_out,self.pre_grouping_dict)
        for key in renamed.keys():
            if key not in REFERENCE_ALL_SCHEMES_LIST:
                print(f"Scheme {key} not in reference list of all schemes")
                print("Exiting")
                sys.exit()

        if self.dict_with_general_grouping is not None:
            renamed = ut.rename_scheme_dict(renamed,self.dict_with_general_grouping)
        return range

# input is a key to work with underlying data
class SchemeGroupingInfo():
    def __init__(self,grouping_file_reference=None):
        if grouping_file_reference is None:
            grouping_file_reference = s.GROUPING_OF_SCHEMES_TO_USE

        if grouping_file_reference is None:
            print("No grouping file reference specified")
            sys.exit()

        group_to_scheme_dict = get_translation_dict(grouping_file_reference, dir="groupings")

        self.scheme_to_group_dict = ut.reverse_dict_keys_vals(group_to_scheme_dict)
        self.group_to_scheme_dict = group_to_scheme_dict

        self.schemes_in_use_list = sorted(list(self.scheme_to_group_dict.keys()))
        self.groups_in_use_list = sorted(list(self.group_to_scheme_dict.keys()))

        for key in self.scheme_to_group_dict.keys():
            if key not in REFERENCE_ALL_SCHEMES_LIST:
                print(f"Scheme {key} not in reference list of all schemes")

    def scheme_to_group(self, scheme_dict, kick_out_non_references=True): # translate schemes to groups, schemes not specified are filtered out
        if kick_out_non_references: # kick out schemes not in the grouping reference
            scheme_dict = ut.filter_schemes_out_of_list(scheme_dict, self.schemes_in_use_list, INCLUDE=True)
        group_dict = ut.rename_scheme_dict(scheme_dict,self.scheme_to_group_dict)
        return group_dict

    def group_to_scheme(self,scheme_dict):
        scheme_dict = ut.rename_scheme_dict(scheme_dict,self.group_to_scheme_dict)
        return scheme_dict




