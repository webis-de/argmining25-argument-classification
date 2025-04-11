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

FILE_PATH = s.PROJECT_ROOT / "argu_class_data"
REFERENCE_ALL_SCHEMES_LIST = parse_yaml(FILE_PATH / "all_argument_keys_list.yaml")
SCHEMES_TO_EXCLUDE = parse_yaml(FILE_PATH / "schemes_to_exclude.yaml")


# this dict is used to translate the schemes in the dataset
def get_standard_format_translation_dict():
    reference_standardize_dict = {}
    existing_files_path = FILE_PATH / "dataset_renaming"
    for file in existing_files_path.iterdir():
        translation_dict = parse_yaml(file)["ref"]
        for key,name in translation_dict.items():
            if name not in REFERENCE_ALL_SCHEMES_LIST:
                print(f"Scheme {key} not in reference list of all schemes")
                print("Exiting")
                sys.exit()
            if key in reference_standardize_dict:
                existing_name = reference_standardize_dict[key]
                if name != existing_name:
                    print(f"Scheme {key} already exists in standardization dict with different name: {existing_name} vs {name}")
                    print("Exiting")
                    sys.exit()
            reference_standardize_dict[key] = name
    return reference_standardize_dict



# input is a key to work with underlying data
class SchemeFormatGroup():
    def __init__(self,standard_format_reference = None, grouping_file_reference=None):

        if grouping_file_reference is None:
            grouping_file_reference = s.GROUPING_OF_SCHEMES_TO_USE
            print(f"Grouping file reference from settings: {grouping_file_reference}")
            if grouping_file_reference is None:
                print("No grouping file reference specified")
                sys.exit()

        self.standard_format_reference = standard_format_reference

        self.group_to_scheme_dict  = get_translation_dict(grouping_file_reference, dir="groupings")
        self.scheme_to_group_dict = ut.reverse_dict_keys_vals(self.group_to_scheme_dict)
       
        self.schemes_in_use_list = sorted(list(self.scheme_to_group_dict.keys()))
        self.groups_in_use_list = sorted(list(self.group_to_scheme_dict.keys()))

        for key in self.scheme_to_group_dict.keys(): # check if all schemes, which are being translated, are in the reference list
            if key not in REFERENCE_ALL_SCHEMES_LIST:
                print(f"Scheme {key} not in reference list of all schemes")


    def to_standard_format(self, argument_list):
        if self.standard_format_reference is None:
            print("Standard format reference not specified, using default")
            translation_dict = get_standard_format_translation_dict()
        else:
            translation_dict = get_translation_dict(self.standard_format_reference, dir="dataset_renaming")

        renamed_arg_list = ut.rename_schemes_in_argument_list(argument_list,translation_dict)
        return renamed_arg_list


    def scheme_to_group(self, argument_list): # translate schemes to groups, schemes not specified are filtered out
        renamed_arg_list = ut.rename_schemes_in_argument_list(argument_list,self.scheme_to_group_dict)
        return renamed_arg_list


    def group_to_scheme(self,argument_list):
        renamed_arg_list = ut.rename_schemes_in_argument_list(argument_list,self.group_to_scheme_dict)
        return renamed_arg_list




