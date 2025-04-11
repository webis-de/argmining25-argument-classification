import copy
import random
from collections import defaultdict
from pathlib import Path

import argu_class_data.scheme_translation as kt
import model_settings as ms
import settings as s

random.seed(s.SEED)


def rename_scheme_frequency_dict(existing_dict , dict_with_new_scheme_names , *args):
    if isinstance(dict_with_new_scheme_names, str):
        dict_with_new_scheme_names = kt.get_translation_dict(dict_with_new_scheme_names, *args)

    dict_to_work = defaultdict(int)
    for scheme, arguments in existing_dict.items():
        new_scheme_name = dict_with_new_scheme_names.get(scheme , scheme) # use existing name for doing renaming
        if not isinstance(new_scheme_name,list):
            new_scheme_name = [new_scheme_name]
        for ins in new_scheme_name:
            dict_to_work[ins] += arguments # append argus to multiple instances
        if len(new_scheme_name) > 1:
            print("Duplicating Arguments to new schemes: {} : {} ".format(scheme,','.join(new_scheme_name)))
    return dict_to_work


# convert a list of arguments to a dictionary of schemes
def convert_argument_list_to_schemes_dict(data_list):
    scheme_dict = defaultdict(list)
    for argu in data_list:
        scheme_dict[argu[ms.SCHEME]].append(argu)
    return scheme_dict


# rename the schemes in a dictionary
def rename_schemes_in_argument_list(argument_list, translation_dict):

    argument_list_to_return = []
    for argument in argument_list:

        scheme = argument[ms.SCHEME]
        schemes_for_translation = translation_dict.get(scheme, scheme)
        if schemes_for_translation == scheme:
            print(f"Scheme {scheme} not in translation dict, doing nothing")
        
        # we add in the option that multiple schemes can be translated to a new scheme
        if not isinstance(schemes_for_translation, list): 
            schemes_for_translation = [schemes_for_translation]
            
        for new_scheme_name in schemes_for_translation:
            argument_copy = copy.deepcopy(argument)
            argument_copy[ms.SCHEME] = new_scheme_name
            argument_list_to_return.append(argument_copy)
            
        if len(schemes_for_translation) > 1:
            print("Duplicating Arguments to new schemes: {} : {} ".format(scheme, ','.join(schemes_for_translation)))

    return argument_list_to_return



def join_dicts(dicts_to_join_list,start_dict = None):
    if start_dict is None:
        start_dict = defaultdict(list)
    else:
        start_dict = copy.deepcopy(start_dict)

    if not isinstance(dicts_to_join_list,list):
        dicts_to_join_list = [dicts_to_join_list]
    for entry in dicts_to_join_list:
        for key,values in entry.items():
            if isinstance(values,list):
                start_dict[key].extend(values)
            else:
                start_dict[key].append(values)
    return start_dict


def split_list(lst, size=50) :
    return [lst[i :i + size] for i in range(0, len(lst), size)]


# can also be used to combine multiple frequency dicts
def get_frequency(scheme_args_dict):
    dict_to_return = dict()
    for key, val in scheme_args_dict.items():
        assert isinstance(val,list)
        dict_to_return[key] = len(val)
    return dict_to_return


# filter out samples, which are not corresponding to a specific Walton scheme
def filter_schemes_out_of_list(argument_data_list, argus_filter_list, INCLUDE=False):
    filtered_out = []
    
    argument_data_list_filtered = []
    argus_filter_set = {s.lower() for s in argus_filter_list}  # Upper and lower case is not taken into account
    for argu in argument_data_list:
        scheme = argu[ms.SCHEME]

        scheme_lower = scheme.lower()
        should_filter = scheme_lower in argus_filter_set # test if the scheme is in the filter list

        # Skip if:
        # 1. INCLUDE=True and scheme is NOT in filter list (we only want schemes FROM the list)
        # 2. INCLUDE=False and scheme IS in filter list (we want to exclude schemes IN the list)
        if (INCLUDE and not should_filter) or (not INCLUDE and should_filter):
            print(f"Filtering out scheme: {scheme}")
            filtered_out.append(scheme)
            continue  

        argument_data_list_filtered.append(argu)

    if not INCLUDE: # check if all schemes in the filter list are in the filtered out list
        for x in argus_filter_list:
            if x not in filtered_out:
                print(f"Scheme {x} could not be referenced for filtering out.")

    print(f"Filtered out {len(filtered_out)} samples")
    return argument_data_list_filtered

# reverse keys and values in dictionary, used for mapping
def reverse_dict_keys_vals(dict_with_keys):
    dict_with_keys_copy = copy.deepcopy(dict_with_keys)
    reverse_dict = dict()
    for key,val in dict_with_keys_copy.items():
        if not isinstance(val,list):
            val = [val]
        for x in val:
            ref = key
            if x in reverse_dict.keys(): # for case of multiple grouping
                old_ref = reverse_dict[x]
                if not isinstance(old_ref,list):
                    old_ref = [old_ref]
                old_ref.append(key)
                ref = old_ref

            reverse_dict[x] = ref
    return reverse_dict

def is_absolute_path(path):
    """
    Check if the given path is an absolute path using pathlib.
    
    Args:
        path (str or Path): The path to check
        
    Returns:
        bool: True if the path is absolute, False otherwise
    """
    path_obj = Path(path) if isinstance(path, str) else path
    return path_obj.is_absolute()





