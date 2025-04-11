import sys

import model_settings as ms

# META FIELDS SPECIFICATION, this information is stored together with the results, and enables an identification of the results
META_FIELDS = [ms.MODEL_NAME, 
               ms.COLLECTION, 
               ms.EXPERIMENT_DESCRIPTION, 
               ms.EXPERIMENT_TAG, 
               ms.EXPERIMENT_NBR, 
               ms.SPLIT,
               ms.SPLIT_IDENTIFIER,
                ms.DATASET_NAME, 
                ms.GLOBAL_ID
                ]

# helper to create meta information on the fly
def filter_tag_from_meta_fields(data_dict_list):
    data_dict_to_return = {}
    used_keys = set()  # Track keys that have been used
    meta_keys = set()
    if not isinstance(data_dict_list, list):
        data_dict_list = [data_dict_list]

    for data_dict in data_dict_list:
        for key, value in data_dict.items():
            if key != ms.META and key in used_keys:
                print(f"Warning: {key} for meta information is already set, please check the models meta information")
            if key == ms.META:
                for kev,val in value.items():
                    if kev not in meta_keys:
                        data_dict_to_return[kev] = val
                        meta_keys.add(kev)
                    else:
                        print(f"Warning: {kev} for meta information is already set, please check the models meta information")    

    return data_dict_to_return

