import datetime
import json
import math
import random
import sys

from sklearn.model_selection import train_test_split

import data_handling.data_loader as dl
import model_settings as ms
import mongodb.mongo_handler as mdb
import settings as s
import utils.utils as ut
import argu_class_data.scheme_translation as st
import random

def do_argument_train_dev_test_splitting(data_list) :  # split the data into train, dev, and test

    random.seed(42)
    data_to_use = sorted(data_list, key=lambda x : x[ms.ARGUMENT_ID])
    all_data_dict = ut.convert_argument_list_to_schemes_dict(data_to_use)

    train_data_list = []
    dev_data_list = []
    test_data_list = []

    for schemes_args in all_data_dict.values():
        train_data, temp_data = train_test_split(schemes_args, test_size=0.3, random_state=42)
        test_data, dev_data = train_test_split(temp_data, test_size=1/3, random_state=42)


        if len(dev_data) == 0:
            example = random.sample(train_data,1)
            dev_data = [example]
            train_data = [argument for argument in train_data if argument[ms.ARGUMENT_ID] != example[0][ms.ARGUMENT_ID]]

        if len(test_data) < 11:
            difference = 11 - len(test_data)
            new_test_data = random.sample(train_data, difference)
            new_test_set_ids = [argument[ms.ARGUMENT_ID] for argument in new_test_data]
            train_data = [argument for argument in train_data if argument[ms.ARGUMENT_ID] not in new_test_set_ids]
            test_data = test_data + new_test_data

        assert len(train_data) + len(dev_data) + len(test_data) == len(schemes_args), f"Error in splitting data, {len(train_data)} + {len(dev_data)} + {len(test_data)} != {len(schemes_args)}"
        train_data_list.extend(train_data)
        dev_data_list.extend(dev_data)
        test_data_list.extend(test_data)

    for x in train_data_list :
        x[ms.SPLIT] = ms.TRAIN
    for x in dev_data_list :
        x[ms.SPLIT] = ms.DEV
    for x in test_data_list :
        x[ms.SPLIT] = ms.TEST

    data = train_data_list + dev_data_list + test_data_list

    print(f"Train: {len(train_data_list)}")
    print(f"Dev: {len(dev_data_list)}")
    print(f"Test: {len(test_data_list)}")

    # meta information
    meta_info = {
        ms.COLLECTION : ms.USTV2016_SPLIT,
        ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
        ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES,
        ms.DATASET_NAME : ms.USTV2016_SPLIT,
    }
    # add data to all argument object
    for x in data :
        for key in meta_info.keys() :
            x[key] = meta_info[key]

    # Save data as JSON
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = s.DATA_TO_USE_PATH / "final_datasets" / f"ustv2016_split_data_{ms.SPLIT_SCHEMES}_{current_date}.json"
    output_path = output_file.parent
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f :
        json.dump(data, f, indent=4)

    # upload the data to mongo
    mdb.upload_data_to_mongo(collection_name=ms.USTV2016_SPLIT, batch_data=data,
                             keys_to_check=[ms.ARGUMENT_ID, ms.COLLECTION, ms.EXPERIMENT_NBR, ms.SPLIT_IDENTIFIER])


if __name__ == "__main__" :

    ARGMINING_SCHEMES =  st.parse_yaml(st.FILE_PATH / "argmining_schemes_to_include.yaml")

    SchemeGroupingInfo = st.SchemeFormatGroup()  # use data from env file
    ustv2016_data_list = dl.load_ustv2016_data()

    ustv2016_data_list = SchemeGroupingInfo.to_standard_format(ustv2016_data_list)  # rename schemes to standard format
    ustv2016_data_list = SchemeGroupingInfo.scheme_to_group(ustv2016_data_list)  # do grouping
    ustv2016_data_list = ut.filter_schemes_out_of_list(ustv2016_data_list, ARGMINING_SCHEMES, INCLUDE=True)  # remove expert opinion scheme

    do_argument_train_dev_test_splitting(ustv2016_data_list)

