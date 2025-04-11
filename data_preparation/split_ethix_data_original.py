import datetime
import json
import math
import random

from sklearn.model_selection import train_test_split

import argu_class_data.scheme_translation as skt
import data_handling.data_loader as dl
import model_settings as ms
import mongodb.mongo_handler as mdb
import settings as s
import utils.utils as ut

def do_argument_train_dev_test_splitting(data_list) :  # split the data into train, dev, and test

    data_to_use = sorted(data_list, key=lambda x : x[ms.ARGUMENT_ID])
    topics_all = list(set([x[ms.TOPIC] for x in data_to_use]))
    schemes_all = list(set([x[ms.SCHEME] for x in data_to_use]))

    def check_topics_and_schemes(data_list, min_nbr=5) :
        topics_data_list = (set([x[ms.TOPIC] for x in data_list]))
        schemes_data_list = (set([x[ms.SCHEME] for x in data_list]))
        all_topic_and_schemes = (len(topics_data_list) == len(topics_all)) and (
                    len(schemes_data_list) == len(schemes_all))

        scheme_dict = ut.convert_argument_list_to_schemes_dict(data_list)
        freq_dict = {x : len(scheme_dict[x]) for x in scheme_dict}
        min_nbr_flag = True
        for x in freq_dict.values() :
            if x < min_nbr :  # check if the schemes occurs at least min_nbr times
                min_nbr_flag = False
                break
        return all_topic_and_schemes and min_nbr_flag

    cnt = 1
    while True :

        print(f"Attempt {cnt}")
        random.shuffle(data_to_use)
        train_data, temp_data = train_test_split(data_to_use, test_size=0.3, random_state=42)
        test_data, dev_data = train_test_split(temp_data, test_size=1 / 3, random_state=42)
        check_train = check_topics_and_schemes(train_data, 35)
        check_dev = check_topics_and_schemes(dev_data, 3)
        check_test = check_topics_and_schemes(test_data, 11)
        if all([check_train, check_dev, check_test]) :
            break
        cnt += 1

    for x in train_data :
        x[ms.SPLIT] = ms.TRAIN
    for x in dev_data :
        x[ms.SPLIT] = ms.DEV
    for x in test_data :
        x[ms.SPLIT] = ms.TEST

    data = train_data + dev_data + test_data

    # meta information
    meta_info = {
        ms.COLLECTION : ms.ETHIX_SPLIT_UNALTERED,
        ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
        ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES,
        ms.DATASET_NAME : ms.ETHIX_SPLIT_UNALTERED,
    }
    # add data to all argument object
    for x in data :
        for key in meta_info.keys() :
            x[key] = meta_info[key]

    # Save data as JSON
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = s.DATA_TO_USE_PATH / "final_datasets" / f"ethix_split_data_unaltered_{ms.SPLIT_SCHEMES}_{current_date}.json"
    with open(output_file, 'w', encoding='utf-8') as f :
        json.dump(data, f, indent=4)

    # upload the data to mongo
    mdb.upload_data_to_mongo(collection_name=ms.ETHIX_SPLIT_UNALTERED, batch_data=data,
                             keys_to_check=[ms.ARGUMENT_ID, ms.COLLECTION, ms.EXPERIMENT_NBR, ms.SPLIT_IDENTIFIER])


if __name__ == "__main__" :
    SchemeGroupingInfo = skt.SchemeFormatGroup()  # use data from env file
    ethix_data_list = dl.load_ethix_data()

    ethix_data_list = SchemeGroupingInfo.to_standard_format(ethix_data_list)  # rename schemes to standard format
    do_argument_train_dev_test_splitting(ethix_data_list)

