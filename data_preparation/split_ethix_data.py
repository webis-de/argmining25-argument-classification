import datetime
import json
import math
import random
import sys

from sklearn.model_selection import train_test_split

import argu_class_data.scheme_translation as skt
import data_handling.data_loader as dl
import model_settings as ms
import mongodb.mongo_handler as mdb
import settings as s
import utils.utils as ut
import argu_class_data.scheme_translation as st


class SplitTopicsSchemes():

    def __init__(self,data_list,relation,nbr_split=1):

        random.seed(42)
        self.data_list = data_list
        self.all_topics = list(set([x[ms.TOPIC] for x in data_list]))
        self.all_schemes = list(set([x[ms.SCHEME] for x in data_list]))

        self.train_ratio = relation[ms.TRAIN]
        self.val_ratio = relation[ms.DEV]
        self.test_ratio = relation[ms.TEST]

        self.train_list = []
        self.dev_list = []
        self.test_list = []

        # split the data list 
        self.scheme_topic_dict = self.create_nested_scheme_dict(self.data_list)
        for scheme, topic_dict in self.scheme_topic_dict.items():
            for topic, arguments in topic_dict.items():
                train_data, val_data, test_data = self.split_list_into_three(arguments,train_ratio=self.train_ratio, val_ratio=self.val_ratio, test_ratio=self.test_ratio)
                self.train_list.extend(train_data)
                self.dev_list.extend(val_data)
                self.test_list.extend(test_data)

        print(f"Train: {len(self.train_list)}")
        self.count_number_arguments_per_scheme(self.train_list)
        print("--------------------------------")
        print(f"Dev: {len(self.dev_list)}")
        self.count_number_arguments_per_scheme(self.dev_list)
        print("--------------------------------")
        print(f"Test: {len(self.test_list)}")
        self.count_number_arguments_per_scheme(self.test_list)

        for x in self.train_list:
            x[ms.SPLIT] = ms.TRAIN
        for x in self.dev_list:
            x[ms.SPLIT] = ms.DEV
        for x in self.test_list:
            x[ms.SPLIT] = ms.TEST

        data = self.train_list + self.dev_list + self.test_list

        # meta information
        meta_info = {
            ms.SPLIT_IDENTIFIER: ms.SPLIT_SCHEMES_TOPICS,
            ms.DATASET_NAME: ms.ETHIX_SPLIT,
            ms.EXPERIMENT_NBR: s.EXPERIMENT_NBR,
            ms.COLLECTION: ms.ETHIX_SPLIT
        }
        # add data to all argument object
        for x in data:
            for key in meta_info.keys():
                x[key] = meta_info[key]

        argument_test_dict = ut.convert_argument_list_to_schemes_dict(self.test_list)
        for scheme, arguments in argument_test_dict.items():
            assert len(arguments) >= 11

        # Save data as JSON
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        output_file = s.DATA_TO_USE_PATH / "final_datasets" / f"ethix_split_data_{nbr_split}_{current_date}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        # upload the data to mongo
        mdb.upload_data_to_mongo(collection_name=ms.ETHIX_SPLIT, batch_data=data, keys_to_check=[ms.ARGUMENT_ID, ms.COLLECTION, ms.EXPERIMENT_NBR, ms.SPLIT_IDENTIFIER])



    def split_list_into_three(self,data_list, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """
        Split a list into three parts with the specified ratios.
        
        Args:
            data_list: The list to split
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.1)
            test_ratio: Proportion for test set (default: 0.2)
            seed: Random seed for reproducibility
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """

        
        # Shuffle the data
        shuffled_data = data_list.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices based on the ratios, ensuring at least 1 item per set
        total_size = len(shuffled_data)

        train_size = math.floor(total_size * train_ratio)
        val_size = math.floor(total_size * val_ratio)
        test_size = math.floor(total_size * test_ratio)


        if total_size < 3:
            print("Not enough data to split into three sets. Putting everything into the training set.")
            return shuffled_data, [], []

        if val_size == 0 and test_size == 0:
            val_size = 1
            test_size = 1
            train_size = train_size - 2

        elif val_size == 0 and test_size > 1:
            val_size = 1
            test_size = test_size - 1

        # Split the data into three parts using list slicing
        train_data = shuffled_data[:train_size]                    # From start to train_size
        val_data = shuffled_data[train_size:train_size + val_size] # From train_size to train_size + val_size
        test_data = shuffled_data[train_size + val_size:]          # From train_size + val_size to end

        assert len(train_data) + len(val_data) + len(test_data) == len(shuffled_data), "Data split sizes do not match original data size."
        return train_data, val_data, test_data


        # split the data lists into topics and schemes
    def _split_arguments_into_topics(self,argument_list):
        topic_dict = {topic : [] for topic in self.all_topics   } # only include topics that are present in the argument list
        for argument in argument_list:
            topic_dict[argument[ms.TOPIC]].append(argument)
        return topic_dict
    
    def _split_arguments_into_schemes(self,argument_list):
        schemes_dict = {scheme : [] for scheme in self.all_schemes} # only include schemes that are present in the argument list
        for argument in argument_list:
            schemes_dict[argument[ms.SCHEME]].append(argument)
        return schemes_dict
    
    def create_nested_scheme_dict(self,argument_list):
        scheme_topic_dict = {}
        tmp_dict = self._split_arguments_into_schemes(argument_list)
        for scheme, arguments in tmp_dict.items():
            scheme_topic_dict[scheme] = self._split_arguments_into_topics(arguments)
        return scheme_topic_dict
    

    def count_number_arguments_per_scheme(self,argument_list):
        scheme_topic_dict = self._split_arguments_into_schemes(argument_list)
        for scheme, arguments in scheme_topic_dict.items():
            print(f"{scheme} - {len(arguments)}")


def do_argument_train_dev_test_splitting(data_list): # split the data into train, dev, and test

    data_to_use = sorted(data_list, key=lambda x: x[ms.ARGUMENT_ID])
    topics_all = list(set([x[ms.TOPIC] for x in data_to_use]))
    schemes_all = list(set([x[ms.SCHEME] for x in data_to_use]))

    def check_topics_and_schemes(data_list,min_nbr=5):
        topics_data_list = (set([x[ms.TOPIC] for x in data_list]))
        schemes_data_list = (set([x[ms.SCHEME] for x in data_list]))
        all_topic_and_schemes = (len(topics_data_list) == len(topics_all)) and (len(schemes_data_list) == len(schemes_all))

        scheme_dict = ut.convert_argument_list_to_schemes_dict(data_list)
        freq_dict = {x:len(scheme_dict[x]) for x in scheme_dict}
        min_nbr_flag = True
        for x in freq_dict.values():
            if x < min_nbr: # check if the schemes occurs at least min_nbr times
                min_nbr_flag = False
                break
        return all_topic_and_schemes and min_nbr_flag

    cnt = 1
    while True:

        print(f"Attempt {cnt}")
        random.shuffle(data_to_use)
        train_data, temp_data = train_test_split(data_to_use, test_size=0.3, random_state=42)
        test_data, dev_data = train_test_split(temp_data, test_size=1/3, random_state=42)
        check_train = check_topics_and_schemes(train_data,35)
        check_dev = check_topics_and_schemes(dev_data,3)
        check_test = check_topics_and_schemes(test_data,11)
        if all([check_train,check_dev,check_test]):
            break
        cnt += 1

    for x in train_data:
        x[ms.SPLIT] = ms.TRAIN
    for x in dev_data:
        x[ms.SPLIT] = ms.DEV
    for x in test_data:
        x[ms.SPLIT] = ms.TEST

    data = train_data + dev_data + test_data

     # meta information
    meta_info = {
        ms.COLLECTION: ms.ETHIX_SPLIT,
        ms.EXPERIMENT_NBR: s.EXPERIMENT_NBR,
        ms.SPLIT_IDENTIFIER: ms.SPLIT_SCHEMES,
        ms.DATASET_NAME: ms.ETHIX_SPLIT,
    }
    # add data to all argument object
    for x in data:
        for key in meta_info.keys():
            x[key] = meta_info[key]

    # Save data as JSON
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = s.DATA_TO_USE_PATH / "final_datasets" / f"ethix_split_data_{ms.SPLIT_SCHEMES}_{current_date}.json"
    output_path = output_file.parent
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    # upload the data to mongo
    mdb.upload_data_to_mongo(collection_name=ms.ETHIX_SPLIT, batch_data=data,
                             keys_to_check=[ms.ARGUMENT_ID, ms.COLLECTION, ms.EXPERIMENT_NBR, ms.SPLIT_IDENTIFIER])


if __name__ == "__main__":

    ARGMINING_SCHEMES =  st.parse_yaml(st.FILE_PATH / "argmining_schemes_to_include.yaml")


    SchemeGroupingInfo = skt.SchemeFormatGroup() # use data from env file
    ethix_data_list = dl.load_ethix_data()

    ethix_data_list = SchemeGroupingInfo.to_standard_format(ethix_data_list) # rename schemes to standard format
    ethix_data_list = SchemeGroupingInfo.scheme_to_group(ethix_data_list)  # do grouping
    ethix_data_list = ut.filter_schemes_out_of_list(ethix_data_list, ARGMINING_SCHEMES, INCLUDE=True)  # remove expert opinion scheme

    SplitTopicsSchemes(ethix_data_list,{ms.TRAIN:0.7,ms.DEV:0.1,ms.TEST:0.2}, nbr_split=s.EXPERIMENT_NBR)
    # do_argument_train_dev_test_splitting(ethix_data_list)

