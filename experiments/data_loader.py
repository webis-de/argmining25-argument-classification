import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import mongodb.mongo_handler as mdb

import model_settings as ms
import settings as s


file_path_ethix_data_raw = s.DATA_TO_USE_PATH / "ethix_split_data_1.json"

def load_ethix_data(): # load the data directly from the json file, for analytical purposes
    with open(file_path_ethix_data_raw, "r") as f:
        data = json.load(f)
    return data



@dataclass
class Dataset :
    meta: Dict[str, Any]
    data: List


mapppings_of_datasets = {
    ms.ETHIX_EVALUATION_TEST : {
        ms.COLLECTION : ms.ETHIX,
        ms.SPLIT : ms.TEST,
    }
}

# This class serves as inbetween for loading data from mongo db, it splits the dataset into smaller chunks, this happens with the idea that possible multiprocessing can be applied at this point, some time

class LoadDataForEvaluation():
    def __init__(self,data_list_to_load=None):

        if isinstance(data_list_to_load,str):
            data_list_to_load = [data_list_to_load]

        # load the required datasets directly from the mongo db database
        self.datasets = {}

        for instance in data_list_to_load:
            details = mapppings_of_datasets[instance]
            data = mdb.get_data_from_mongo(details)
            self.datasets[instance] = data


    def create_data_lists(self,nbr_each_datalist=None): # key for choosing is the dataset name
        if nbr_each_datalist is None:
            nbr_each_datalist = s.NBR_EACH_DATALIST
        dataset_list_to_return = []

        for dataset_desc, data  in self.datasets.items():
            data_split = self.split_data_list(data,split_size=nbr_each_datalist)
            nbr_single_splits = []
            for single_data_split in data_split:
                nbr_single_splits.append(len(single_data_split))
                meta = {ms.DATASET_NAME : dataset_desc,
                        ms.SPLIT : s.PART_OF_DATASET # include if it used for train, dev, test
                        }
                dataset_chunk = Dataset(meta=meta, data=single_data_split)
                dataset_list_to_return.append(dataset_chunk)
            assert sum(nbr_single_splits) == len(data), "The number of data points in the splits is not the same as the original data"
        return dataset_list_to_return


    def split_data_list(self, argument_list, splitter=None, split_size=None) :
        """
        Splits the argument list into smaller chunks based on either a specified number of splits or a specified split size.

        Parameters:
            argument_list (list): The list to be split.
            splitter (int, optional): Number of splits (if given). Defaults to None.
            split_size (int, optional): The size of each split (if given). Defaults to None.

        Returns:
            list: A list of split lists.
        """
        if splitter is None and split_size is None :
            raise ValueError("Either 'splitter' or 'split_size' must be provided.")

        if splitter is not None and split_size is not None :
            raise ValueError("Provide either 'splitter' or 'split_size', not both.")

        # Split based on the number of splits
        if splitter is not None :
            split_size = len(argument_list) / splitter
            split_size = math.ceil(split_size)
        else:
            splitter = math.ceil(len(argument_list) / split_size)

        # Split based on the size of each split
        return [argument_list[i * split_size : (i + 1) * split_size] for i in
                range(splitter)]






if __name__ == "__main__":
    LoadDataForEvaluation()