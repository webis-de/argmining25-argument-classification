import json
import math
import sys
from pathlib import Path

import model_settings as ms
import mongodb.mongo_handler as mdb
import settings as s


# helper for loading json files directly
def load_ethix_data(): # load the data directly from the json file, for analytical purposes
    file_path_ethix_data_raw = s.DATA_TO_USE_PATH / "ethix-dataset.json"
    if not file_path_ethix_data_raw.exists():
        print(f"File {file_path_ethix_data_raw} does not exist, returning empty list")
        return []
    with open(file_path_ethix_data_raw, "r") as f:
        data = json.load(f)
    return data

def load_ustv2016_data(): # load the data directly from the json file, for analytical purposes
    file_path_ethix_data_raw = s.DATA_TO_USE_PATH / "ustv2016-dataset.json"
    if not file_path_ethix_data_raw.exists():
        print(f"File {file_path_ethix_data_raw} does not exist, returning empty list")
        return []
    with open(file_path_ethix_data_raw, "r") as f:
        data = json.load(f)
    return data

# def load_ethix_data_train_dev_test_split(dataset): # load the data directly from the json file, for analytical purposes
#     output_file = s.DATA_TO_USE_PATH / "final_datasets" / dataset
#     if not output_file.exists():
#         print(f"File {output_file} does not exist, returning empty list")
#         return []
#     with open(output_file, "r") as f:
#         data = json.load(f)
#     return data

def load_ethix_artificial_generated_data():
    generated_args_dir = s.DATA_TO_USE_PATH / "generated_arguments"
    all_data = []
    
    if not generated_args_dir.exists():
        print(f"Directory {generated_args_dir} does not exist, returning empty list")
        return []
        
    for json_file in generated_args_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
            
    return all_data

# general function for loading json files
def load_json(file_path: str):
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = s.DATA_TO_USE_PATH / file_path
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        sys.exit(1)

# general function for saving json files
def save_json(file_path: str, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


mapppings_of_datasets = {
    ms.ETHIX_EVALUATION_TEST : {
        ms.COLLECTION : ms.ETHIX_SPLIT,
        ms.SPLIT : ms.TEST,
        ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES_TOPICS,
    },

    ms.USTV2016_EVALUATION_TEST : {
        ms.COLLECTION : ms.USTV2016_SPLIT,
        ms.SPLIT : ms.TEST,
        ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES,
    }
}



# This class serves as inbetween for loading data from mongo db, it splits the dataset into smaller chunks, this happens with the idea that possible multiprocessing can be applied at this point, some time

class LoadDataForEvaluation():
    def __init__(self,data_list_to_load=None):

        if isinstance(data_list_to_load,str):
            data_list_to_load = [data_list_to_load]

        # create required datasets
        self.datasets = {}

        for x in data_list_to_load:
            details = mapppings_of_datasets[x]
            dataset_name = details[ms.COLLECTION]
            data = mdb.get_data_from_mongo(filter_dict=details)
            if len(data) == 0:
                raise ValueError(f"No data found for {x} in MongoDB with filter {details}")
            #data = load_json(details[ms.DATASET_NAME])
            #data = mdb.get_data_from_mongo(filter_dict=details)
            self.datasets[dataset_name] = data

          

    def create_data_lists(self,nbr_each_datalist=None): # key for choosing is the dataset name
        if nbr_each_datalist is None:
            nbr_each_datalist = s.NBR_EACH_DATALIST
        dataset_list_to_return = []

        for dataset_name, data  in self.datasets.items():

            data_split = self.split_data_list(data,split_size=nbr_each_datalist)
            nbr_single_splits = []
            for single_data_split in data_split:
                
                nbr_single_splits.append(len(single_data_split))

                split =list(set([a[ms.SPLIT] for a in single_data_split]))
                if len(split) > 1:
                    raise ValueError(f"The split {split} is not the same for all the data in the split"
                    )
                
                split_nbr =list(set([a[ms.SPLIT_IDENTIFIER] for a in single_data_split]))
                if len(split_nbr) > 1:
                    raise ValueError(f"The split {split} is not the same for all the data in the split"
                    )

                dataset_chunk = {  
                    ms.META : {
                    ms.DATASET_NAME : dataset_name,
                    ms.SPLIT : split[0],
                    ms.SPLIT_IDENTIFIER : split_nbr[0],
                    } ,
                    ms.DATA : single_data_split}
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