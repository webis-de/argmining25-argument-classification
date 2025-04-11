# in this file all the different configurations for the definition files are stored
import model_settings as ms
import settings as s
import mongodb.mongo_handler as mdb
from collections import defaultdict
import random

ETHIX = {ms.COLLECTION: ms.ETHIX_SPLIT, ms.DATASET_NAME : ms.ETHIX_SPLIT, ms.SPLIT : ms.TEST, ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES_TOPICS}
USTV =  {ms.COLLECTION : ms.USTV2016_SPLIT, ms.DATASET_NAME : ms.USTV2016_SPLIT, ms.SPLIT : ms.TEST, ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES}

DATASETS_TO_USE = [ ETHIX, USTV]

MULTI_CLASS_ENCODER = {ms.COLLECTION : ms.MULTI_CLASS_ENCODER,
                       ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS,
                       ms.EXPERIMENT_TAG:ms.FEW_SHOT,
                       ms.MODEL_NAME : s.OPEN_AI_MODEL,
                       ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
                       }


MEANS_END_ENCODER = {

    ms.COLLECTION : ms.MEANS_END_ENCODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_ENCODER,
    ms.MODEL_NAME : s.OPEN_AI_MODEL,
    ms.EXPERIMENT_TAG : ms.FEW_SHOT,
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR
}


MEANS_END_PATH_ENCODER= {

    ms.COLLECTION : ms.MEANS_END_PATH_ENCODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_PATH_ENCODER,
    ms.MODEL_NAME : s.OPEN_AI_MODEL,
    ms.EXPERIMENT_TAG : ms.FEW_SHOT,
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR
}


MULTI_CLASS_DECODER = {ms.COLLECTION : ms.MULTI_CLASS_DECODER,
                       ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_DECODER,
                       ms.MODEL_NAME: s.BERT_MODEL,
                       ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
                       }


MEANS_END_PATH_DECODER = {

    ms.COLLECTION : ms.MEANS_END_PATH_DECODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_PATH_DECODER,
    ms.MODEL_NAME : s.BERT_MODEL,
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}



MEANS_END_DECODER = {

    ms.COLLECTION : ms.MEANS_END_DECODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_DECODER,
    ms.MODEL_NAME : s.BERT_MODEL,
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}




MULTI_CLASS_APPROACHES = [MULTI_CLASS_ENCODER, MULTI_CLASS_DECODER]
MEANS_END_APPROACHES = [MEANS_END_ENCODER, MEANS_END_DECODER]
MEANS_END_PATH_APPROACHES = [MEANS_END_PATH_ENCODER, MEANS_END_PATH_DECODER]

def get_models_in_common():

    dataset_argids_dict = {}

    all_approaches_to_be_considered = MULTI_CLASS_APPROACHES + MEANS_END_APPROACHES + MEANS_END_PATH_APPROACHES
    for dataset in DATASETS_TO_USE:
        argument_ids_all_approaches = []
        data_of_approaches = []

        for approach in all_approaches_to_be_considered:
            dataset_copy = dataset.copy()
            approach_copy = approach.copy()
            config = {**dataset_copy, **approach_copy}
            arguments = mdb.get_data_from_mongo(filter_dict=config)
            if len(arguments) == 0:
                print(f"no data for {config}")
                continue
            data_of_approaches.append(arguments)
            approach_argument_ids = [argument[ms.ARGUMENT_ID] for argument in arguments]
            argument_ids_all_approaches.append(approach_argument_ids)
        # get the intersection of all argument ids
        arg_ids_in_common = set(argument_ids_all_approaches[0]).intersection(*argument_ids_all_approaches[1 :])
        if len(arg_ids_in_common) == 0:
            raise ValueError(f"no common argument ids for {dataset[ms.DATASET_NAME]} - GOOD LUCK")

        common_argument_schemes_dict = defaultdict(list)
        for argument in data_of_approaches[0]:
            if argument[ms.ARGUMENT_ID] in arg_ids_in_common:
                argument_id = argument[ms.ARGUMENT_ID]
                scheme = argument[ms.SCHEME]
                common_argument_schemes_dict[scheme].append(argument_id)

        arg_ids_to_consider = []
        # cycle through the designated schemes and select 10 arguments per scheme
        for scheme,ids in common_argument_schemes_dict.items():
            if len(ids) < 10:
                raise ValueError(f"not enough arguments for {scheme} - only {len(ids)}")
            argument_ids_sorted = sorted(ids)
            random.seed(42)
            argument_id_to_use = random.sample(argument_ids_sorted, 10)
            arg_ids_to_consider += argument_id_to_use
        dataset_argids_dict[dataset[ms.DATASET_NAME]] = arg_ids_to_consider
    return dataset_argids_dict

def arguments_to_consider():
    data = mdb.get_data_from_mongo(collection_name=ms.ARGUMENT_IDS_TO_CONSIDER)
    assert len(data) == 1, f"Expected only one entry in {ms.ARGUMENT_IDS_TO_CONSIDER} collection"
    return data[0][ms.ARGUMENT_IDS_TO_CONSIDER]

if __name__ == "__main__":
    # get the argument ids for the datasets
    dataset_argids_dict = get_models_in_common()
    data = {ms.ARGUMENT_IDS_TO_CONSIDER: dataset_argids_dict}


    mdb.upload_data_to_mongo(collection_name=ms.ARGUMENT_IDS_TO_CONSIDER, batch_data=[data],keys_to_check=[ms.ARGUMENT_IDS_TO_CONSIDER])
    print(dataset_argids_dict)

    # go through all employed configs and retrieve the corresponding dataparts

# go through all employed configs and retrieve the corresponding dataparts