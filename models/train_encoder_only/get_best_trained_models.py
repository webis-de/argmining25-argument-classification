import data_handling.data_loader as dl
import settings as s
from itertools import product
import mongodb.mongo_handler as mdb
import model_settings as ms
import copy
import models.train_encoder_only.encoder_only_utils as eut



available_nodes = eut.get_available_nodes()
dataset_names = [ms.ETHIX_SPLIT,ms.USTV2016_SPLIT]
models = [s.BERT_MODEL]
experiment_nbrs = [s.EXPERIMENT_NBR]



class HyperparameterSelector():
    def __init__(self,filter_dict, new_collection):
        self.filter_dict = filter_dict
        self.collection_to_upload = new_collection
        self.upload_runs()

    #final_result_dataset = { model : {x for x in dataset_names } for model in models }
    def find_best_experiment_config(self,filter_dict):
        # we sort according to sortkey

        available_data = mdb.get_data_from_mongo(filter_dict=filter_dict)
        if len(available_data) == 0:
            raise ValueError(f"No data found for the given filter_dict: {filter_dict}")

        sort_criteria1 = lambda x : x[ms._EVALUATED_DATA][ms.DEV][ms.METRICS_ALL_SCHEMES][ms.MICRO][ms.F1]
        sort_criteria2 = lambda x : x[ms._EVALUATED_DATA][ms.DEV][ms.METRICS_ALL_SCHEMES][ms.MACRO][ms.F1]

        key = lambda x : (
            x[ms._EVALUATED_DATA][ms.DEV][ms.METRICS_ALL_SCHEMES][ms.MICRO][ms.F1],
            x[ms._EVALUATED_DATA][ms.DEV][ms.METRICS_ALL_SCHEMES][ms.MACRO][ms.F1],
        )

        available_data_sorted = sorted(available_data, key=key, reverse=True)
        for result in available_data_sorted:
            x = sort_criteria1(result)
            print(x)

        return available_data_sorted[0]


    def upload_runs(self):

        for dataset_name, model_name, experiment_nbr in product(dataset_names, models, experiment_nbrs):
            query = {
                ms.DATASET_NAME : dataset_name,
                ms.MODEL_NAME : model_name,
                ms.EXPERIMENT_NBR : experiment_nbr,
            }
            query.update(copy.deepcopy(self.filter_dict))
            best_result = self.find_best_experiment_config(query)

            #   # Save the best result to MongoDB
            mdb.upload_data_to_mongo(collection_name=self.collection_to_upload, batch_data=[best_result],
                                     keys_to_check=list(query.keys()))



# these functions are used to obtain
def get_best_model_hyperparameters_means_end(model_name, dataset_name):
    query = {
        ms.MODEL_NAME : model_name,
        ms.DATASET_NAME : dataset_name,
        ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
    }

    best_models = mdb.get_data_from_mongo(collection_name=ms.BEST_HYPERPARAMETERS_MEANS_END_COLLECTION,filter_dict=query)
    if len(best_models) == 0:
        raise ValueError(f"No data found for the given filter_dict: {query}")

    node_data_dict = {}
    for result in best_models:
        node_name = result[ms.NODE]
        if node_name in node_data_dict:
            raise ValueError(f"Duplicate node found: {node_name}. Please check the data.")
        node_data_dict[node_name] = result
    return node_data_dict


def get_best_model_hyperparameters_multi_class(model_name, dataset_name):
    query = {
        ms.MODEL_NAME : model_name,
        ms.DATASET_NAME : dataset_name,
        ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
    }

    best_models = mdb.get_data_from_mongo(collection_name=ms.BEST_HYPERPARAMETERS_MULTI_CLASS_COLLECTION,filter_dict=query)
    if len(best_models) != 1:
        raise ValueError(f"No/Multiple data found for the given filter_dict: {query}, {len(best_models)}")
    return best_models[0]


if __name__ == "__main__":

    for node in available_nodes:
        query_means_end = {
            ms.COLLECTION : ms.MEANS_END_TRAIN_COLLECTION,
            ms.NODE : node }
        HyperparameterSelector(filter_dict=query_means_end, new_collection=ms.BEST_HYPERPARAMETERS_MEANS_END_COLLECTION)


    query_multi_class = {
        ms.COLLECTION : ms.MULTI_CLASS_TRAIN_COLLECTION,
    }
    HyperparameterSelector(filter_dict=query_multi_class, new_collection=ms.BEST_HYPERPARAMETERS_MULTI_CLASS_COLLECTION)

