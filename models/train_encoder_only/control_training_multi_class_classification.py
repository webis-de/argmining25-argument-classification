import model_settings as ms
import settings as s
import copy
import meta_handler as mh
from itertools import product
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification, Trainer, RobertaTokenizer, RobertaForSequenceClassification
import models.train_encoder_only.encoder_data_loader as edl
import models.train_encoder_only.model_train_base as mtb
import mongodb.mongo_handler as mdb
from tqdm import tqdm


# parameters for loading required data for multiclass classification
filter_dict_ethix_split = {
                            ms.COLLECTION: ms.ETHIX_SPLIT,
                            ms.SPLIT_IDENTIFIER: ms.SPLIT_SCHEMES_TOPICS,
                            ms.EXPERIMENT_NBR: s.EXPERIMENT_NBR,
                           }

filter_dict_ustv2016_split = {
                            ms.COLLECTION: ms.USTV2016_SPLIT,
                            ms.SPLIT_IDENTIFIER: ms.SPLIT_SCHEMES,
                            ms.EXPERIMENT_NBR: s.EXPERIMENT_NBR,
                           }

#datasets_to_use = [dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ethix_split_data_1_2025-04-14.json")]
datasets_to_use = [
mdb.get_data_from_mongo(filter_dict=filter_dict_ustv2016_split),
mdb.get_data_from_mongo(filter_dict=filter_dict_ethix_split)
]

for x in datasets_to_use:
    if len(x) == 0:
        raise ValueError(f"Dataset {x} is empty, please check the filter_dict and the database.")


# parameters for loading required data for multiclass classification
# filter_dict_ethix_data_multi_class_schemes_unchanged = {
#                             ms.COLLECTION: ms.ETHIX_SPLIT_UNALTERED,
#                             ms.SPLIT_IDENTIFIER: ms.SPLIT_SCHEMES_TOPICS,
#                             ms.EXPERIMENT_NBR: s.EXPERIMENT_NBR,
#                            }


# convention if we want to use a pretrained dataset, we can apply new data
# we use bert, roberta and modern-bert

# config data for models
multiclass_bert = {
    ms.META: {
        ms.COLLECTION: ms.MULTI_CLASS_TRAIN_COLLECTION,
        ms.EXPERIMENT_DESCRIPTION: "multiclass-data",
        ms.EXPERIMENT_TAG : "",
        ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
        ms.MODEL_NAME: s.BERT_MODEL,
    },

    ms._DATACLASS: {
        ms.MODEL_NAME: s.BERT_MODEL,
        ms._MODELCLASS : BertForSequenceClassification.from_pretrained,
        ms._TOKENIZER : BertTokenizer.from_pretrained,
        ms._TRAINER : Trainer,
    }
}


if __name__ == "__main__":

    batch_sizes = [8]
    learning_rates = [3e-5]
    epochs_max = [10]
    experiment_configs = [multiclass_bert]

    experiment_configs_final = []

    # mix the several experiment structures together
    # mix the several experiment structures together
    for bs, lr, epoch, experiment_config, dataset_to_use_config in product(batch_sizes, learning_rates, epochs_max, experiment_configs, datasets_to_use):

        new_parameters = {ms.BATCH_SIZE: bs,
                          ms.LEARNING_RATE : lr,
                          ms.EPOCHS_MAX : epoch,
                          }

        experiment_config_copy = copy.deepcopy(experiment_config)
        # add additional parameters to the experiment config
        experiment_config_copy[ms.META].update(copy.deepcopy(new_parameters))

        #datasets_to_use_list = mdb.get_data_from_mongo(filter_dict=dataset_to_use_config)
        dataset_to_use_copy = copy.deepcopy(dataset_to_use_config)
        dataset_to_use_copy_parsed = edl.prepare_multiclass_dataset(dataset_to_use_copy)

        everything = [experiment_config_copy, dataset_to_use_copy_parsed]

        meta = mh.filter_tag_from_meta_fields(everything)
        meta[ms.GLOBAL_ID] = s.GLOBAL_ID  # specify this id for the corresponding classification experiments

        # update the experiment config with new required data
        experiment_config_copy[ms.META] = meta
        experiment_config_copy[ms._DATASETCLASS] = dataset_to_use_copy_parsed

        experiment_configs_final.append(copy.deepcopy(experiment_config_copy))

    for experiment_config in tqdm(experiment_configs_final):
        meta_config = experiment_config[ms.META]
        exist = mdb.get_data_from_mongo(filter_dict=meta_config)
        if len(exist) > 0:
            tqdm.write(f"Experiment {meta_config[ms.EXPERIMENT_DESCRIPTION]} already exists, skip")
            continue
        TrainClassifier = mtb.Encoder_Class(experiment_config)
        TrainClassifier.train_model()

