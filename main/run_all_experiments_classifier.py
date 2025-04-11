import copy
import random

from loguru import logger


import experiments.experiment_multi_class_classifier  as  mcc
import experiments.experiment_means_end_classifier as mec
import experiments.orchestrating as och
import meta_handler as mh
import model_settings as ms
import settings as s
import data_handling.data_loader as dl

me_path_bert = {

    ms.CONTROL_INSTANCE : mec.ASKSchemePathAnalysis,

    ms.META: {
    ms.MODEL_NAME : s.BERT_MODEL,
    ms.COLLECTION : ms.MEANS_END_PATH_DECODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_PATH_DECODER,
    ms.EXPERIMENT_TAG : "",
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}}


me_classify_bert = {

    ms.CONTROL_INSTANCE : mec.ASKSchemesClassify,

    ms.META: {
    ms.MODEL_NAME : s.BERT_MODEL,
    ms.COLLECTION : ms.MEANS_END_DECODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_DECODER,
    ms.EXPERIMENT_TAG : "",
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}}


mc_classify_bert = {

    ms.CONTROL_INSTANCE : mcc.ControlCalculationArguments,

    ms.META: {
    ms.MODEL_NAME : s.BERT_MODEL,
    ms.COLLECTION : ms.MULTI_CLASS_DECODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_DECODER,
    ms.EXPERIMENT_TAG : "",
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}}


datasets_to_use = [ms.USTV2016_EVALUATION_TEST,ms.ETHIX_EVALUATION_TEST] # is meta name to get the desired data
dataset_list = dl.LoadDataForEvaluation(datasets_to_use).create_data_lists()

if __name__ == "__main__" :

    random.seed(42)

    # list all the configs that shall be used for the corresponding classification experiment

    # specification for test purposes
    configs_to_use = [mc_classify_bert,me_path_bert,me_classify_bert]

    experiment_configs_final = []
    for config in configs_to_use :
        for dataset in dataset_list :
            for config in configs_to_use :

                # create copy of the original dicts
                dataset_copy = copy.deepcopy(dataset)
                config_copy = copy.deepcopy(config)


                everything = [dataset_copy, config_copy]

                meta = mh.filter_tag_from_meta_fields(everything)
                meta[ms.GLOBAL_ID] = s.GLOBAL_ID  # specify this id for the corresponding classfication experiments
                _data = {**config_copy}

                experiment_configs_final.append(copy.deepcopy({ms.META : meta,
                                                               ms._DATASETCLASS : dataset_copy,
                                                               ms._DATACLASS : _data
                                                               }))

        random.shuffle(experiment_configs_final)
        logger.info(f"Nbr Exp. Configs: {len(experiment_configs_final)}")
        experiment_class_list = []
        for x in experiment_configs_final :
            instanciator = x[ms._DATACLASS][ms.CONTROL_INSTANCE]
            experiment_class_list.append(instanciator(**x))

        och.Orchestrator(experiment_class_list,
                         additional_fields_check_existence=[
                             ms.DATASET_NAME,
                             ms.SPLIT,
                             ms.SPLIT_IDENTIFIER,
                             ms.MODEL_NAME,
                             ms.EXPERIMENT_DESCRIPTION,
                             ms.EXPERIMENT_TAG,
                             ms.EXPERIMENT_NBR,
                             ms.GLOBAL_ID
                         ])