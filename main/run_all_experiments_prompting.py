import copy
import random

from loguru import logger

import data_handling.data_loader as dl
import experiments.experiment_means_end_llm as me
import experiments.experiment_multi_class_llm as mc
import experiments.orchestrating as och
import meta_handler as mh
import model_settings as ms
import models.ollama_classify_argument as lla
import models.openai_classify_argument as gpt
import settings as s

# General Config Model for training and getting referenced data


ollama_ref = { ms.META : {ms.MODEL_NAME : s.OLLAMA_MODEL } ,ms._MODELCLASS : lla.OllamaBase}
gpt_ref =  {ms.META : {ms.MODEL_NAME : s.OPEN_AI_MODEL } , ms._MODELCLASS : gpt.OpenAIModel}
llm_models_to_use = [gpt_ref] #, gpt_ref] # definition with the corresponding models to use

#ms.ETHIX_EVALUATION_TEST
datasets_to_use = [ms.USTV2016_EVALUATION_TEST,ms.ETHIX_EVALUATION_TEST] # is meta name to get the desired data
dataset_list = dl.LoadDataForEvaluation(datasets_to_use).create_data_lists()

# ask templates, with the corresponding finetune option we can specify the model
# me_classify_zero = {
#     ms.CONTROL_INSTANCE : me.ASKSchemesClassify,
#     ms.PROMPT_TEMPLATE : "ask-decision-tree-binary-data.txt",
#     ms.META: {
#     ms.COLLECTION : ms.MEANS_END_ENCODER,
#     ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_ENCODER,
#     ms.EXPERIMENT_TAG : ms.ZERO_SHOT,
#     ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
# }}

me_classify_few = {
    ms.CONTROL_INSTANCE : me.ASKSchemesClassify,
    ms.PROMPT_TEMPLATE : ["ask-decision-tree-binary-data.txt", "examples.txt"],  # specify the examples which shall be loaded
    ms.META: {
    ms.COLLECTION : ms.MEANS_END_ENCODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_ENCODER,
    ms.EXPERIMENT_TAG : ms.FEW_SHOT,
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}}

#### ME - PATH CLASSIFICATION

# me_path_zero = {
#     ms.CONTROL_INSTANCE : me.ASKSchemePathAnalysis,
#     ms.PROMPT_TEMPLATE : "ask-decision-tree-binary-data.txt",
#
#     ms.META: {
#     ms.COLLECTION : ms.MEANS_END_PATH_ENCODER,
#     ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_PATH_ENCODER,
#     ms.EXPERIMENT_TAG : ms.ZERO_SHOT,
#     ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
# }}


me_path_few = {

    ms.CONTROL_INSTANCE : me.ASKSchemePathAnalysis,
    ms.PROMPT_TEMPLATE : ["ask-decision-tree-binary-data.txt", "examples.txt"],

    ms.META: {

    ms.COLLECTION : ms.MEANS_END_PATH_ENCODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_PATH_ENCODER,
    ms.EXPERIMENT_TAG : ms.FEW_SHOT,
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}}


### MC - MULTI CLASS CLASSIFICATION

# LLM MULTIC_CLASS_NO DEFINITIONS


# mc_all_schemes_no_definitions_zero = {
#
#     ms.CONTROL_INSTANCE : mc.ControlCalculationArguments,
#     ms.PROMPT_TEMPLATE : "all-schemes-no-definitions.txt",
#
#     ms.META: {
#
#     ms.COLLECTION : ms.MULTI_CLASS_ENCODER,
#     ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_NO_DEFINITIONS,
#     ms.EXPERIMENT_TAG : ms.ZERO_SHOT,
#     ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
# }}

# mc_all_schemes_no_definitions_few = {
#     ms.CONTROL_INSTANCE : mc.ControlCalculationArguments,
#     ms.PROMPT_TEMPLATE : ["all-schemes-no-definitions.txt", "examples.txt"],
#
#     ms.META: {
#
#     ms.COLLECTION : ms.MULTI_CLASS_ENCODER,
#     ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_NO_DEFINITIONS,
#     ms.EXPERIMENT_TAG : ms.FEW_SHOT,
#     ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
# }}
#


# WITH DEFINITIONS

#
# mc_all_schemes_with_definitions_zero = {
#     ms.CONTROL_INSTANCE : mc.ControlCalculationArguments,
#     ms.PROMPT_TEMPLATE : "all-schemes-all-definitions.txt",
#
#     ms.META: {
#
#     ms.COLLECTION : ms.MULTI_CLASS_ENCODER,
#     ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS,
#     ms.EXPERIMENT_TAG : ms.ZERO_SHOT,
#     ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
# }}

mc_all_schemes_with_definitions_few = {
    ms.CONTROL_INSTANCE : mc.ControlCalculationArguments,
    ms.PROMPT_TEMPLATE : ["all-schemes-all-definitions.txt", "examples.txt"],

    ms.META: {

    ms.COLLECTION : ms.MULTI_CLASS_ENCODER,
    ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS,
    ms.EXPERIMENT_TAG : ms.FEW_SHOT,
    ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR,
}}






if __name__ == "__main__":

    random.seed(42)

    # list all the configs that shall be used for the corresponding classification experiment

    # mc_all_schemes_no_definitions_exp_configs = [mc_all_schemes_no_definitions_zero, mc_all_schemes_no_definitions_few]
    # mc_all_schemes_all_definitions_exp_configs = [mc_all_schemes_with_definitions_zero, mc_all_schemes_with_definitions_few]
    # me_classify_exp_configs = [me_classify_zero, me_classify_few]
    # me_paths = [me_path_zero, me_path_few]



    # select the required arguments data combinations
    configs_to_use = [mc_all_schemes_with_definitions_few,me_path_few,me_classify_few]

    experiment_configs_final = []
    for config in configs_to_use :
        for dataset in dataset_list :
            for config in configs_to_use :
                for LLM in llm_models_to_use:

                    # create copy of the original dicts
                    dataset_copy = copy.deepcopy(dataset)
                    LLM_copy = copy.deepcopy(LLM)
                    config_copy = copy.deepcopy(config)

                    if config_copy[ms.PROMPT_TEMPLATE] is not None: # no LLM Needed
                        everything = [dataset_copy, LLM_copy, config_copy]
                    else:
                        everything = [dataset_copy, config_copy]

                    meta = mh.filter_tag_from_meta_fields(everything)
                    meta[ms.GLOBAL_ID] = s.GLOBAL_ID #  specify this id for the corresponding classfication experiments
                    _data = {**LLM_copy, **config_copy}

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
