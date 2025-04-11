import copy
import random

import evaluation.eval_filter as ef
import model_settings as ms
import mongodb.mongo_handler as mdb
import settings as s
from collections import defaultdict
import utils.utils as ut

# ensure that the used argument ids in the table are the same for all models, and all evaluations to enable a fair comparison

# these are all possible combinations of the evaluation
multi_class_no_def_zeroshot_model = {ms.COLLECTION : ms.MULTI_CLASS_ENCODER, ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_NO_DEFINITIONS, ms.DATASET_NAME : ms.ETHIX_EVALUATION_TEST, ms.SPLIT : ms.TEST, ms.EXPERIMENT_TAG: ms.ZERO_SHOT, ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR}
#multi_class_all_def_zeroshot_model = { ms.COLLECTION : ms.MULTI_CLASS, ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS, ms.DATASET_NAME : ms.ETHIX_EVALUATION,  ms.SPLIT : ms.TEST, ms.EXPERIMENT_TAG: ms.ZERO_SHOT, ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR}
#means_end_zeroshot_model = {ms.COLLECTION : ms.MEANS_END_TREE_CLASSIFICATION, ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_TREE_CLASSIFICATION, ms.DATASET_NAME : ms.ETHIX_EVALUATION, ms.SPLIT : ms.TEST, ms.EXPERIMENT_TAG: ms.ZERO_SHOT, ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR}

#multi_class_no_def_fewshot_model = { ms.COLLECTION : ms.MULTI_CLASS, ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_NO_DEFINITIONS, ms.DATASET_NAME : ms.ETHIX_EVALUATION,  ms.SPLIT : ms.TEST, ms.EXPERIMENT_TAG: ms.FEW_SHOT, ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR}
#multi_class_all_def_fewshot_model = { ms.COLLECTION : ms.MULTI_CLASS, ms.EXPERIMENT_DESCRIPTION : ms.MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS, ms.DATASET_NAME : ms.ETHIX_EVALUATION, ms.SPLIT : ms.TEST, ms.EXPERIMENT_TAG: ms.FEW_SHOT, ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR}
#means_end_fewshot_model = {ms.COLLECTION : ms.MEANS_END_TREE_CLASSIFICATION, ms.EXPERIMENT_DESCRIPTION : ms.MEANS_END_TREE_CLASSIFICATION, ms.DATASET_NAME : ms.ETHIX_EVALUATION, ms.SPLIT : ms.TEST, ms.EXPERIMENT_TAG: ms.FEW_SHOT, ms.EXPERIMENT_NBR : s.EXPERIMENT_NBR}

# add corresponding bert classifier models

full_list = [
    multi_class_no_def_zeroshot_model,
    #multi_class_all_def_zeroshot_model,
    #means_end_zeroshot_model,

    #multi_class_all_def_fewshot_model,
    #multi_class_no_def_fewshot_model,
   #means_end_fewshot_model,


]

full_arg_common_config_list = []
full_arg_id_list = []

llms = [s.OLLAMA_MODEL]   # [ms.LLAMA, ms.GPT]
for llm in llms:
    for config in full_list:
        copy_config = config.copy()
        copy_config[ms.MODEL_NAME] = llm
        argument_list = mdb.get_data_from_mongo(filter_dict=copy_config)
        argids = []
        for argument in argument_list:
            argids.append(argument[ms.ARGUMENT_ID])
        if len(argids) == 0:
            print(f'No argument found for config: {copy_config}')
            continue
        full_arg_common_config_list.append(copy_config)
        full_arg_id_list.append(argids)

if len(full_arg_common_config_list) == 1:
    COMMON_ARG_IDS_ALL_EXPERIMENTS = set(full_arg_id_list[0])
else:
    COMMON_ARG_IDS_ALL_EXPERIMENTS = set(full_arg_id_list[0]).intersection(*full_arg_id_list[1 :])
    if len(COMMON_ARG_IDS_ALL_EXPERIMENTS) == 0:
        print('No common argument ids - GOOD LUCK')

# select ten arguments foreach scheme
copy_config = full_arg_common_config_list[0]
argument_list = mdb.get_data_from_mongo(filter_dict=copy_config)

argid_dict = defaultdict(list)
for arg in argument_list:
    argid_dict[arg[ms.ARGUMENT_ID]].append(arg) # sort multiple ids, because of multiple

argument_list_unique = [x[0] for x in argid_dict.values()] # remove duplicates
argument_list_unique = sorted(argument_list_unique, key=lambda x: x[ms.ARGUMENT_ID])
argument_list_dict = ut.convert_argument_list_to_schemes_dict(argument_list_unique)


COMMON_ARG_IDS_TO_DISPLAY = []
random.seed(42)
# split these arg ids so ten are displayed for each scheme
for scheme, args in argument_list_dict.items():
    argids = [arg[ms.ARGUMENT_ID] for arg in args if arg[ms.ARGUMENT_ID] in COMMON_ARG_IDS_ALL_EXPERIMENTS]
    if len(argids) < 10:
        print(f'Not enough arg ids Scheme {scheme} - GOOD LUCK')
        COMMON_ARG_IDS_TO_DISPLAY.extend(argids)
    else:
        sample = random.sample(argids, 10)
        COMMON_ARG_IDS_TO_DISPLAY.extend(sample)

