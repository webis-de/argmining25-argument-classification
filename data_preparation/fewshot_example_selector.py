import random
import model_settings as ms
import ask_decision_tree.viz_analyze_visser as ask_tree_visser
import settings as s
from collections import defaultdict
import data_handling.data_loader as dl
import mongodb.mongo_handler as mdb
import utils.utils as ut
import json

ASK_SCHEME_PATHS_DICT = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_paths.json")
ASK_SCHEME_QUESTIONS_DICT = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_questions.json")


def create_means_end_few_shot_data(data_list_to_parse,dataset_name=None):
    random.seed(42)

    data_dict_to_parse = ut.convert_argument_list_to_schemes_dict(data_list_to_parse) # convert data list to dictionary with schemes as keys
    # sort arguments according to positive and negative answers to the specified question
    node_questions_training_dict = defaultdict(lambda : {"OPTION-A" : [], "OPTION-B" : []})
    for scheme, arguments in data_dict_to_parse.items() :
        scheme_paths = ASK_SCHEME_PATHS_DICT[scheme]
        assert len(scheme_paths) == 1
        scheme_path = scheme_paths[0]

        for tuple in scheme_path :
            node_id = tuple[0]
            correct_answer = tuple[1]
            if correct_answer == 0:
                correct_answer = "OPTION-A"
            elif correct_answer == 1:
                correct_answer = "OPTION-B"
            else:
                raise ValueError(f"Unknown answer {correct_answer} for node {node_id}")
            node_questions_training_dict[node_id][correct_answer].extend(arguments)


    example_me_data_dict = {}
    for scheme, pos_neg in node_questions_training_dict.items() :
        first_options_examples = random.sample(pos_neg["OPTION-A"],2)
        second_option_examples = random.sample(pos_neg["OPTION-B"],2)

        example_me_data_dict[scheme] = {"OPTION-A" : first_options_examples, "OPTION-B": second_option_examples}

    # file_name_path = s.DATA_TO_USE_PATH / s.MEANS_END_FEW_SHOT_DATA
    # with open(file_name_path, "w") as f :
    #     f.write(json.dumps(example_me_data_dict, indent=4))

    example_config = {ms.COLLECTION : ms.EXAMPLES_COLLECTION,
                 ms.EXPERIMENT_TAG : ms.MEANS_END,
                 ms.DATASET_NAME : dataset_name,
                 ms.EXAMPLES : example_me_data_dict, }

    mdb.upload_data_to_mongo(collection_name=ms.EXAMPLES_COLLECTION, batch_data=[example_config],
                             keys_to_check=[ms.COLLECTION, ms.EXPERIMENT_TAG, ms.DATASET_NAME])




# select two random arguments for each scheme
def create_multi_class_few_shot_data(data_list_to_parse,dataset_name=None):
    random.seed(42)

    data_dict_to_parse = ut.convert_argument_list_to_schemes_dict(data_list_to_parse) # convert data list to dictionary with schemes as keys
    
    example_mc_data_dict = dict()
    for scheme, arguments in data_dict_to_parse.items() :
        arguments = random.sample(arguments, 2)
        example_mc_data_dict[scheme] = arguments

    example_config = { ms.COLLECTION : ms.EXAMPLES_COLLECTION,
    ms.EXPERIMENT_TAG : ms.MULTI_CLASS_ENCODER,
    ms.DATASET_NAME : dataset_name,
    ms.EXAMPLES : example_mc_data_dict,}

    mdb.upload_data_to_mongo(collection_name=ms.EXAMPLES_COLLECTION, batch_data=[example_config],
                             keys_to_check=[ms.COLLECTION, ms.EXPERIMENT_TAG, ms.DATASET_NAME])
    # file_name_path = s.DATA_TO_USE_PATH / s.MULTI_CLASS_FEW_SHOT_DATA
    # with open(file_name_path, "w") as f :
    #     f.write(json.dumps(example_mc_data_dict, indent=4))


if __name__ == "__main__" :
    # load the underlying datasets
    ethix_split_data = mdb.get_data_from_mongo(filter_dict={ms.COLLECTION : ms.ETHIX_SPLIT, ms.SPLIT : ms.TRAIN, ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES_TOPICS})
    ustv_split_data = mdb.get_data_from_mongo(filter_dict={ms.COLLECTION : ms.USTV2016_SPLIT, ms.SPLIT : ms.TRAIN, ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES})

    create_means_end_few_shot_data(ethix_split_data,ms.ETHIX_SPLIT)
    create_multi_class_few_shot_data(ethix_split_data,ms.ETHIX_SPLIT)

    create_means_end_few_shot_data(ustv_split_data,ms.USTV2016_SPLIT)
    create_multi_class_few_shot_data(ustv_split_data,ms.USTV2016_SPLIT)
