import copy
import random
import sys

import data_handling.data_loader as dl
import models.train_encoder_only.encoder_only_utils as eut
import mongodb.mongo_handler as mdb
import model_settings as ms
import settings as s
random.seed(42)

# check the keys of the data
# we specify the needed data instances
# is needed to determine which schemes are needed for additional generation

# load and if necessary generate the required data for training

language_model_to_use_for_creation_of_dataset = s.OLLAMA_MODEL

class LoadGenerateData():
    def __init__(self,decision_tree_node_number):

        self.decision_tree_node_number = str(decision_tree_node_number)

        self.train_data_split_nbr = s.EXPERIMENT_NBR

        self.original_data_list = mdb.get_data_from_mongo(collection_name=ms.ETHIX_ORIGINAL, filter_dict = {ms.SPLIT_IDENTIFIER : s.EXPERIMENT_NBR, ms.SPLIT : ms.TRAIN}) # load the training data, from the mondodb database
        # self.original_data_list =  dl.load_ethix_data_train_dev_test_split("ethix_split_data_1_2025-04-10.json")
        # self.original_data_list = [x for x in self.original_data_list if x[ms.SPLIT] == ms.TRAIN] # load the training data

        # get overview of all available schemes and topics for all existing splits
        self.all_schemes = list(set([argument[ms.SCHEME] for argument in self.original_data_list]))
        self.all_topics = list(set([argument[ms.TOPIC] for argument in self.original_data_list]))


        # (collection_name=s.ETHIX_ORIGINAL, filter_dict={ms.SPLIT: ms.TRAIN, ms.DATA_SPLIT_NBR : self.train_data_split})# load the specific train data, from the mondodb database
        #self.generated_data_list = mdb.get_data_from_mongo(collection_name=s.ETHIX_SYNTHETIC)

        self.generated_data_list = mdb.get_data_from_mongo(collection_name=ms.ETHIX_SYNTHETIC, filter_dict = {ms.SPLIT_IDENTIFIER : s.EXPERIMENT_NBR, ms.MODEL_NAME : language_model_to_use_for_creation_of_dataset})

        # if there is no synthetic data, we create a generic dict for schemes and topics
        if len(self.generated_data_list) > 0:
            self.generated_data_scheme_topic_dict = self.create_nested_scheme_dict(self.generated_data_list)
        else:
            self.generated_data_scheme_topic_dict = {scheme : {topic : [] for topic in self.all_topics} for scheme in self.all_schemes}

        self.number_of_args_per_classification_step = 500 # determines the balanced classes

        reference_map_dict = ask_utils.get_decision_tree_labels(self.decision_tree_node_number) # get the reference map dict for the current node in the decision tree
        for argument in self.original_data_list:
            argument[ms.LABEL_TEMP] = reference_map_dict.get(argument[ms.SCHEME], ms.UNDEFINED) 

        # sort the data according to the two possible answer options
        self.option1_scheme_data = [argument for argument in self.original_data_list if argument[ms.LABEL_TEMP] == "0"]
        self.option2_scheme_data = [argument for argument in self.original_data_list if argument[ms.LABEL_TEMP] == "1"]
                
        self.scheme_topic_node_dict = None
        self.scheme_topic_all_data_dict_reference = None # copy the original data for reference
        self.scheme_single_option_dict = None
        self.node_arguments_data_list = []

        self._argument_list_with_args_to_be_generated = None
        self.number_of_args_to_be_generated = 0


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
    
    def count_number_total_arguments(self,scheme_topic_dict):
        number_total_arguments = 0
        for _, topic_dict in scheme_topic_dict.items():
            for _, arguments in topic_dict.items():
                number_total_arguments += len(arguments)
        return number_total_arguments

    # when we have a full augmented dataset, we can use this one for the best creation of the mode parameters
    def upload_train_data_to_mongo(self):

        if self.number_of_args_to_be_generated > 0 :
            print(
                f"Cannot continue, because there are still {self.number_of_args_to_be_generated} arguments to be generated")
            sys.exit()

        node_argument_list_tmp = []
        for node_data in self.node_arguments_data_list :
            # flatten the nested dict
            for scheme, topic_dict in node_data.items() :
                for topic, arguments in topic_dict.items() :
                    for argument in arguments :
                        node_argument_list_tmp.append(argument)

        self.node_arguments_data_list = node_argument_list_tmp

        for argument in self.node_arguments_data_list:
            argument[ms.SPLIT] = ms.TRAIN
            argument[ms.SPLIT_IDENTIFIER] = self.train_data_split_nbr
            argument[ms.NODE] = self.decision_tree_node_number
            argument[ms.MODEL_NAME] = language_model_to_use_for_creation_of_dataset

        mdb.upload_data_to_mongo(collection_name=ms.ETHIX_TRAINING, batch_data =self.node_arguments_data_list, keys_to_check=[ms.ARGUMENT, ms.NODE, ms.MODEL_NAME, ms.SPLIT_IDENTIFIER], filter_dict={ms.SPLIT: ms.TRAIN, ms.SPLIT_IDENTIFIER : self.train_data_split_nbr}) # upload the data to the mongodb database

        # dl.save_json(
        #     s.DATA_TO_USE_PATH / "training_data" / f"arguments_for_train_node_{self.decision_tree_node_number}_train_split_{self.train_data_split_nbr}.json",
        #     node_argument_list_tmp)


    def identify_missing_arguments(self):
        for idx, data in enumerate([self.option1_scheme_data,self.option2_scheme_data]):
            self._argument_list_with_args_to_be_generated = []
            self.create_full_dataset(data)

            self.node_arguments_data_list.append(self.scheme_single_option_dict)

            if len(self._argument_list_with_args_to_be_generated) > 0:
                print(f"Have to generate {len(self._argument_list_with_args_to_be_generated)} missing arguments")

                # save the arguments to be generated
                dl.save_json(s.DATA_TO_USE_PATH / "generated_arguments_raw" / f"arguments_to_be_generated_node_{self.decision_tree_node_number}_train_split_{self.train_data_split_nbr}_answer{idx}.json", self._argument_list_with_args_to_be_generated)

                self.number_of_args_to_be_generated += len(self._argument_list_with_args_to_be_generated)
                self._argument_list_with_args_to_be_generated = []




    # create balanced dataset
    # collect distributed number of arguments per scheme and topic, if not missing args are being generated
    # start with empty balanced dataset
  
    def create_full_dataset(self,data_list_to_use):

        schemes_to_use = list(set([argument[ms.SCHEME] for argument in data_list_to_use])) # a node can only be trained for particular schemes, for other schemes the data is not available and therefore not defined

        self.scheme_single_option_dict = {scheme : {topic : [] for topic in self.all_topics} for scheme in schemes_to_use}
        self.scheme_topic_node_dict = self.create_nested_scheme_dict(copy.deepcopy(data_list_to_use)) # generate nested scheme dict from the data list to use, this can only 
        # include data for particular schemes, based on the current node in the decision tree
        self.scheme_topic_all_data_dict_reference = self.create_nested_scheme_dict(copy.deepcopy(self.original_data_list)) # copy of all available train data, which is not synthetic
        
        nbr_in_final_dataset = 0

        while nbr_in_final_dataset < self.number_of_args_per_classification_step:
            self.create_balanced_dataset_one_run()
            nbr_in_final_dataset = self.count_number_total_arguments(self.scheme_single_option_dict)
            print(f"Number of arguments in final dataset: {nbr_in_final_dataset}")

    def create_balanced_dataset_one_run(self): # do a single run to collect the data for all schemes and topics
         
        for scheme, topic_dict in self.scheme_single_option_dict.items():
            for topic, _ in topic_dict.items():
                argument = self.get_argument_from_original_data(scheme,topic)
                if argument is None:
                    argument = self.get_argument_from_synthetic_data(scheme,topic) # get argument from synthetic data
                if argument is None:
                    argument = self.set_synthetic_argument_generation(scheme,topic) # we have to generate a new synthetic argument
                self.scheme_single_option_dict[scheme][topic].append(argument)


    def get_argument_from_original_data(self,scheme,topic):
        arguments_for_topic = self.scheme_topic_node_dict[scheme]
        if topic not in arguments_for_topic:
            print(f"Topic {topic} not in arguments for scheme {scheme}")
            return None
        arguments = arguments_for_topic[topic]
        if len(arguments) > 0:
            return arguments.pop(random.randint(0, len(arguments) - 1))
        else:
            return None
        
    def get_argument_from_synthetic_data(self,scheme,topic):
        arguments_for_topic = self.generated_data_scheme_topic_dict[scheme]
        if topic not in arguments_for_topic:
            print(f"Topic {topic} not in arguments for scheme {scheme}")
            return None 
        arguments = arguments_for_topic[topic]
        if len(arguments) > 0:
            return arguments.pop(random.randint(0, len(arguments) - 1))
        else:
            return None

    def set_synthetic_argument_generation(self,scheme,topic):

        new_topic = self.find_new_topic(scheme,topic) # the argument shall be generated for a new topic, but for the same scheme
        arguments_to_select = self.scheme_topic_all_data_dict_reference[scheme][new_topic]
        
        argument = arguments_to_select[random.randint(0, len(arguments_to_select) - 1)]
        self._argument_list_with_args_to_be_generated.append({ms.SCHEME : scheme, "argument" : argument[ms.ARGUMENT], "topic": topic,  "new_topic" : topic})
        return None


    def find_new_topic(self,scheme,topic):
        while True:
            foreign_topic = random.choice(self.all_topics)
            if foreign_topic != topic and len(self.scheme_topic_all_data_dict_reference[scheme][foreign_topic]) > 0:
                return foreign_topic


    # basic idea is that labels of the data have to be new arranged in order to allow for training, at each position an new labeling is required


if __name__ == "__main__":
    # do the data generation for all needed and required models
    load_data = LoadGenerateData("17")
    load_data.identify_missing_arguments()

    # general data creation for all nodes
    # for node_id, node_info in node_infos.items():
    #     if len(node_info.children) == 0:
    #         continue

    #     # do the data generation for all needed and required models
    #     load_data = LoadGenerateData(node_id)
    #     load_data.generate_missing_arguments()




