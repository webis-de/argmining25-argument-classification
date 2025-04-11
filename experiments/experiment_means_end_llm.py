import json
import re
import sys
import textwrap
from loguru import logger
from tqdm import tqdm

import experiments.orchestrating as och
import model_settings as ms
import models.models_base_class as mbc
import models.models_to_use as mtu
import prompt_templates.load_templates as lt
import settings as s
import data_handling.data_loader as dl

# check if active
ASK_SCHEME_QUESTIONS = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_questions.json")
ASK_SCHEME_PATHS = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_paths.json")
start_node = "17"
import mongodb.mongo_handler as mdb
# to do wirte dynamic code to generate the start node

def extract_answer_number(model_text):
    if model_text is None:
        return None
        
    pattern = r'option-(a|b)'
    match = re.search(pattern, model_text.lower()) # searches for the first match
    if match:
        text = match.group()
        if "option-a" in text:
            return 0
        if "option-b" in text:
            return 1
        return None

class ASKExamples():
    def __init__(self,dataset_name=None):

        self.dataset_name = dataset_name
        self.ask_nodes_example_dict = None
        self.get_ask_examples()


    # we adjust code, so that two examples are presented for each scheme
    def get_ask_examples(self):
        ask_example_dict_raw = mdb.get_data_from_mongo(collection_name=ms.EXAMPLES_COLLECTION,
                                                       filter_dict={ms.DATASET_NAME: self.dataset_name,
                                                                    ms.EXPERIMENT_TAG: ms.MEANS_END,
                                                                    })
        assert len(ask_example_dict_raw) == 1
        ask_example_dict = ask_example_dict_raw[0][ms.EXAMPLES]

        final_dict = {}

        single_example_template = textwrap.dedent("""
        Argument: '{argument}'
        Option-A: '{optionA}'
        Option-B: '{optionB}'
        Correct-Answer: '{correct_answer}'
        """)

        # removed topic

        for node_name,node_examples_raw in ask_example_dict.items():
            
            options = ASK_SCHEME_QUESTIONS[node_name]["options"]
            optionA = node_examples_raw["OPTION-A"] # in json keys are always strings
            optionB = node_examples_raw["OPTION-B"]

            examples_list = []
            
            for example in optionA:
                single_node_example = single_example_template.format(argument=example[ms.ARGUMENT].strip(), opic=example.get(ms.TOPIC,"").strip(),optionA=options[0].strip(),optionB = options[1].strip(),correct_answer="Option-A")
                examples_list.append(single_node_example.strip())

            for example in optionB:
                single_node_example = single_example_template.format(argument=example[ms.ARGUMENT].strip(),topic=example.get(ms.TOPIC,"").strip(), optionA=options[0].strip(),optionB = options[1].strip(),correct_answer="Option-B")
                examples_list.append(single_node_example.strip())

            final_dict[node_name] =  "\n\n".join(examples_list)
        self.ask_nodes_example_dict = final_dict        

class ASKSchemesClassify():
    def __init__(self,**kwargs):
        self.LLM = mbc.init_model_prompt_base(**kwargs)
        self.Dataset = kwargs[ms._DATASETCLASS]
        self.meta = kwargs[ms.META]

        self.ask_scheme= ASK_SCHEME_QUESTIONS
        self.ask_example = ASKExamples(self.meta[ms.DATASET_NAME])

        # meta data is used fo classifying corresponding data strings
        self.storage_classified_schemes_list = []

    def classify_data(self):
        for start_number in tqdm(range(1, len(self.Dataset[ms.DATA]) + 1)) :
            data = self.Dataset[ms.DATA][start_number - 1]
            argument = data[ms.ARGUMENT]
            scheme = data[ms.SCHEME]
            topic = data.get(ms.TOPIC,"")
            node_id, answers_raw = self.do_single_classification(argument,topic)
            if node_id is None :
                continue
            data_dict = {ms.ARGUMENT_ID : data[ms.ARGUMENT_ID],
                         ms.SCHEME : scheme,
                         ms.PREDICTION : node_id,
                         ms.MODEL_RAW_OUTPUT : answers_raw}
            self.storage_classified_schemes_list.append(data_dict)

    def do_single_classification(self,argument,topic):
        retrieved_raw_answers = []
        node_id = str(start_node)
        while len(self.ask_scheme[node_id]["children"]) > 0:
            options = self.ask_scheme[node_id]["options"]
            optionA = options[0]
            optionB = options[1]
            examples = self.ask_example.ask_nodes_example_dict[node_id]
            # removed topic
            answer_raw = self.LLM.do_prompt(argument=argument, optionA=optionA, optionB=optionB,examples=examples)
            answer_index = extract_answer_number(answer_raw)
            if answer_index is None:
                return None, retrieved_raw_answers
            node_id = self.ask_scheme[node_id]["children"][answer_index]
            retrieved_raw_answers.append(answer_raw)
        return node_id, retrieved_raw_answers


# do calculation for known paths
class ASKSchemePathAnalysis():
    def __init__(self, **kwargs):
        self.LLM = mbc.init_model_prompt_base(**kwargs)
        self.Dataset = kwargs[ms._DATASETCLASS]
        self.meta = kwargs[ms.META]

        self.ask_scheme = ASK_SCHEME_QUESTIONS
        self.scheme_paths = ASK_SCHEME_PATHS
        self.ask_example = ASKExamples(self.meta[ms.DATASET_NAME])

        # meta data is used fo classifying corresponding data strings
        self.storage_classified_schemes_list = []

    def classify_data(self):
        for start_number in tqdm(range(1, len(self.Dataset[ms.DATA]) + 1)) :
            data = self.Dataset[ms.DATA][start_number - 1]
            argument = data[ms.ARGUMENT]
            topic = data.get(ms.TOPIC,"")
            scheme = data[ms.SCHEME]

            # get all corresponding paths for the classification
            classification_paths_list = []
            existing_scheme_paths = self.scheme_paths[scheme]
            assert len(existing_scheme_paths) == 1
            for idx,classification_path in enumerate(existing_scheme_paths):
                decision_list = []
                broken_classification = False
                for question_node,correct_decision in classification_path:
                    answer_index_raw = self.do_single_classification(argument,topic,question_node)
                    answer_index = extract_answer_number(answer_index_raw)
                    if answer_index is None:
                        print(f"Error: No answer found for question node {question_node}")
                        broken_classification = True

                    data_dict = {ms.NODE : question_node , # slight different notation, so no keys are needed
                                 ms.SCHEME : correct_decision,
                                 ms.PREDICTION : answer_index,
                                 ms.MODEL_RAW_OUTPUT : answer_index_raw}
                    decision_list.append(data_dict)
                    
                if not broken_classification: # only append the data when the classification was successful for the complete path
                    classification_paths_list.append(decision_list)

            data_dict = {ms.ARGUMENT_ID : data[ms.ARGUMENT_ID], ms.SCHEME : scheme, ms.PREDICTION : classification_paths_list }
            self.storage_classified_schemes_list.append(data_dict)

    def do_single_classification(self,argument,topic,node_id):
        options = self.ask_scheme[node_id]["options"]
        optionA = options[0]
        optionB = options[1]
        examples = self.ask_example.ask_nodes_example_dict[node_id]
        # removed topic
        answer_index = self.LLM.do_prompt(argument=argument, optionA=optionA,
                                         optionB=optionB, examples=examples)
        return answer_index





