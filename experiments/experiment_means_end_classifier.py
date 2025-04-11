
from tqdm import tqdm
import experiments.orchestrating as och
import model_settings as ms
import json
import settings as s
import data_handling.data_loader as dl
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification
import torch
import models.train_encoder_only.get_best_trained_models as get_best
import models.train_encoder_only.encoder_only_utils as ecu
# check if active
ASK_SCHEME_GROUP_QUESTIONS_DICT = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_questions.json")
GROUP_ARGUMENTATION_KEYS_REF_DICT = dl.load_json(s.DATA_TO_USE_PATH / "final_datasets" / "ask_scheme_paths.json")
start_node = "17"

# implement own way of classifying necessary data
class ASKSchemesClassify() :
    def __init__(self, **kwargs) :

        self.Dataset = kwargs[ms._DATASETCLASS]

        self.ask_scheme = ASK_SCHEME_GROUP_QUESTIONS_DICT
        self.scheme_paths = GROUP_ARGUMENTATION_KEYS_REF_DICT

        self.meta = kwargs[ms.META]

        # get corresponding model location for retrieving desired data
        dataset_name = self.meta[ms.DATASET_NAME] # self.meta[ms.DATASET_NAME]
        model_name = self.meta[ms.MODEL_NAME]
        self.dictionary_model_logs = get_best.get_best_model_hyperparameters_means_end(model_name, dataset_name)

        self.storage_classified_schemes_list = []

    def classify_data(self) :
        for start_number in tqdm(range(1, len(self.Dataset[ms.DATA]) + 1)) :
            data = self.Dataset[ms.DATA][start_number - 1]
            argument = data[ms.ARGUMENT]
            scheme = data[ms.SCHEME]
            node_id, answers_raw = self.do_single_argument_classification(argument)
            if node_id is None :
                continue
            data_dict = {ms.ARGUMENT_ID : data[ms.ARGUMENT_ID],
                         ms.SCHEME : scheme,
                         ms.PREDICTION : node_id,
                         ms.MODEL_RAW_OUTPUT : answers_raw}
            self.storage_classified_schemes_list.append(data_dict)


    def do_single_argument_classification(self, argument) :
        retrieved_raw_answers = []
        node_id = str(start_node)
        while len(self.ask_scheme[node_id]["children"]) > 0 :
            answer_index, answer_formatted = self.do_single_node_classification(node_id, argument)
            if answer_index is None :
                return None, retrieved_raw_answers
            node_id = self.ask_scheme[node_id]["children"][answer_index]
            retrieved_raw_answers.append(answer_formatted)
        return node_id, retrieved_raw_answers

    def do_single_node_classification(self, node, argument) :

        trained_model_information = self.dictionary_model_logs[node]
        path_with_saved_weights = trained_model_information[ms.PATH_TO_MODEL_WEIGHTS]

        model = AutoModelForSequenceClassification.from_pretrained(path_with_saved_weights)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(path_with_saved_weights)
        max_length = tokenizer.model_max_length

        scheme_to_indices_dict =  trained_model_information[ms.SCHEMES_TO_INDICES_DICT]
        labels_to_schemes_dict =  ecu.reverse_dict(scheme_to_indices_dict)

        arg_tokenized = tokenizer(argument, padding=True, truncation=True, return_tensors="pt")

        # Move inputs to the same device as the model (optional but recommended)
        input = {key : value.to(model.device) for key, value in arg_tokenized.items()}

        with torch.no_grad() :
            outputs = model(**input)
            logits = outputs.logits
            predicted_scheme = torch.argmax(logits, dim=1).item()  # get single prediction as int
            assert predicted_scheme == scheme_to_indices_dict[labels_to_schemes_dict[predicted_scheme]]
            return predicted_scheme, labels_to_schemes_dict[predicted_scheme]

# do calculation for known paths
class ASKSchemePathAnalysis() :
    def __init__(self, **kwargs) :

        self.Dataset = kwargs[ms._DATASETCLASS]

        self.ask_scheme = ASK_SCHEME_GROUP_QUESTIONS_DICT
        self.scheme_paths = GROUP_ARGUMENTATION_KEYS_REF_DICT

        self.meta = kwargs[ms.META]

        dataset_name = self.meta[ms.DATASET_NAME] # self.meta[ms.DATASET_NAME]
        model_name = self.meta[ms.MODEL_NAME]
        self.dictionary_model_logs = get_best.get_best_model_hyperparameters_means_end(model_name, dataset_name)

        self.storage_classified_schemes_list = []

    def classify_data(self) :
        for start_number in tqdm(range(1, len(self.Dataset[ms.DATA]) + 1)) :
            data = self.Dataset[ms.DATA][start_number - 1]
            argument = data[ms.ARGUMENT]
            scheme = data[ms.SCHEME]

            # get all corresponding paths for the classification
            classification_paths_list = []
            existing_scheme_paths = self.scheme_paths[scheme]
            assert len(existing_scheme_paths) == 1
            for idx, classification_path in enumerate(existing_scheme_paths) :
                decision_list = []
                broken_classification = False
                for question_node, correct_decision in classification_path :
                    answer_index,answer_formatted = self.do_single_node_classification(question_node,argument)
                    if answer_index is None :
                        print(f"Error: No answer found for question node {question_node}")
                        broken_classification = True
                    assert answer_formatted == str(answer_index)
                    data_dict = {ms.NODE : question_node,  # slight different notation, so no keys are needed
                                 ms.SCHEME : int(correct_decision),
                                 ms.PREDICTION : int(answer_formatted),
                                 ms.MODEL_RAW_OUTPUT : answer_formatted}
                    decision_list.append(data_dict)

                if not broken_classification :  # only append the data when the classification was successful for the complete path
                    classification_paths_list.append(decision_list)

            data_dict = {ms.ARGUMENT_ID : data[ms.ARGUMENT_ID], ms.SCHEME : scheme,
                         ms.PREDICTION : classification_paths_list}
            self.storage_classified_schemes_list.append(data_dict)

    def do_single_node_classification(self, node, argument) :

        trained_model_information = self.dictionary_model_logs[node]
        path_with_saved_weights = trained_model_information[ms.PATH_TO_MODEL_WEIGHTS]

        model = AutoModelForSequenceClassification.from_pretrained(path_with_saved_weights)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(path_with_saved_weights)
        max_length = tokenizer.model_max_length

        scheme_to_indices_dict = trained_model_information[ms.SCHEMES_TO_INDICES_DICT]
        labels_to_schemes_dict = ecu.reverse_dict(scheme_to_indices_dict)

        arg_tokenized = tokenizer(argument, padding=True, truncation=True, return_tensors="pt")

        # Move inputs to the same device as the model (optional but recommended)
        input = {key : value.to(model.device) for key, value in arg_tokenized.items()}

        with torch.no_grad() :
            outputs = model(**input)
            logits = outputs.logits
            predicted_scheme = torch.argmax(logits, dim=1).item()  # get single prediction as int
            assert predicted_scheme == scheme_to_indices_dict[labels_to_schemes_dict[predicted_scheme]]
            return predicted_scheme, labels_to_schemes_dict[predicted_scheme]






