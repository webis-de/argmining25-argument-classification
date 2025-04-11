import re
from loguru import logger
from tqdm import tqdm
import textwrap
import argu_schemes_definitions.ethix_def.parse_ethix_definition as pw
import model_settings as ms
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification
import torch
import models.train_encoder_only.get_best_trained_models as get_best
import models.train_encoder_only.encoder_only_utils as ecu

class ControlCalculationArguments() :
    def __init__(self, **kwargs) :

        self.Dataset = kwargs[ms._DATASETCLASS]

        # meta data is used fo classifying corresponding data strings
        self.meta = kwargs[ms.META]

        # get corresponding model location for retrieving desired data
        dataset_name = self.meta[ms.DATASET_NAME]  # self.meta[ms.DATASET_NAME]
        model_name = self.meta[ms.MODEL_NAME]

        trained_model_information = get_best.get_best_model_hyperparameters_multi_class(model_name, dataset_name)

        path_with_saved_weights = trained_model_information[ms.PATH_TO_MODEL_WEIGHTS]

        self.model = AutoModelForSequenceClassification.from_pretrained(path_with_saved_weights)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(path_with_saved_weights)
        max_length = self.tokenizer.model_max_length

        self.scheme_to_indices_dict = trained_model_information[ms.SCHEMES_TO_INDICES_DICT]
        self.labels_to_schemes_dict = ecu.reverse_dict(self.scheme_to_indices_dict)



        self.storage_classified_schemes_list = []

    def classify_data(self) :

        for start_number in tqdm(range(1, len(self.Dataset[ms.DATA]) + 1)) :
            data = self.Dataset[ms.DATA][start_number - 1]
            argument = data[ms.ARGUMENT]
            scheme = data[ms.SCHEME]

            scheme_predicted_raw = self.do_single_classification(argument)
            scheme_predicted = self.labels_to_schemes_dict[scheme_predicted_raw]

            data_dict = {ms.ARGUMENT_ID : data[ms.ARGUMENT_ID],
                         ms.SCHEME : scheme,
                         ms.PREDICTION : scheme_predicted,
                         ms.MODEL_RAW_OUTPUT : scheme_predicted_raw,
                         ms.ARGUMENT : argument
                         }
            self.storage_classified_schemes_list.append(data_dict)

    def do_single_classification(self,argument) :

        arg_tokenized = self.tokenizer(argument, padding=True, truncation=True, return_tensors="pt")

        # Move inputs to the same device as the model (optional but recommended)
        input = {key : value.to(self.model.device) for key, value in arg_tokenized.items()}

        with torch.no_grad() :
            outputs = self.model(**input)
            logits = outputs.logits
            predicted_scheme_raw = torch.argmax(logits, dim=1).item()  # get single prediction as int
            return predicted_scheme_raw





