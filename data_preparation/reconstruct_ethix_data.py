import copy
import json
import random

from loguru import logger
from tqdm import tqdm

import experiments.orchestrating as och
import model_settings as ms
import models.ollama_classify_argument as lla
import models.models_base_class as mbc
import settings as s

ethix_data_cleaned_path = s.DATA_TO_USE_PATH / 'ethix-dataset-cleaned.json'
ethix_prompt_template_path = s.PROJECT_ROOT / "data_preparation" / "prompt_template_ethix_completion.txt"

def load_ethix_data(): # load the data directly from the json file
    with open(ethix_data_cleaned_path, 'r') as file:
        data = json.load(file)
    return data

class ArgumentReconstruction():
    def __init__(self, **kwargs) :

        self.LLM = mbc.init_model_prompt_base(**kwargs[ms.MODEL_NAME])
        self.Dataset = kwargs[ms.DATASET]
        self.answer_func = self.filter_answer_of_llm

        # meta data is used fo classifying corresponding data strings
        self.meta = kwargs[ms.META]
        self.meta.update(self.LLM.meta)
        self.meta.update(self.Dataset.meta)
        self.storage_classified_schemes_list = []

    def filter_answer_of_llm(self,answer_from_model) :
        try:
            preprocess = answer_from_model.strip()
            parts = preprocess.split('Part-')
            parts = [part.strip() for part in parts if part]  # Clean empty entries

            # Extract each part's content separately
            premise = parts[0].split(':', 1)[1].strip(" []")  # Get text after 'Part-1:'
            if ";" in premise:
                premise = premise.split(";")

            conclusion = parts[1].split(':', 1)[1].strip(" []")  # Get text after 'Part-2

            full_string = f"Premise: {premise}\nConclusion: {conclusion}"

            return {ms.PREMISE_LIST:premise, ms.CONCLUSION_LIST : conclusion, ms.ARGUMENT : full_string}

        except Exception as e:
            logger.error(f"Error in filter_answer_of_llm: {e}")
            return None

    def classify_data(self):
        for start_number in tqdm(range(1, len(self.Dataset.argument_list) + 1)) :
            data = self.Dataset.argument_list[start_number - 1]
            argument = data[ms.ARGUMENT]
            topic = data[ms.TOPIC]
            scheme = data[ms.SCHEME]
            reconstructed_argument, answers_raw = self.do_single_classification(input_argument=argument,input_topic=topic)
            if reconstructed_argument is None :
                continue
            data_dict = {ms.ARGUMENT_ID : data[ms.ARGUMENT_ID],
                         ms.ARGUMENT_RAW : argument,
                         ms.SCHEME : scheme,
                         ms.ARGUMENT : reconstructed_argument,
                         ms.MODEL_RAW : answers_raw}
            self.storage_classified_schemes_list.append(data_dict)

    def do_single_classification(self,**kwargs):
        answer = self.LLM.do_prompt(**kwargs)
        answer_scheme = self.answer_func(answer)
        return answer_scheme, answer



def do_argument_reconstruction():
    random.seed(42)

    data_to_use = load_ethix_data()
    with open(ethix_prompt_template_path, "r") as file:
        vrakatseli_prompt_template = file.read()

    models_to_use = [lla.OllamaModel,]
    language_model_list = [{ms.PROMPT_TEMPLATE : vrakatseli_prompt_template, ms.MODEL_NAME : model} for model in models_to_use]

    meta = {
        ms.COLLECTION : ms.ARGUMENT_RECONSTRUCTION,
        ms.EXPERIMENT_DESCRIPTION : "Ethix Dataset Reconstruction",
        ms.EXPERIMENT_TAG : "",
        ms.NBR_EXPERIMENT : 1,
    }

    experiment_configs = []
    for LLM in language_model_list :
        for dataset in data_to_use :
            experiment_configs.append(
                copy.deepcopy({ms.META : meta, ms.MODEL_NAME : LLM, ms.DATASET : dataset,
                 }))
    random.shuffle(experiment_configs)
    logger.info(f"Nbr Exp. Configs: {len(experiment_configs)}")

    experiment_class_list = []
    for i,x in enumerate(experiment_configs):
        experiment_class_list.append(ArgumentReconstruction(**x))
    och.Orchestrator(experiment_class_list, additional_fields_check_existence=[ms.MODEL_NAME])


if __name__ == "__main__":
    do_argument_reconstruction()