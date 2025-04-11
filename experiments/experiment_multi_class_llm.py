
import re

from loguru import logger
from tqdm import tqdm
import textwrap
import argu_schemes_definitions.ethix_def.parse_ethix_definition as pw
import data_handling.data_loader as dl
import experiments.orchestrating as och
import model_settings as ms
import models.models_base_class as mbc
import models.models_to_use as mtu
import prompt_templates.load_templates as lt
import settings as s
import mongodb.mongo_handler as mdb

WaltonSchemesDefObj = pw.ParseSchemeDefinition()

class SchemesExamplesDefinitions():
    def __init__(self,dataset_name=None):

        self.dataset_name = dataset_name
        self.scheme_names_template = None
        self.scheme_examples_template = None
        self.scheme_definitions_template = None

        # init and prepare data for classification
        self.get_scheme_examples()
        self.create_schemes_definitions()


    def create_schemes_definitions(self, ) :
        schemes = WaltonSchemesDefObj.scheme_order_names

        name_list = []
        definition_list = []
        for single_scheme in schemes :
            scheme_info = WaltonSchemesDefObj.schemes_definition_dict[single_scheme]
            definition_list.append(scheme_info[ms.SCHEME].strip())
            name_list.append(scheme_info[ms.SCHEME_ORIG].strip())

        assert len(name_list) == len(definition_list)

        self.scheme_names_template= "\n".join(name_list).strip()
        self.scheme_definitions_template = "\n\n".join(definition_list).strip()

    def get_scheme_examples(self) :
        all_scheme_examples_dict_raw = mdb.get_data_from_mongo(collection_name=ms.EXAMPLES_COLLECTION,
                                                       filter_dict={ms.DATASET_NAME : self.dataset_name,
                                                                    ms.EXPERIMENT_TAG : ms.MULTI_CLASS_ENCODER,
                                                                    })
        assert len(all_scheme_examples_dict_raw) == 1
        all_scheme_examples_dict = all_scheme_examples_dict_raw[0][ms.EXAMPLES]
        final_examples = []

        single_example = textwrap.dedent("""
            Argument: {argument}
            Scheme: {scheme}
            """)
        for scheme in WaltonSchemesDefObj.scheme_order_names:             # get schemes according to order

            if scheme not in all_scheme_examples_dict.keys():
                logger.warning(f"Scheme {scheme} not found in examples.")
                continue

            scheme_examples = all_scheme_examples_dict[scheme] # get example for scheme
            
            for example in scheme_examples:
                argument = example[ms.ARGUMENT] # get example argument
                scheme = example[ms.SCHEME]
                example_formatted = single_example.format(argument=argument.strip(), scheme=scheme.strip())
                example_formatted = example_formatted.strip()
                final_examples.append(example_formatted)
            
        self.scheme_examples_template = "\n".join(final_examples)


    

class ControlCalculationArguments():
    def __init__(self, **kwargs):
        super().__init__()

        self.LLM = mbc.init_model_prompt_base(**kwargs)
        self.Dataset = kwargs[ms._DATASETCLASS]
        self.meta = kwargs[ms.META]

        self.schemes_examples_definitions = SchemesExamplesDefinitions(self.meta[ms.DATASET_NAME])

        # meta data is used fo classifying corresponding data strings
        self.storage_classified_schemes_list = []


    def classify_data(self):
        for start_number in tqdm(range(1, len(self.Dataset[ms.DATA]) + 1)):
            data = self.Dataset[ms.DATA][start_number - 1]
            argument = data[ms.ARGUMENT]
            scheme = data[ms.SCHEME]
            topic = data.get(ms.TOPIC)
            scheme_predicted, answer_raw = self.do_single_classification(argument,topic)
            if scheme_predicted is None:
                continue
            data_dict = {ms.ARGUMENT_ID: data[ms.ARGUMENT_ID],
                         ms.SCHEME: scheme,
                         ms.PREDICTION: scheme_predicted,
                         ms.MODEL_RAW_OUTPUT : answer_raw,
                         ms.ARGUMENT: argument
                         }
            self.storage_classified_schemes_list.append(data_dict)

    def do_single_classification(self,argument,topic):

        schemes_template = self.schemes_examples_definitions.scheme_names_template
        definitions_template = self.schemes_examples_definitions.scheme_definitions_template
        examples_template = self.schemes_examples_definitions.scheme_examples_template
        # removed topic
        answer = self.LLM.do_prompt(argument=argument,schemes=schemes_template,definitions=definitions_template,examples=examples_template)
        answer_scheme = self.match_scheme_group(answer)
        return answer_scheme, answer


    def match_scheme_group(self, model_text, schemes=None) :
        if schemes is None :
            schemes = list(WaltonSchemesDefObj.schemes_definition_names_to_standard_names_dict.keys())
        if model_text is None :
            return None

        pattern = "|".join(schemes)
        match = re.search(pattern, model_text.lower())
        if match :
            text = match.group()
            return WaltonSchemesDefObj.schemes_definition_names_to_standard_names_dict[text]
        return None






