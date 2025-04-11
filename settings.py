import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
TEMP_DATA_STORAGE = PROJECT_ROOT / "temp_data"
TEMP_DATA_STORAGE.mkdir(exist_ok=True, parents=True)
SEED = 42

OPEN_AI_MODEL = "gpt-4o-mini-2024-07-18"
# OPEN_AI_MODEL_2 = "gpt-4o-2024-08-06"
OLLAMA_MODEL = "gemma:7b" #"llama3.2:3b"
BERT_MODEL = "bert-base-uncased"

# parameters for generating corresponding tokens
TOP_P = 0.1
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 100

# these are the schemes, which are being used for the classification task, the schemes have been retrieved
# they are used to construct the corresponding parts of the decision tree
# configuration of schemes which are used for the classification of the task

OLLAMA_URL = os.getenv("OLLAMA_URL", None)
OPENAI_KEY = os.getenv("OPENAI_KEY", None)
PROJECT_ID = os.getenv("PROJECT_ID", None)
LOCATION = os.getenv("LOCATION", None)


PART_OF_DATASET = os.getenv("PART_OF_DATASET", "")

GROUPING_OF_SCHEMES_TO_USE = os.getenv("GROUPING_OF_SCHEMES_TO_USE", None)
STANDARD_FORMAT_REF = os.getenv("STANDARD_FORMAT_REF", None)

# PATHS for storing related weigths
MODEL_TRAINED_STORAGE_PATH = PROJECT_ROOT / "model_weights"
MODEL_TRAINED_STORAGE_PATH.mkdir(exist_ok=True, parents=True)

# PATHS for storing related weigths
MODEL_TRAINED_EVALUATION_PATH = PROJECT_ROOT / "model_logs"
MODEL_TRAINED_EVALUATION_PATH.mkdir(exist_ok=True, parents=True)



DATA_TO_USE_PATH = PROJECT_ROOT / "data_to_use"
DATA_TO_USE_PATH.mkdir(exist_ok=True, parents=True)

MEANS_END_FEW_SHOT_DATA = "ask-few-shot-samples.json"
MULTI_CLASS_FEW_SHOT_DATA = "multi-class-few-shot-samples.json"


def check_if_set(name):
    set_file_name = os.getenv(name, None)
    if set_file_name is None:
        return False
    if isinstance(set_file_name, str) :
        set_file_name = set_file_name.lower() in ("yes", "true", "t", "1")
    return set_file_name


# specify multi processing
process_tmp = os.getenv("NUM_PROCESSES",1)
NUM_PROCESSES = int(process_tmp)
NBR_EACH_DATALIST = 20


# List with several parameters for the classification task
MONGO_DB_NAME = "Walton-Argument-Classification-Data"
EXPERIMENT_NBR = os.getenv("EXPERIMENT_NBR", 1)
GLOBAL_ID = "2025-04-8-1"
