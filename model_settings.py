ARGUMENT = "argument"
ARGUMENT_ID = "argument_id"
SCHEME = "scheme"
SCHEME_ORIG = "scheme_orig"

PREDICTION = "prediction"
GROUND = "ground"

NODE = "node"
NODE_RAW = "node_raw"

MODEL_NAME = "model_name"
_MODELCLASS = "model_class"
MODEL_RAW_OUTPUT = "model_raw_output"
META = "meta"

DATA = "data"

# use of underscore to better differentiate between content field, this keys carry no data which shall be stored directly
_DATACLASS = "_dataclass"
_DATASETCLASS = "_datasetclass"


FINETUNED = "finetuned"
CONTROL_INSTANCE = "control"

CLASSIFIER = "classifier"
LLAMA = "llama"
GPT = "gpt"
SPLIT = "split"
SPLIT_IDENTIFIER = "split_identifier"


COLLECTION = "collection"

# specify the implemented collection types for storing the required data

# Data Split Names

SPLIT_SCHEMES_TOPICS = "split_schemes_topics"
SPLIT_SCHEMES = "split_schemes"

# idea is that the corresponding data objects can be uploaded to elastic
EXPERIMENT_DESCRIPTION = "experiment_description"
EXPERIMENT_TAG = "tag"
EXPERIMENT_NBR = "experiment_nbr"

ANSWER_FUNC = "answer_function"
SCHEME_FUNC = "scheme_function"

# EXPERIMENT NAMES
MEANS_END_ENCODER = "means_end_encoder"
MEANS_END_DECODER = "neans_end_decoder"
MEANS_END_PATH_ENCODER = "means_end_path_encoder"
MEANS_END_PATH_DECODER = "means_end_path_decoder"

MULTI_CLASS_DECODER = "multi_class_classifier"
MULTI_CLASS_ENCODER = "multi_class_encoder"
MULTI_CLASS_ALL_SCHEMES_NO_DEFINITIONS = "multi_class_all_schemes_no_definitions"
MULTI_CLASS_ALL_SCHEMES_ALL_DEFINITIONS = "multi_class_all_schemes_all_definitions"

# preparation of the corresponding data objects

ARGUMENT_LIST = "argument_list"
ARGUMENT_RAW = "argument_raw"
ARGUMENT_FORMAL = "argument_formal"

FILE_ID = "file_id"
ID = "id"
STANCE = "stance"
TOPIC = "topic"
PREMISE_LIST = "premise_list"
CONCLUSION_LIST = "claim_list"
SCHEME_PATHS = "scheme_paths"

PROMPT_STYLE = "prompt_style"
EXAMPLES = "examples"
NO_EXAMPLES = "no_examples"


# dataframe creation
PERCENTAGE = "percentage"
NBR = "nbr"

# LABEL_TEMPLATE
LABEL_TEMP = "label_temp"
NBR_LABELS = "nbr Labels"
SCHEMES_TO_INDICES_DICT = "scheme_to_indices_dict"
INDICES_TO_SCHEMES_DICT = "indices_to_schemes_dict"
# EVALUATION

UNDEFINED = "undefined"
SINGLE = "single_metrics"
TOTAL = "TOTAL"
EVALUATION = "evaluation"

F_BETA = "fbeta05"
F_BETA_UNDEF = "fbeta05_undef"
F1 = "f1"
F1_UNDEF = "f1_undef"
PRECISION = "precision"
PRECISION_UNDEF = "precision_undef"
RECALL = "recall"
RECALL_UNDEF = "recall_undef"
ACCURACY = "accuracy"

GROUND_VALS_NBR = "nbr_ground_vals"
TOTAL_0 = "total_zeros"
TOTAL_1 = "total_ones"

MACRO = "macro"
MICRO = "micro"
WEIGHTED = "weighted"


DATASET_NAME = "dataset_name"
ETHIX = "ethix"
ETHIX_SPLIT = "ethix_split"
ETHIX_ORIGINAL = "ethix_original"
ETHIX_SYNTHETIC = "ethix_synthetic"
ETHIX_SPLIT_UNALTERED = "ethix_split_unaltered" # copy of the original data
ETHIX_EVALUATION_TEST = "ethix_evaluation_test"
USTV2016 = "ustv2016"
USTV2016_SPLIT = "ustv2016_split"
USTV2016_EVALUATION_TEST = "ustv2016_evaluation_test"



PROMPT_TEMPLATE = "prompt_template"

METRICS_ALL_SCHEMES = "metrics_all_schemes"
METRICS_SINGLE_SCHEMES = "metrics_single_schemes"

ASK_SCHEME_NODE_DATA_DICT = "ask_scheme_node_data_dict"
ASK_TOTAL_EVAL_DICT = "ask_total_eval_dict"
TOTAL_0_METRICS = "total_0_metrics"
SHARE_ANSWER_0 = "share_answer_0"

TOTAL_1_METRICS = "total_1_metrics"
SHARE_ANSWER_1 = "share_answer_1"

# Needed for datalaoader
ARGUMENT_FIELD = "argument_field"
GROUPING_OF_SCHEMES_TO_USE = "grouping_of_schemes_to_use"
GENERAL_GROUPING_DICT_REF = "general_grouping_dict_ref"

GLOBAL_ID = "global_id"

ZERO_SHOT = "zero_shot"
FEW_SHOT = "few_shot"

TRAIN = "train"
TEST = "test"
DEV = "dev"


# PARAMETERS FOR FINETUNING CORRESPONDING MODELS
# for getting needed huggingface data
_TOKENIZER = "tokenizer"
_TRAINER = "trainer"
_LOSS_FUNCTION = "loss_function" # used to adjust the loss function, for better weighting of the involved data
_EVALUATED_DATA = "evaluated_data" # used to adjust the loss function, for better weighting of the involved data

MAX_LENGTH = "max_length"
BATCH_SIZE = "batch_sizes"
LEARNING_RATE = "learning_rate"
EPOCHS_MAX = "epochs_max"
EPOCH = "epoch"
F1_AVERAGE = "f1_average"
TRAIN_LOSS = "train_loss"
PATH_TO_MODEL_WEIGHTS = "path_to_model_weights"
PATH_TO_MODEL_WEIGHTS_RELATIVE = "path_to_model_weights_relative"

MEANS_END = "means_end"
TRAINED_MODEL_LOGS = "trained_model_logs"
DICTIONARY_MODEL_LOGS = "dictionary_model_logs"

EXPERIMENT_UID = "experiment_uid"

BINARY_DATASET_FOR_NODE_TEMPLATE = "binary_dataset_for_node_{node}_for_dataset_{dataset"

LABEL = "label"
TEXT = "text"

MEANS_END_TRAIN_COLLECTION = "MEANS_END_TRAINING"  # collection name for the means-end training
MULTI_CLASS_TRAIN_COLLECTION = "MULTICLASS_TRAINING"  # collection name for the multi-class training

BEST_HYPERPARAMETERS_MEANS_END_COLLECTION = "BEST_HYPERPARAMETERS_MEANS_END"
BEST_HYPERPARAMETERS_MULTI_CLASS_COLLECTION = "BEST_HYPERPARAMETERS_MULTI_CLASS"
EXAMPLES_COLLECTION = "EXAMPLES_COLLECTION"





ARGUMENT_IDS_TO_CONSIDER = "argument_ids_to_consider"
# overview of used meta parameters
