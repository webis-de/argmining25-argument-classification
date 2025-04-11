# do the data loading process for existing multiclass approaches
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification
import torch
import data_handling.data_loader as dl
import model_settings as ms
import models.train_encoder_only.model_train_base as mtb
from torch.utils.data import DataLoader, Dataset
import torch




load_config = dl.load_json("/Users/max/work/projects/argmining25-argument-classification/model_weights/log_a2ec4e2a-3b39-48a9-aad1-1fd0096227ce.json")

path_with_saved_weights = load_config[ms.PATH_TO_MODEL_WEIGHTS]

model = AutoModelForSequenceClassification.from_pretrained(path_with_saved_weights)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(path_with_saved_weights)
max_length = tokenizer.model_max_length

dataset = load_config[ms._DATASETCLASS]
labels_to_schemes = dataset[ms.INDICES_TO_SCHEMES_DICT]
train_dataset = dataset[ms.DATA][ms.TRAIN]["text"]
train_dataset_label = dataset[ms.DATA][ms.TRAIN]["label"]

dev_dataset = dataset[ms.DATA][ms.DEV]["text"]
dev_dataset_label = dataset[ms.DATA][ms.DEV]["label"]

test_dataset = dataset[ms.DATA][ms.TEST]["text"]
test_dataset_label = dataset[ms.DATA][ms.TEST]["label"]


dev_tokenized_input = tokenizer(dev_dataset, padding=True, truncation=True, return_tensors="pt")

# Inference
with torch.no_grad():
        outputs = model(**dev_tokenized_input)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
        pred_schemes_full = [labels_to_schemes[str(pred)] for pred in predictions]

orig_dev = load_config[ms.DATA][ms.DEV][ms.PREDICTION]

assert pred_schemes_full == orig_dev

predictions_for = []

for instance in dev_dataset :
    # Tokenize a single instance
    inputs = tokenizer(instance, padding=True, truncation=True, return_tensors="pt")

    # Move inputs to the same device as the model (optional but recommended)
    inputs = {key : value.to(model.device) for key, value in inputs.items()}

    # Inference
    with torch.no_grad() :
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()  # get single prediction as int
        predictions_for.append(pred)

predictions_for_schemes = [labels_to_schemes[str(pred)] for pred in predictions_for]
assert predictions_for_schemes == orig_dev

mewo = 1