# USE BERT IN
from transformers import TrainingArguments
from datasets import Dataset
from models.train_encoder_only.metrics_creator import EvaluateFeatures
import model_settings as ms
import settings as s
from models.train_encoder_only.train_callback import EncoderCallback


import os

# os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YmJhOTgxMi1hNzlmLTQ5M2ItYjJjMy02ODg1YWM0OGNhNjEifQ=="
# os.environ["NEPTUNE_PROJECT"] = "kmax-tech/argument-scheme-classification"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# your credentials
# do upload of data

# is done outside class to avoid forking issues/warnings
def encode_dataset(dataset_dict, tokenizer) :  # dataset is a list of dictionaries, each resembling several information
    dataset = Dataset.from_dict(dataset_dict)  # create dataset instance for better processing
    max_length = tokenizer.model_max_length
    # dataset_max_length = 0
    #
    # for i in range(len(dataset)) :
    #     i_length = len(dataset[i]["text"])
    #     if i_length > dataset_max_length :
    #         dataset_max_length = i_length
    #
    # dataset_max_length += 10
    # if dataset_max_length > max_length :
    #     print(f"Maximum length of dataset is {dataset_max_length}, max length of tokenizer is {max_length}")
    #     print(f"Truncating dataset to {max_length}")
    #     dataset_max_length = max_length

    def tokenize_function(examples) :
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length ) #, max_length=dataset_max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

class Encoder_Class():
    def __init__(self,encoder_config):

        self.meta = encoder_config[ms.META] # provide new meta information

        self.model_data = encoder_config[ms._DATACLASS]
        self.dataset = encoder_config[ms._DATASETCLASS]

        self.model = self.model_data[ms._MODELCLASS](pretrained_model_name_or_path=self.model_data[ms.MODEL_NAME], num_labels=self.dataset[ms.NBR_LABELS])

        self.tokenizer = self.model_data[ms._TOKENIZER](self.model_data[ms.MODEL_NAME])

        self.train_dataset = encode_dataset(self.dataset[ms.DATA][ms.TRAIN],self.tokenizer)
        self.dev_dataset = encode_dataset(self.dataset[ms.DATA][ms.DEV],self.tokenizer)
        self.test_dataset = encode_dataset(self.dataset[ms.DATA][ms.TEST],self.tokenizer)

        self.compute_metrics = EvaluateFeatures(self.dataset[ms.INDICES_TO_SCHEMES_DICT])

        trainer_class = self.model_data[ms._TRAINER]

        training_args = TrainingArguments(
            run_name=self.meta[ms.MODEL_NAME],
            output_dir=s.MODEL_TRAINED_STORAGE_PATH,  # output directory for model checkpoints
            num_train_epochs=self.meta[ms.EPOCHS_MAX],  # number of training epochs
            per_device_train_batch_size=self.meta[ms.BATCH_SIZE],  # batch size for training
            per_device_eval_batch_size=self.meta[ms.BATCH_SIZE],  # batch size for evaluation
            eval_strategy="epoch",  # evaluate at the end of every epoch
            # save_strategy="epoch",
            # logging_dir=s.MODEL_TRAINED_EVALUATION_PATH,  # directory for storing logs
            # Log training metrics (like loss) every 10 steps during training
            logging_steps=10,
        )

        # Define Trainer
        self.trainer = trainer_class(
            model=self.model,  # the model to train
            args=training_args,  # training arguments
            tokenizer = self.tokenizer,
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.dev_dataset,  # evaluation dataset
            callbacks=[EncoderCallback(self,self.meta)]
        )



    def train_model(self):

        # Train the model
        self.trainer.train()
        # self.trainer.save_model('./final_model')
        # results = self.trainer.evaluate()


