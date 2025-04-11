from transformers import TrainerCallback, TrainerState, TrainerControl
import model_settings as ms
from pathlib import Path
import uuid
import settings as s
import models.train_encoder_only.save_train_data_handler as stdh
import  mongodb.mongo_handler as mdb
import json
class EncoderCallback(TrainerCallback) :
    def __init__(self, trainer_instance,meta) : # meta is designated to contain the needed data points for a better evaluation
        self.trainer_instance = trainer_instance
        self.meta = meta
        #super().__init__()  # Call parent class init

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs) :
        print("Starting training")
        for key in self.meta:
            print(f"{key}: {self.meta[key]}")

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) :

        print(f"Epoch {state.epoch} ended.")

        # Log epoch and loss
        current_epoch = state.epoch
        train_loss = state.log_history[-1]["loss"] if state.log_history and "loss" in state.log_history[-1] else None
        if train_loss is not None:
            train_loss = round(train_loss, 5)

        data_dict = {ms.EPOCH : current_epoch, ms.TRAIN_LOSS : train_loss}
        data_dict.update(self.trainer_instance.meta)

        predictions_raw_dev = self.trainer_instance.trainer.predict(self.trainer_instance.dev_dataset)
        metrics_dev = self.trainer_instance.compute_metrics.compute_metrics(predictions_raw_dev)
        
        f1_score_dev = metrics_dev[ms.METRICS_ALL_SCHEMES][ms.MACRO][ms.F1]
        print(f"DEV Epoch {current_epoch} F1 MACRO score: {f1_score_dev}")

        predictions_raw_test = self.trainer_instance.trainer.predict(self.trainer_instance.test_dataset)
        metrics_test = self.trainer_instance.compute_metrics.compute_metrics(predictions_raw_test)
        f1_score_test = metrics_test[ms.METRICS_ALL_SCHEMES][ms.MACRO][ms.F1]
        print(f"Test Epoch {current_epoch} F1 MACRO score: {f1_score_test}")

        data_dict.update({ms._EVALUATED_DATA : {ms.DEV : metrics_dev, ms.TEST : metrics_test},
                          ms.SCHEMES_TO_INDICES_DICT : self.trainer_instance.dataset[ms.SCHEMES_TO_INDICES_DICT],
                          }) # save the requested data

        # Systematic model saving
        collection = self.trainer_instance.meta.get(ms.COLLECTION, "unknown_collection")
        if collection == "unknown_collection" :
            print("No collection name provided.")
        model_name = self.trainer_instance.meta.get(ms.MODEL_NAME, "unknown_model")
        if model_name == "unknown_model" :
            print("No model name provided.")
        experiment_nbr = self.trainer_instance.meta.get(ms.EXPERIMENT_NBR, "unknown_experiment")
        if experiment_nbr == "unknown_experiment" :
            print("No experiment number provided.")

        experiment_uid = str(uuid.uuid4())  # e.g., '0b57b3b4-56d0-4c5e-a4f9-9f0fba1a8c3d'
        save_dir = s.MODEL_TRAINED_STORAGE_PATH /f"{collection}" /f"{model_name}_experiment_nbr_{experiment_nbr}" / f"uid_{experiment_uid}"
        save_dir_relative = Path(f"{collection}") / f"{model_name}_experiment_nbr_{experiment_nbr}" / f"uid_{experiment_uid}"

        if save_dir.exists():
            raise FileExistsError(f"Directory {save_dir} already exists. Please choose a different name.")

        save_dir.mkdir(parents=True)

        self.trainer_instance.trainer.model.save_pretrained(save_dir)
        self.trainer_instance.trainer.tokenizer.save_pretrained(save_dir)

        data_dict.update({ms.PATH_TO_MODEL_WEIGHTS : str(save_dir),
                          ms.PATH_TO_MODEL_WEIGHTS_RELATIVE : str(save_dir_relative),
                          ms.EXPERIMENT_UID : experiment_uid})

        collection = data_dict[ms.COLLECTION]

        stdh.save_data(data_dict)

        mdb.upload_data_to_mongo(collection_name=collection,batch_data=[data_dict])

        print(f"\nSaved checkpoint for epoch {current_epoch} at {save_dir}")
        return control

