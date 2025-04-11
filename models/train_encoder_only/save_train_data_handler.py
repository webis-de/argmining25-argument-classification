import settings as s
import model_settings as ms
import json
import mongodb.mongo_handler as mdb
import model_settings as ms
def save_data(data_dict):
    experiment_uid = data_dict[ms.EXPERIMENT_UID]
    sanitized_experiment_name = f"log_{experiment_uid}.json"
    full_log_path = s.MODEL_TRAINED_EVALUATION_PATH / sanitized_experiment_name
    if full_log_path.exists():
        print(f"File {full_log_path} already exists. Skipping saving.")
        return
    with open(full_log_path, 'w') as f:
        json.dump(data_dict, f)


if __name__ == "__main__":

    all_data = []
    # go through the save data files and open them accordingly
    for json_file in s.MODEL_TRAINED_EVALUATION_PATH.glob("*.json") :
        try :
            with open(json_file, "r") as f :
                data = json.load(f)
                if isinstance(data, list) :
                    all_data.extend(data)
                else :
                    all_data.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    # upload the data to the mongodb database

    for data_instance in all_data:
        macro_f1_test = data_instance[ms.DATA][ms.TEST][ms.METRICS_ALL_SCHEMES][ms.MACRO][ms.F1]
        micro_f1_test = data_instance[ms.DATA][ms.TEST][ms.METRICS_ALL_SCHEMES][ms.MICRO][ms.F1]

        macro_f1_dev = data_instance[ms.DATA][ms.DEV][ms.METRICS_ALL_SCHEMES][ms.MACRO][ms.F1]
        micro_f1_dev = data_instance[ms.DATA][ms.DEV][ms.METRICS_ALL_SCHEMES][ms.MICRO][ms.F1]

        data_instance.update({"macro_f1_test" : macro_f1_test,
                              "micro_f1_test" : micro_f1_test,
                              "macro_f1_dev" : macro_f1_dev,
                              "micro_f1_dev" : micro_f1_dev })

    mdb.upload_data_to_mongo(collection_name=ms.TRAIN_LOGS, batch_data=all_data)