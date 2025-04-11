import csv
import json
import uuid
from collections import defaultdict

import model_settings as ms
import settings as s
import utils.utils_data_creation as ud

dataset_name = "Ethix-Dataset.csv"
dataset_name_path = s.PROJECT_ROOT / "ethix_dataset" / dataset_name
label_dict = {
    "0": "argument from example",
    "1": "argument from values",
    "2": "argument from positive consequences",
    "3": "argument from cause to effect",
    "4": "argument from expert opinion",
    "5": "argument from negative consequences",
    "6": "argument from alternatives",
    "7": "argument from analogy"}

class WorkWithLabels(): # can be used for potential argument_reconstruction
    def __init__(self):
        self.label_dict = defaultdict(list)
        for key, value in label_dict.items():
            self.label_dict[value].append(key)


if __name__ == "__main__":

    argument_list = []

    with open(dataset_name_path, mode="r") as file:
        data_table = list(csv.reader(file))

    for i,row in enumerate(data_table[1:]): # Skip the header row
        if i == 243:
            mewo = 1

        argument_raw = row[0].strip()
        argument_raw = argument_raw.replace("\n", " ")
        argument = ud.clean_text(argument_raw)

        topic = row[1]
        scheme_name = label_dict[row[2]]
        dict_tmp = ud.parse_dict()
        dict_tmp[ms.ARGUMENT] = argument
        dict_tmp[ms.TOPIC] = topic.strip()
        dict_tmp[ms.SCHEME] = scheme_name
        dict_tmp[ms.ARGUMENT_ID] = str(uuid.uuid4())
        dict_tmp[ms.DATASET_NAME] = ms.ETHIX
        argument_list.append(dict_tmp)

    dp = ud.CountDuplicates()
    final_argu_list = dp.check_for_duplicates(argument_list)

    file_path = s.DATA_TO_USE_PATH / 'ethix-dataset.json'
    
    # Open the file in text write mode
    with open(file_path, 'w') as file:
        json.dump(final_argu_list, file, indent=4)
