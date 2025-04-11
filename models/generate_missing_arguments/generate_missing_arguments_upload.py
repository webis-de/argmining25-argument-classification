import models.generate_missing_arguments.generate_missing_argument_ollama as gmo
import models.generate_missing_arguments.generate_missing_argument_gpt4_mini
import re
import settings as s
import data_handling.data_loader as dl
import model_settings as ms
import mongodb.mongo_handler as mdb


if __name__ == "__main__":
    # load json with data files, which shall be generated
    path_to_check = s.DATA_TO_USE_PATH / "generated_arguments"
    for file in path_to_check.glob("*.json" ):
        filename = file.stem

        match_node = re.search(r'node_(\d+)', filename)
        node = str(int(match_node.group(1)))  # 17
        match_split = re.search(r'split_(\d+)', filename)
        split = str(int(match_split.group(1)))
        match_split = re.search(r'language_model_(.+).json', filename)
        language_model = str(int(match_split.group(1)))
        loaded_data_list = dl.load_json(file)

        # generate the needed synthetic data
        for argument in loaded_data_list:
            argument[ms.MODEL_NAME] = language_model
            argument[ms.NODE] = node
            argument[ms.SPLIT] = ms.TRAIN
            argument[ms.SPLIT_IDENTIFIER] = split
            argument[ms.DATASET_NAME] = ms.ETHIX_SYNTHETIC

        mdb.upload_data_to_mongo(collection_name=ms.ETHIX_SYNTHETIC, batch_data=loaded_data_list)

