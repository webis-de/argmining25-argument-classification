import models.generate_missing_arguments.generate_missing_argument_ollama as gmo
import models.generate_missing_arguments.generate_missing_argument_gpt4_mini
import re
import settings as s
import data_handling.data_loader as dl
import model_settings as ms


models_to_use_for_config = [gmo.GenerateMissingArgumentOllama]

if __name__ == "__main__":
    # load json with data files, which shall be generated
    path_to_check = s.DATA_TO_USE_PATH / "generated_arguments_raw"
    for file in path_to_check.glob("*.json" ):
        filename = file.stem
        filename = filename.replace("arguments_to_be_generated_", "")

        data = dl.load_json(file)

        for models_to in models_to_use_for_config:
            ModelGen = models_to(data)
            model_name = ModelGen.model_name
            generated_data_list = ModelGen.generate_missing_argument()
            dl.save_json(s.DATA_TO_USE_PATH / "generated_arguments" / f"arguments_generated_{filename}_language_model_{model_name}.json", generated_data_list)



