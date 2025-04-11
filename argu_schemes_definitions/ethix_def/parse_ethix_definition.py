import settings as s
import argu_class_data.scheme_translation as skt
import re
import model_settings as ms


path_file_to_parse = s.PROJECT_ROOT / 'argu_schemes_definitions' / 'ethix_def' / 'ethix_definition_own.txt'

# dir for translating the corresponding keys
ethix_schemes_translation_key = skt.get_translation_dict("ethix_translation_keys.yaml", dir="dataset_renaming", style="ref")

# add consequences, this scheme is directly defined in Walton, but in ethix it is split into the subschemes positive and negative consequences
ethix_schemes_translation_key.update({"argument from consequences" : "consequences"})

class ParseSchemeDefinition():
    def __init__(self):

        with open(path_file_to_parse, "r") as file:
            text = file.read()

        self.schemes_definition_names_to_standard_names_dict = {}
        self.walton_nbr_to_scheme_dict = {}

        self.schemes_definition_dict = self.split_text_sections_with_keys(text)

        for key, value in self.schemes_definition_dict.items():
            self.schemes_definition_names_to_standard_names_dict[value[ms.SCHEME_ORIG].lower()] = key

        ordered_data = sorted(self.walton_nbr_to_scheme_dict.items(), key=lambda x : float(x[1].replace('.', '').replace(',', '.')))
        # Create ordered dictionary
        self.scheme_order_names = list({k : v for k, v in ordered_data}.keys())

    def split_text_sections_with_keys(self,text):
        matches = re.findall(r'(\n?#\d+\.\d*\.*\s*)(.*?)(?=\n?#\d+\.\d*\.*\s*|\Z)', text, re.DOTALL)
        result = {}
        sections = [(num.strip(), content.strip()) for num, content in matches]

        for num, content in sections :
            print(f"Section: {num}\nContent: {content}\n")        # Split the section into lines, with the first line as the key
            lines = content.strip().split('\n', 1)

            scheme_name = lines[0].strip()  # First line as key
            short_form_key = ethix_schemes_translation_key[scheme_name.lower()]
            if short_form_key is None:
                print(f"Key: {scheme_name} -> {short_form_key}")
            scheme_description = lines[1].strip() if len(lines) > 1 else ""  # Rest of the section as value
            scheme_description_full = scheme_name + "\n" + scheme_description

            num_formatted = num.replace("#", "").strip()
            self.walton_nbr_to_scheme_dict[short_form_key] = num_formatted
            reference_dict = {ms.SCHEME : scheme_description_full, ms.NBR : num_formatted , ms.SCHEME_ORIG: scheme_name}
            result[short_form_key] = reference_dict

        return result

    def get_schemes_definitions(self):
        with open(path_file_to_parse, "r") as file:
            text = file.read()
        sections = self.split_text_sections_with_keys(text)
        return sections


if __name__ == "__main__":
    x = ParseSchemeDefinition()
    new_stuff = x.schemes_definition_names_to_standard_names_dict
