import model_settings as ms
from collections import defaultdict
from pathlib import Path

def parse_dict():
    return {
        ms.PREMISE_LIST : [],
        ms.CONCLUSION_LIST : [],
        ms.ARGUMENT : "",
        ms.SCHEME : "",
        ms.ARGUMENT_ID : "",
    }
def format_text(text):
    text = text.replace(" ,", ",").replace(" .", ".").replace(" '", "'")
    return text

def clean_text(text) :
    text = text.strip()
    text = format_text(text)
    if not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
        text += "."
    if text[0].islower() :  # Check if the first character is lowercase
        text = text[0].upper() + text[1 :]  # Capitalize the first character
    return text

def create_argument_structure(premises_sorted, claims_sorted):

    all_text_premises_premises = ['Premise: ' + clean_text(x["text"]) + '\n' for x in premises_sorted]
    all_text_premises_claim = ['Conclusion: ' + clean_text(x["text"]) + '\n' for x in claims_sorted]
    if len(all_text_premises_claim) > 0:
        all_text_premises_claim[-1] = all_text_premises_claim[-1][:-1]
    final_text_list = all_text_premises_premises + all_text_premises_claim
    final_text = "".join(final_text_list)
    return final_text


# get list of files with containing arguments
def get_files(start_path):
    files_list = []
    filepath_start = Path(start_path)
    dirs_to_check = [filepath_start]
    while len(dirs_to_check) != 0:
        current_dir = dirs_to_check.pop()
        for file in current_dir.iterdir():
            if file.is_dir():
                dirs_to_check.append(file)
            else: 
                files_list.append(file)
    return files_list



class CountDuplicates():
    def __init__(self):
        self.final_args = []
        self.arg_set = defaultdict(set)
        self.nbr_duplicates = 0

    def check_for_duplicates(self, argu_list, key=ms.ARGUMENT):
        scheme = "all"
        for arg in argu_list:
            if len(arg[ms.ARGUMENT]) == 0 or len(arg[ms.ARGUMENT]) == 0:
                print("Found empty argument")
                continue
            arg_text = arg[key]
            if arg_text in self.arg_set[scheme]:
                self.nbr_duplicates += 1
            else:
                self.arg_set[scheme].add(arg_text)
                self.final_args.append(arg)

        print(f"Detected {self.nbr_duplicates} Duplicates")
        return self.final_args

    # check for established arguments
    fn = 1
