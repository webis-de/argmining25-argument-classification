import settings as s
import argu_class_data.scheme_translation as kt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

path_file = s.PROJECT_ROOT / "argu_class_data" / "ask_decision_trees" / "visser_argumentation_scheme_key_dict.txt"

@dataclass
class NodeInformation :
    id: str
    parents: List = field(default_factory=list)
    children: List = field(default_factory=list)
    questions: List = field(default_factory=list)

class ArgumentSchemeKeyDict():
    def __init__(self):

        self.key_translate_dicts = kt.get_translation_dict("visser_translation_keys", dir="dataset_renaming") # get specified translation dict for the ask keys
        self.do_preprocessing()
        self.create_tree_information()

    def do_preprocessing(self):
        with open(path_file, "r") as f:
            scheme_lines = f.readlines()

        decision_pairs = []
        current_decision = []

        for x in scheme_lines: # go through each single line
            cleaned = x.strip()
            if cleaned == "": # empty line marks start of new classification pair
                if current_decision != []:
                    decision_pairs.append(current_decision)
                current_decision = []
            else:
                current_decision.append(cleaned)

        self.question_argument_dict = defaultdict(list)

        for x in decision_pairs: # go through each classification pair and perform string processing
            assert len(x) == 2
            first_line = x[0].split(";")
            first_line = [x.strip() for x in first_line]
            second_line = x[1].split(";")
            second_line = [x.strip() for x in second_line]

            assert len(first_line) == len(second_line) == 3 # three entries in each line

            nbr_ident = first_line[0].split("(")[0].strip()
            self.question_argument_dict[nbr_ident] = [first_line, second_line]

        self.create_tree_information()

    # process preliminary dict into specific data object
    def create_tree_information(self):
        questions_dict = dict()


        parent_dict = dict()
        for key, value_list in self.question_argument_dict.items():
            key = self.key_translate_dicts.get(key, key)
            assert len(value_list) == 2
            data_first = value_list[0] # first line of the classification pair
            data_second = value_list[1] # second line of the classification pair

            # index 2 is the child of the current node
            children = [self.key_translate_dicts.get(data_first[2], data_first[2]),self.key_translate_dicts.get(data_second[2], data_second[2])] # get the children of the current node
            questions = [data_first[1], data_second[1]] # get the questions of the current node
            for children_key in children:
                parent_dict[children_key] = key # add the current node as parent to the children

            node_info = NodeInformation(id=key,
                                        questions=questions,
                                        children=children)
            questions_dict[key] = node_info

        # add leaves
        leaves = dict() # separate dict for identifying leaves, these nodes are only mentioned as children from other nodes
        for key, node_info_object in questions_dict.items():
            for children_key in node_info_object.children:
                if children_key not in questions_dict:
                    leaves[children_key] = NodeInformation(id=children_key)
        questions_dict.update(leaves)

        # update parent information
        for key, node_info_object in questions_dict.items():
            parents = parent_dict.get(key, None)
            if parents:
                node_info_object.parents = [parents] # add information about parent to the child

        self.scheme_path_dict = questions_dict


if __name__ == "__main__":
    x = ArgumentSchemeKeyDict()




    mewo = 1
















