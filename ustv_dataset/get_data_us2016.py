import json
from collections import defaultdict
from pathlib import Path
# in this regard often
from collections import defaultdict
import re
import settings as s
import model_settings as ms
import copy
import datetime
import utils.utils_data_creation as ud
import utils.utils as ut
import uuid
import pickle

# get list of files with containing argument

class GetDataVisser():
    def __init__(self,files_to_check):
        self.node_types_text_dict = defaultdict(list)
        self.node_schemes_info = defaultdict(list) # overall information about the nodes

        self.walton_schemes_dict = defaultdict(list) # schemes which contain only schemes from the original dataset

        self.node_types_to_be_ignored = ["YA","LA","TA",'L']

        for file in files_to_check :
            self.analyze_single_file(file)

    def process_data(self, data_dict) :
        for scheme, argus in data_dict.items() :
            for single_argument_object in argus:
                final_premise_list = self.format_nodes(single_argument_object[ms.PREMISE_LIST])
                final_claim_list = self.format_nodes(single_argument_object[ms.CONCLUSION_LIST])
                final_argument_list =  self.format_nodes(single_argument_object[ms.PREMISE_LIST] + single_argument_object[ms.CONCLUSION_LIST])

                if len(final_argument_list) == 0:
                    print("Found Argument without Premises or Claims")
                    continue

                # create final argument object
                single_argument_object[ms.PREMISE_LIST] = final_premise_list
                single_argument_object[ms.CONCLUSION_LIST] = final_claim_list
                single_argument_object[ms.ARGUMENT_LIST] = final_argument_list
                single_argument_object[ms.ARGUMENT] = " ".join(final_argument_list)
                single_argument_object[ms.ARGUMENT_ID] = str(uuid.uuid4())
                single_argument_object[ms.DATASET_NAME] = ms.USTV2016

            self.walton_schemes_dict[scheme].extend(argus)

    def format_nodes(self, nodes_list):
        nodes_list = sorted(nodes_list, key=lambda x: x["timestamp"])
        nodes_list_formatted = [ud.clean_text(x["text"]) for x in nodes_list]
        return nodes_list_formatted

    def format_single_node(self, node_id):
        node = copy.deepcopy(self.all_nodes_evaluation_dict[node_id])
        node["timestamp"] = datetime.datetime.strptime(node["timestamp"], '%Y-%m-%d %H:%M:%S')
        node_type = node["type"]
        if node_type not in self.node_types_to_be_ignored :
            self.node_types_text_dict[node_type].append(node["text"])
            return node
        return None

    def analyze_single_file(self, file_path):
        self.node_schemes = defaultdict(list)
        self.all_nodes_evaluation_dict = dict()
        argument_dict = defaultdict(list)

        # select corresponding matching nodesets
        pattern = r"nodeset(\d+)\.json"
        match = re.search(pattern, file_path.name)
        if match:
            file_id = match.group(1)
        else:
            return
        print(f'Processing file number: {file_id}')

        # load data from the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        scheme_node_ids_list = []

        # store information about all nodes
        for node in data['nodes']:
            node_id = str(node['nodeID'])
            node_type = node["type"]
            self.node_schemes_info[node_type].append(node['text'])

            if node_id in self.all_nodes_evaluation_dict.keys():
                print(f"Node {node_id} already exists in File")
                continue

            self.all_nodes_evaluation_dict[node_id] = node
            if "scheme" in node: # get corresponding scheme id nodes
                scheme_node_ids_list.append(node_id)
            elif "Default Inference" in node["text"]:
                scheme_node_ids_list.append(node_id)

        # go through list of node id for schemes and get incoming and outcoming nodes
        for scheme_node_id in scheme_node_ids_list:
            scheme_node = self.all_nodes_evaluation_dict[scheme_node_id]
            scheme_name = scheme_node["text"]

            dict_tmp = ud.parse_dict()
            dict_tmp[ms.SCHEME] = scheme_name
            dict_tmp[ms.ID] = scheme_node_id
            dict_tmp[ms.FILE_ID] = [file_id]

            # go through all corresponding edges and see to which nodes the scheme is connected
            for edge in data['edges']:
                if str(edge["toID"]) == scheme_node_id:
                    edge_node = self.format_single_node(str(edge["fromID"]))
                    if edge_node is not None:
                        dict_tmp[ms.PREMISE_LIST].append(edge_node)

                if str(edge["fromID"]) == scheme_node_id:
                    edge_node = self.format_single_node(str(edge["toID"]))
                    if edge_node is not None:
                        dict_tmp[ms.CONCLUSION_LIST].append(edge_node)

            argument_dict[scheme_name].append(dict_tmp)
        self.process_data(argument_dict)


if __name__ == "__main__":
    path_data = s.PROJECT_ROOT / "ustv_dataset" / 'data' / "argus_tv16_walton"
    files = ud.get_files(path_data)
    VisserData = GetDataVisser(files)

    dp = ud.CountDuplicates()
    argument_list = []
    for scheme, argus in VisserData.walton_schemes_dict.items():
        argument_list.extend(argus)

    final_argu_list = dp.check_for_duplicates(argument_list)

    scheme_dict = ut.convert_argument_list_to_schemes_dict(final_argu_list)

    file_path = s.DATA_TO_USE_PATH / 'ustv2016-dataset.json'

    # Save the data as JSON
    with open(file_path, 'w') as file:
        json.dump(final_argu_list, file, indent=4)



