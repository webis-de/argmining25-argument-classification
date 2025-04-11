import copy
import json
import sys
from collections import defaultdict
from copy import deepcopy

import graphviz
from loguru import logger
import argu_class_data.scheme_translation as st
import ask_decision_tree.visser_decision_tree_data as vdt
import settings as s


# parse the argument decision tree
class TreeViz():

    def __init__(self, nodes_to_use = None):
        self.scheme_grouping = st.SchemeFormatGroup() # for tree trimming information about scheme groups is required

        if nodes_to_use is not None:
            for x in nodes_to_use:
                assert x in st.REFERENCE_ALL_SCHEMES_LIST
        if nodes_to_use is None:
            nodes_to_use = self.scheme_grouping.schemes_in_use_list
        nodes_to_use = copy.deepcopy(nodes_to_use)

        if "consequences" in nodes_to_use: # specific for Visser that consequences are more fine-grained
            nodes_to_use.remove("consequences")
            nodes_to_use.extend(["positive consequences", "negative consequences"])

        self.nodes_to_use = nodes_to_use
        self.nodes_to_use += list(set(self.scheme_grouping.group_to_scheme_dict.keys())) # additional names, which are used for scheme grouping
        self.nodes_leaves = []

        self.nodes = dict()
        self.node_label_feature_number_dict = dict()

        self.show_edge_label = True
        self.show_data_set_frequency = True

        self.start_node = None

        ask_data = vdt.ArgumentSchemeKeyDict() # load the ask tree from file
        self.ask_nodes_dict = ask_data.scheme_path_dict
        self.tree_processing()

    # we have multiple stages of the tree which we can visualize
    def vizualize_tree(self,ask_nodes_dict, nodes_viz_inf_dict=None):

        dot = graphviz.Digraph('visser_tree_classification', comment='Questions for ArguSchemes', node_attr={'width': '0.5', 'height': '0.5', 'fixedsize': 'true', 'shape': 'circle'}, edge_attr={'arrowhead': 'none'},engine="dot")
        dot.attr(rankdir='TB', ranksep='0.5')
        for node_id,NodeInformation in ask_nodes_dict.items():

            node_label = str(node_id)
            dot.node(node_id,  label=node_label,  shape= 'box',  fixedsize='true', width='0.5', height='0.5') # , color=color) #,xlabel = xlabel)

            children = NodeInformation.children
            for child in children:
                dot.edge(node_id, child)  #label=str(x[1])

        print(dot.source)
        dot.view()


    # when purge is declared, the references to the node and its children are removed
    def remove_single_node(self, node, node_edges_dict,purge=False): # single child nodes are removed
        if node not in node_edges_dict:
            return

        node_info = node_edges_dict[node]

        # adjust references in the node_parents
        node_parents = node_info.parents
        for parent in node_parents:
            parent_info_node = node_edges_dict[parent]
            node_info_parent_node_pos = parent_info_node.children.index(node)

            if purge:
                del parent_info_node.children[node_info_parent_node_pos]  # remove corresponding edge und question
                del parent_info_node.questions[node_info_parent_node_pos]  # remove corresponding edge und question

            else:
                assert len(node_info.children) == 1 # is only applied to single child nodes
                parent_info_node.children[node_info_parent_node_pos] = node_info.children[0] # use reference to the child



        # adjusts references in children
        children_nodes = node_info.children  # name of child
        for child in children_nodes:
            child_info_node = node_edges_dict[child]
            if purge:
                child_info_node.parents = []
            else:
                child_info_node.parents = node_parents

        del node_edges_dict[node]  # remove node object from dictionary

    # get all nodes sorted to depth, leaves first and then parents towards root node
    def build_tree_hierarchy(self, node_edges_dict,leaves=None) :
        hierarchy_list = []
        if leaves is None :
            leaves = copy.deepcopy(self.nodes_leaves)
        hierarchy_list.append(leaves)
        while leaves :
            parents = self._get_nodes_at_depth(node_edges_dict, leaves)
            if not parents :
                break
            hierarchy_list.append(parents)
            leaves = parents
        return hierarchy_list

    def _get_nodes_at_depth(self, node_edges_dict, nodes) :
        parents_set = set()
        for node in nodes:
            node_info = node_edges_dict[node]
            for parent in node_info.parents:
                parents_set.add(parent)
        return list(parents_set)


    # combine paths which just consist of single edges
    # we are starting with leaves and looking for corresponding parents
    def find_paths_edges(self, node, node_paths_considered=[], node_edges_dict=None):
        if len(node_paths_considered) == 0:
            node_paths_considered = [[node]]

        # find connection in the corresponding node_edges_dict
        parents = []
        if node in node_edges_dict:
            parents = node_edges_dict[node].parents

        if len(parents) > 1:
            logger.info(f"Found multiple Parents {parents}")

        if len(parents) == 0:
            return node_paths_considered

        new_node_paths_considered = []
        for parent in parents: # deal with potential multiple parents (in case a graph is passed)
            if parent not in node_edges_dict:
                continue
            for node_path in node_paths_considered:
                node_path_copy = [deepcopy(node_path) + [parent]]
                node_path_appended = self.find_paths_edges(parent,node_path_copy, node_edges_dict)
                new_node_paths_considered.extend(node_path_appended)

        if new_node_paths_considered == []: # case that all parents are not in the considered dict
            return node_paths_considered
        return new_node_paths_considered

    def get_single_child_nodes(self,node_edges_dict):
        single_child_nodes_dict = dict()
        for node, node_info_object in node_edges_dict.items():
            if len(node_info_object.children) == 1:
                single_child_nodes_dict[node] = node_info_object
        return single_child_nodes_dict


    def trim_clean_tree(self,node_edges_dict):
        self.nodes_leaves = self.find_leaves(self.ask_nodes_dict) # specify new leaves
        def repeated_action_performed(func, *args): # repeat action until action cannot be longer performed
            cnt = 0
            while True :
                action_performed = func(*args)
                cnt += 1
                if not action_performed:
                    print(f"Performed {cnt} times {func}")
                    return cnt


        while True:
            cnt_leaves = repeated_action_performed(self.remove_non_needed_leaves,node_edges_dict) # remove non-needed leaves for scheme classification
            self.nodes_leaves = self.find_leaves(self.ask_nodes_dict) # specify new leaves
            cnt_deleted = repeated_action_performed(self.remove_single_child_nodes,node_edges_dict)
            cnt_clean = repeated_action_performed(self.clean_node_children_and_parents,node_edges_dict) # after summarization of the keys, we can encounter new leaves
            if cnt_leaves == 1 and cnt_deleted == 1 and cnt_clean == 1:
                break

    def clean_node_children_and_parents(self, node_edges_dict):
        action_performed = False
        for node_id,node_object in node_edges_dict.items():
            parents = node_object.parents
            parent_no_duplicates = list(set(parents))
            children = node_object.children
            children_no_duplicates = list(set(children))

            if len(parents) != len(parent_no_duplicates) :
                print("Duplicate parents",node_id,parents)
            if len(children) != len(children_no_duplicates) :
                print("Duplicate children",node_id,children)
                node_object.children = children_no_duplicates
                node_object.questions = [] # indicate that this node is not needed
                action_performed = True
        return action_performed

    def remove_single_child_nodes(self, node_edges_dict):  # single child nodes are removed
        action_performed = False
        tree_hierarchy = self.build_tree_hierarchy(node_edges_dict)
        nodes_with_single_child_dict = self.get_single_child_nodes(node_edges_dict)

        # leaves are appearing first in the tree
        for level in tree_hierarchy:
            for node in level:
                if node in nodes_with_single_child_dict:
                    self.remove_single_node(node, node_edges_dict)
                    action_performed = True
        return action_performed

    def obtain_scheme_paths(self,node_edges_dict,leaves) :
        scheme_path_dict = dict()
        for node in leaves:
            paths = self.find_paths_edges(node, node_edges_dict=node_edges_dict)
            paths_final = []
            for single_path in paths:
                single_path.reverse()
                #assert single_path[0] == '1' # we start with root
                decision_path = []
                for i in range(0,len(single_path)): # extend path of nodes, with the required answers
                    node_i = single_path[i]
                    node_info_i = node_edges_dict[node_i]
                    node_i_plus_1 = single_path[i+1]
                    answer = node_info_i.children.index(node_i_plus_1)
                    decision_path.append((node_i,answer))
                    if node_i_plus_1 == node:
                        break
                paths_final.append(decision_path)
            scheme_path_dict[node] = paths_final
        return scheme_path_dict


    # build specified scheme groups
    def group_schemes(self,node_edges_dict):

        group_dict = defaultdict(list)
        for scheme in self.nodes_leaves: # assume leaves are only scheme names
            # find the group of the scheme
            scheme_group = self.scheme_grouping.scheme_to_group_dict.get(scheme,scheme)
            if scheme_group == scheme:
                continue
            group_dict[scheme_group].append(scheme)

        for scheme_group, single_schemes_list in group_dict.items():
            group_parents = []
            for single_scheme in single_schemes_list:
                single_node_scheme_info_object = node_edges_dict[single_scheme]
                assert len(single_node_scheme_info_object.children) == 0 # classification node shall have no children
                parents = single_node_scheme_info_object.parents
                group_parents.extend(parents)
                del node_edges_dict[single_scheme] # the single scheme nodes are not needed anymore

            # init the object to node_edges_dict
            scheme_group_parents_no_duplicates = list(set(group_parents))
            group_node_object = vdt.NodeInformation(id=scheme_group, parents=scheme_group_parents_no_duplicates)
            node_edges_dict[scheme_group] = group_node_object

            for parent in scheme_group_parents_no_duplicates:
                single_node_scheme_info_object = node_edges_dict[parent]
                children = single_node_scheme_info_object.children
                children = [self.scheme_grouping.scheme_to_group_dict.get(x,x) for x in children] # convert names of schemes to list

                child_list_no_duplicates = list(set(children))
                if len(children) != len(child_list_no_duplicates) :
                    children = child_list_no_duplicates
                    single_node_scheme_info_object.questions = []

                single_node_scheme_info_object.children = children

            group_scheme_info_object = vdt.NodeInformation(id=scheme_group, parents=scheme_group_parents_no_duplicates)
            node_edges_dict[scheme_group] = group_scheme_info_object


    def find_leaves(self,node_edges_dict):
        leaves = []
        for node,node_info in node_edges_dict.items():
            if len(node_info.children) == 0:
                    leaves.append(node)
        return leaves

    # remove all leaves which are not needed, these correspond to schemes we are not being classified
    def remove_non_needed_leaves(self, node_edges_dict):
        action_performed = False
        leaves = self.find_leaves(node_edges_dict)
        for node in leaves :
            if node not in self.nodes_to_use :
                self.remove_single_node(node, node_edges_dict, purge=True) # remove node and its children complete
                action_performed = True
        return action_performed

    def tree_processing(self):

        self.info_node_dict_full = copy.deepcopy(self.ask_nodes_dict)

        self.remove_non_needed_leaves(self.ask_nodes_dict) # remove non-needed leaves for scheme classification
        self.trim_clean_tree(self.ask_nodes_dict)  # remove single child nodes

        self.info_node_dict_trimmed = copy.deepcopy(self.ask_nodes_dict)

        self.group_schemes(self.ask_nodes_dict) # combine multiple schemes to one group
        self.trim_clean_tree(self.ask_nodes_dict)

        for node,key in self.ask_nodes_dict.items(): # final sanity check
            if node in self.nodes_to_use:
                assert len(key.children) == 0
                assert len(key.questions) == 0
            else:
                assert len(key.children) == 2
                assert len(key.questions) == 2

        for node, key in self.ask_nodes_dict.items():
            if len(key.parents) == 0:
                if self.start_node is not None:
                    print(f"Start Node: {self.start_node} already defined")
                    sys.exit(1)
                self.start_node = node
        if self.start_node is None:
            print("No Start Node defined")
            sys.exit(1)

        self.info_node_dict_grouped = copy.deepcopy(self.ask_nodes_dict)

        # obtain the scheme paths, for the final dict
        self.scheme_path_dict = self.obtain_scheme_paths(self.ask_nodes_dict, self.nodes_leaves)


if __name__ == '__main__':
    ARGMINING_SCHEMES = st.parse_yaml(st.FILE_PATH / "argmining_schemes_to_include.yaml")
    x = TreeViz(nodes_to_use=ARGMINING_SCHEMES)

    ASK_SCHEME_QUESTIONS_DICT = x.ask_nodes_dict
    ASK_SCHEME_PATH_DICT = x.scheme_path_dict

    # Save ASK_SCHEME_QUESTIONS_DICT
    with open(s.DATA_TO_USE_PATH / 'final_datasets' /'ask_scheme_questions.json', 'w') as f:
        # Convert NodeInformation objects to serializable dict
        serializable_dict = {}
        for node_id, node_info in ASK_SCHEME_QUESTIONS_DICT.items():
            serializable_dict[node_id] = {
                'children': node_info.children,
                'parents': node_info.parents,
                'options': node_info.questions
            }
        json.dump(serializable_dict, f, indent=4)

    # Save ASK_SCHEME_PATH_DICT 
    with open(s.DATA_TO_USE_PATH / 'final_datasets' / 'ask_scheme_paths.json', 'w') as f:
        json.dump(ASK_SCHEME_PATH_DICT, f, indent=4)

    #x.vizualize_tree(x.info_node_dict_full)
    #x.vizualize_tree(x.info_node_dict_trimmed)
    x.vizualize_tree(x.info_node_dict_grouped)

    # print overview of the used questions
    for key, value in x.ask_nodes_dict.items():
        print(key)
        questions = value.questions
        for question in questions:
            print(question)
        print("====")

