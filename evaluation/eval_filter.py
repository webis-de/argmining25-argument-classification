import settings as s
import model_settings as ms
import evaluation.eval_base_class as eb
import data_handling.data_handler as dh
import copy
import mongodb.mongo_handler as mdb
# helper with underlying filters for the paper
from collections import defaultdict
random_seed = 42

# is set to get a specified number of arguments corresponding to a specific scheme
def get_specified_args_from_database(filter_orig, min_nbr=None, max_nbr=None, argument_ids_to_get_list=None):

    # get schemes dict for a particular evaluation
    filter_dict = copy.deepcopy(filter_orig)
    retrieved_argument_list = mdb.get_data_from_mongo(filter_dict=filter_dict)

    if argument_ids_to_get_list is not None:
        retrieved_argument_list = [arg for arg in retrieved_argument_list if arg[ms.ARGUMENT_ID] in argument_ids_to_get_list]
        assert len(retrieved_argument_list)  == len(argument_ids_to_get_list), f"not all argument ids are in the database {argument_ids_to_get_list} - {len(retrieved_argument_list)}"
    cal_helper = dh.DataLoadBase()
    return cal_helper.clean_data(retrieved_argument_list, min_nbr=min_nbr, max_nbr=max_nbr)


