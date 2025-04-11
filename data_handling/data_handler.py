import random
from collections import defaultdict

import model_settings as ms
import settings as s



class DataLoadBase():
    def __init__(self):
        random.seed(s.SEED)

    def prepare_argument_data_dict(self, argument_list):
        # flatten for easier sorting
        argument_list = sorted(argument_list, key=lambda x: x[ms.ARGUMENT_ID])
        dict_to_return = defaultdict(list)
        for argument in argument_list:
            scheme_name = argument[ms.SCHEME]
            dict_to_return[scheme_name].append(argument)
        return dict_to_return

    def clean_data(self, argument_dict, min_nbr=None, max_nbr=None):
        prepared_data_dict = self.prepare_argument_data_dict(argument_dict)
        if min_nbr is not None:
            prepared_data_dict = self.set_lower_bound_argument_frequency(prepared_data_dict, min_nbr)
        if max_nbr is not None:
            prepared_data_dict = self.set_upper_bound_argument_frequency(prepared_data_dict, max_nbr)
        return prepared_data_dict

    def set_upper_bound_argument_frequency(self, argument_dict, upper_bound=None) :
        dict_to_return = {}

        # do it in a reproducible way
        for scheme_name,arguments in argument_dict.items():

            if upper_bound is None or upper_bound >= len(arguments) :
                filtered = arguments  # Return all arguments if no upper bound or too large
            else :
                filtered = random.sample(arguments, k=upper_bound)  # Use k=upper_bound

            dict_to_return[scheme_name] = filtered
        return dict_to_return

    def set_lower_bound_argument_frequency(self, argument_dict,lower_bound=None):
        dict_to_return = dict()
        for scheme_name,arguments in argument_dict.items():
            if len(arguments) < lower_bound:
                continue
            dict_to_return[scheme_name] = arguments
        return dict_to_return
