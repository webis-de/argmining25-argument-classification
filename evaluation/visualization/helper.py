import pandas as pd

import model_settings as ms
from collections import defaultdict
import re
import evaluation.visualization.viz_keys as k
import settings

def adjust_to_latex(prd_base):
    dict_to_return = dict()
    for x,y in prd_base.items():
        dat_list = y
        if not isinstance(y, list):
            dat_list = [y]
        final_list = []
        for sing in dat_list:
            if isinstance(sing, float) or isinstance(sing, int):
                val = "{:.2f}".format(sing)
                dat = val.split(".")
                if dat[0] == "1":
                    val = "100"
                else:
                    val = dat[1]
                sing = "\cell{"+val+"}"
            final_list.append(sing)
        fin_element = final_list
        if len(final_list) == 1:
            fin_element = final_list[0]
        dict_to_return.update({x:fin_element})
    return dict_to_return

# do renaming of underlying dataframe

def apply_function_with_exclusions(df, data_configuration=None):
    if data_configuration is None:
        data_configuration = dict()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    """
    Apply a function to each cell of a DataFrame, excluding specific row-column combinations.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    func (function): The function to apply to each cell.
    exclude_combinations (list of tuples): List of (row_index, column_name) combinations to exclude.

    Returns:
    pd.DataFrame: Transformed DataFrame after applying the function.
    """
    exclude_combinations = data_configuration.get(k.EXCLUDE, [])
    include_combinations = data_configuration.get(k.INCLUDE, [])

    exclude_combinations = [ (str(x[0]),str(x[1])) for x in exclude_combinations]
    include_combinations = [ (str(x[0]),str(x[1])) for x in include_combinations]

    cell_replace_func = data_configuration.get(k.CELL_REPLACE, lambda x : x)
    empty_value = data_configuration.get(k.EMPTY_VALUE,"")
    empty_replace_func = data_configuration.get(k.EMPTY_REPLACE, lambda x : empty_value)

    def transformation_function(value, row, col):
        if value == empty_value :
            return empty_replace_func(value)

        # Check if the (row, col) should be excluded
        if (row, col) in exclude_combinations or (row, "*") in exclude_combinations or (
        "*", col) in exclude_combinations :
            return value  # Skip transformation

        if len(include_combinations) > 0:
            if ("*", col) in include_combinations or (row, "*") in include_combinations :
                return cell_replace_func(value)
            return value

        return cell_replace_func(value)

    df_transformed = df.copy().astype(object)

    for row in df.index :
        for col in df.columns :
            df_transformed.at[row, col] = transformation_function(df.at[row, col], row, col)

    return df_transformed

# formatting syntax for identifying various schemes
# work with scheme dfs

def print_pretty_table(df, config_dict=None,padding=2,rows=True,columns=True):
    if config_dict is None:
        config_dict = dict()

    """
    Prints a well-formatted table with aligned columns, respecting exclusions.

    :param df: Pandas DataFrame
    :param config_dict: Configuration dictionary with exclusions and row order
    :return: List of formatted table rows as strings
    """
    final_list = []
    exclude_combinations = config_dict.get(k.EXCLUDE, [])
    translation_dict = config_dict.get(k.ROW_INDEX_TRANSLATION, dict())
    translation_dict  = {str(k):str(v) for k,v in translation_dict.items()}

    row_order = config_dict.get(k.ROW_ORDER, df.index) # in case additional line breaks etc, shall be introduced

    for entry in df.index: # rename index
        entry = str(entry)
        new_val = translation_dict.get(entry, entry)
        df = df.rename(index={entry: new_val})

    df.insert(0, 'index', df.index)
    longest_cells_dict = get_longest_strings_for_column_in_dataframe(df, exclude_combinations)  # Fix: get length

    for entry in row_order :
        if str(entry) in translation_dict:
            entry = translation_dict[str(entry)]

        if entry not in df.index:
            final_list.append(entry)  # Add corresponding latex name
            continue

        row_values = []
        for col in df.columns :

            value =df.loc[entry, col]
            assert not isinstance(value, pd.Series), "List in DataFrame not supported"
            value = str(value)
            rest = ""
            value_diff_column = longest_cells_dict[col] - len(value) # get every string to the right length

            # check addition if value ends with a &
            if value.endswith("&"):
                value = value[:-1]
                rest = "&"

            difference = value_diff_column + padding
            value += (" " * difference) + rest  # Additional Padding if
            row_values.append(value)

        values_joined = "& ".join(row_values) + r" \\"

        final_list.append(values_joined)  # Join row elements

    final_string = "\n".join(final_list)
    print(final_string)
    return final_string



def get_longest_strings_for_column_in_dataframe(df, exclude_combinations=None):
    """
    Finds the longest string in a DataFrame while respecting specific (row, col) exclusions.
    Supports wildcard (*) for rows or columns.

    :param df: The DataFrame to search.
    :param exclude_combinations: A set of (row, col) pairs to exclude, supports wildcards (*).
    :return: The longest string in the DataFrame.
    """
    if exclude_combinations is None:
        exclude_combinations = set()

    column_lengths_dict = defaultdict(int)

    for col in df.columns:

        max_length_column = 0

        for index, row in df.iterrows() :

            # Check if the current (row, col) is in the exclusion set
            if (index, col) in exclude_combinations or (index, "*") in exclude_combinations or ("*", col) in exclude_combinations:
                continue  # Skip excluded cells

            value_len = len(str(row[col]))
            if value_len > max_length_column:
                max_length_column = value_len
        column_lengths_dict[col] = max_length_column

    return column_lengths_dict







# get data from list of data
def search_datasets(data_list,pair=False, **kwargs):
    final_data_list = []
    for data_pair in data_list:
        data = data_pair
        if pair:
            data = data_pair[0]
        flag = 1
        for x,y in kwargs.items():
            if data[x] != y:
                flag = 0
        if flag:
            final_data_list.append(data_pair)
    if len(final_data_list) == 0:
        return None
    return final_data_list


def prepare_print_special_latex_table(table_data):
    final_data_to_print = ""
    for row in table_data:
        final_row = ""
        first = row[:3]
        first_join = "&".join(first)
        final_row += first_join
        rest = row[3:]
        while len(rest) != 0:
            head = rest[:2]
            head_join = "&".join(head)
            final_row += "&&" + head_join
            rest = rest[2:]
        final_data_to_print += final_row + r"\\" + "\n"
    return final_data_to_print

