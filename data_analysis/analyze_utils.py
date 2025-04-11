import copy
import pandas as pd
import model_settings as ms
import settings as s
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

def set_min_max_dataframes(df,min_number=None,max_number=None):
    if min_number is not None:
        df = df[df[ms.NBR] >= min_number].reset_index(drop=True)
    if max_number is not None:
        df[ms.NBR] = df[ms.NBR].apply(lambda x : max_number if x > max_number else x)

    df = sort_dataframe_most_used(df)
    df = calculate_percentage_df(df)
    return df

def sort_dataframe_most_used(df):
    return df.sort_values(by=ms.NBR, ascending=False).reset_index(drop=True)


def calculate_percentage_df(df):
    df[ms.PERCENTAGE] = (df[ms.NBR] / df[ms.NBR].sum()) * 100
    return df

def create_pandas_df(scheme_dict,name=None):
    scheme_counts = {key : len(val) for key, val in scheme_dict.items()}
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(list(scheme_counts.items()), columns=[ms.SCHEME, ms.NBR])
    df = sort_dataframe_most_used(df)
    df = calculate_percentage_df(df)
    if name is not None:
        df[ms.DATASET_NAME] = name
    return df

def combine_data_frames(dataframes,name=None):
    # Combine the DataFrames
    combined_df = pd.concat([df[[ms.SCHEME, ms.NBR]] for df in dataframes])
    df = combined_df.groupby(ms.SCHEME, as_index=False).sum()
    df = sort_dataframe_most_used(df)
    df = calculate_percentage_df(df)
    if name is not None:
        df[ms.DATASET] = name
    return df

def print_data_frame_as_table(data_frame):
    # sort dataset according to most used

    sorted_df = sort_dataframe_most_used(data_frame)
    sorted_df = sorted_df[[ms.SCHEME, ms.NBR, ms.PERCENTAGE]]
    sorted_df.loc[:, ms.PERCENTAGE]= sorted_df[ms.PERCENTAGE].round(2) # round the percentage
    sorted_df.index = range(1, len(sorted_df) + 1)
    print(tabulate(sorted_df, headers=sorted_df.columns, tablefmt="latex"))
    print("Total : {}".format(sum(sorted_df[ms.NBR].tolist())))


def show_histogram(df, argus_to_use):
    filtered_df = df[df['Scheme'].isin(argus_to_use)]
    filtered_df['Scheme'] = pd.Categorical(filtered_df['Scheme'],
                                                                categories=argus_to_use, ordered=True)
    filtered_df = filtered_df.sort_values(by='Scheme')
    plt.xticks(rotation=90, fontsize=5)  # Adjust fontsize here
    sns.barplot(x='Scheme', y='Nbr', data=filtered_df)
    plt.title('Scheme Frequency')
    plt.subplots_adjust(bottom=0.25)

    plt.xlabel('')
    plt.ylabel('Nbr')
    plt.savefig("arg_histograms" + ".pdf")  # "Adjust dpi as per your requirement
    plt.show()

