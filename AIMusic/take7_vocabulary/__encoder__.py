from __import__ import *

def encodeList (list_of_lists, df):
    transposed_list = list(map(list, zip(*list_of_lists)))
    unique_values_dict = {df.columns[i]: set(transposed_list[i]) for i in range(len(df.columns))}
    numbered_list = []
    for sublist, col_name in zip(transposed_list, df.columns):
        numbered_sublist = [list(unique_values_dict[col_name]).index(item) for item in sublist]
        numbered_list.append(numbered_sublist)
    numbered_list = list(map(list, zip(*numbered_list)))
    return numbered_list, unique_values_dict

def decodeList(rounded_generated_seq, unique_values_dict):
    new_data_list = list(map(list, zip(*rounded_generated_seq)))
    original_list_of_lists = []
    for col_idx, sublist in enumerate(new_data_list):
         column_name = list(unique_values_dict.keys())[col_idx]
         column_values = list(unique_values_dict[column_name])
         original_sublist = [
            column_values[item] if 0 <= item < len(column_values) else 0 
            for item in sublist]
         original_list_of_lists.append(original_sublist)    
    original_list_of_lists = list(map(list, zip(*original_list_of_lists)))
    return original_list_of_lists