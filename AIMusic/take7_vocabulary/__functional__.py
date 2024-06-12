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

def calculate_metrics(generated_seq, real_seq):
    generated_seq_tensor = torch.tensor(generated_seq, dtype=torch.float32)
    real_seq_tensor = torch.tensor(real_seq, dtype=torch.float32)
    
    # NLL (Negative Log Likelihood) для категориальных данных
    real_indices = torch.argmax(real_seq_tensor, dim=-1)
    generated_probs = F.softmax(generated_seq_tensor, dim=-1)
    nll_loss = F.nll_loss(torch.log(generated_probs + 1e-9), real_indices)

    # KL-divergence
    kl_div = F.kl_div(torch.log_softmax(generated_seq_tensor, dim=-1), F.softmax(real_seq_tensor, dim=-1), reduction='batchmean')
    
    # Diversity: используем дисперсию по каждой колонке
    diversity = torch.var(generated_seq_tensor, dim=0).mean()
    
    return nll_loss.item(), kl_div.item(), diversity.item()

