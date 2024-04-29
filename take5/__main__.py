import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("beethoven_piano_sonatas\measures\\01-1.measures.tsv", sep='\t')
df_str = df.astype(str)
# Инициализация кодировщика OneHotEncoder
encoder = OneHotEncoder()
categorical_columns = ['mc', 'mn', 'quarterbeats', 'duration_qb', 'keysig', 'timesig', 'act_dur', 'mc_offset', 'numbering_offset', 'dont_count', 'barline', 'breaks', 'repeats', 'next']
# Преобразование текстовых данных в OHE и преобразование в numpy array
encoded_df = encoder.fit_transform(df[categorical_columns]).toarray()
# Преобразование в тензор PyTorch
tensor_df = torch.tensor(encoded_df, dtype=torch.float32)
print(tensor_df)


decoded_data = tensor_df.numpy()
# # # Обратное преобразование OHE в исходные категории
decoded_categories = encoder.inverse_transform(decoded_data)
# Создание нового DataFrame из декодированных категорий
decoded_df = pd.DataFrame(decoded_categories, columns=categorical_columns)
print(decoded_df)