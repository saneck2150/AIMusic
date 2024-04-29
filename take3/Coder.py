import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_dataframe(df):
    encoded_df = df.copy()
    label_encoders = {}
    standard_scalers = {}
    
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            encoded_df[column] = label_encoders[column].fit_transform(df[column])
        else:
            standard_scalers[column] = StandardScaler()
            encoded_df[column] = standard_scalers[column].fit_transform(df[[column]])
    
    tensor_data = torch.tensor(encoded_df.values, dtype=torch.float32)
    
    return tensor_data, label_encoders, standard_scalers

def decode_tensor(tensor_data, label_encoders, standard_scalers, columns):
    decoded_df = pd.DataFrame(tensor_data.numpy(), columns=columns)
    
    for column in columns:
        if column in standard_scalers:
            decoded_df[column] = standard_scalers[column].inverse_transform(decoded_df[[column]])
        elif column in label_encoders:
            decoded_df[column] = label_encoders[column].inverse_transform(decoded_df[column].astype(int))
    
    return decoded_df

def decode_dataframe(encoded_data, label_encoders, standard_scalers):
    decoded_data = pd.DataFrame()

    # Обратное преобразование для каждого столбца
    for i, column in enumerate(encoded_data.columns):
        data = encoded_data.iloc[:, i]

        # Если столбец числовой, используем обратное преобразование стандартизации
        if column in standard_scalers:
            scaler = standard_scalers[column]
            decoded_data[column] = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        # Если столбец категориальный, используем обратное преобразование кодирования меток
        elif column in label_encoders:
            encoder = label_encoders[column]
            decoded_data[column] = encoder.inverse_transform(data)

    return decoded_data
