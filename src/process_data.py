import numpy as np
import pandas as pd


def create_Xy_data(data_for_X, data_for_y, input_seq_len: int, output_seq_len: int):
    Xdata, ydata = [], []
    for i in range(input_seq_len, len(data_for_X)-(output_seq_len-1)):
        Xdata.append(data_for_X.iloc[i-input_seq_len:i])
        ydata.append(data_for_y.iloc[i:i+output_seq_len])
    return np.array(Xdata), np.array(ydata)


def split_data(Xdata, ydata, train_pct, val_pct):
    train_val_split = int(train_pct*(Xdata.shape[0]))
    val_test_split = int((train_pct+val_pct)*(Xdata.shape[0]))
    return Xdata[:train_val_split], ydata[:train_val_split], \
            Xdata[train_val_split:val_test_split], ydata[train_val_split:val_test_split], \
            Xdata[val_test_split:], ydata[val_test_split:]


def diff_features(data: pd.DataFrame, columns: list, period: int=1):
    for col in columns:
        data[col] = data[col].diff(period)
    data = data[period:]  # first period rows NaN
    return data


def normalize_custom(data: pd.DataFrame, train_pct: float, columns: list):
    train_data_len = int(train_pct*data.shape[0])
    params = {}
    for col in columns:
        minimum = min(data[col][:train_data_len])
        maximum = max(data[col][:train_data_len])
        data[col] =(data[col] - minimum)/(maximum - minimum)
        params[col] = (minimum, maximum)
    return data, params
