import numpy as np
import pandas as pd


def create_Xy_data(data_for_X, data_for_y, input_seq_len, output_seq_len):  # assert len(data_for_X) == len(data_for_y)
    Xdata, ydata = [], []
    for i in range(input_seq_len, len(data_for_X)-(output_seq_len-1)):
        Xdata.append(data_for_X.iloc[i-input_seq_len:i])
        ydata.append(data_for_y.iloc[i:i+number_of_days])
    return np.array(Xdata), np.array(ydata)