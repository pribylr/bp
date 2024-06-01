import numpy as np
import pandas as pd

class DataProcessor():
    def __init__(self):
        pass
        
    def create_Xy_data(self, data_for_X: pd.DataFrame, data_for_y: pd.DataFrame, input_seq_len: int, output_seq_len: int):
        Xdata, ydata = [], []
        for i in range(input_seq_len, len(data_for_X)-(output_seq_len-1)):
            Xdata.append(data_for_X.iloc[i-input_seq_len:i])
            ydata.append(data_for_y.iloc[i:i+output_seq_len])
        return np.array(Xdata), np.array(ydata)
    
    
    def split_data(self, Xdata: np.array, ydata: np.array, train_pct: float, val_pct: float):
        print(type(Xdata), type(ydata), type(train_pct), type(val_pct))
        train_val_split = int(train_pct*(Xdata.shape[0]))
        val_test_split = int((train_pct+val_pct)*(Xdata.shape[0]))
        return Xdata[:train_val_split], ydata[:train_val_split], \
                Xdata[train_val_split:val_test_split], ydata[train_val_split:val_test_split], \
                Xdata[val_test_split:], ydata[val_test_split:]
    
    
    def diff_features(self, data: pd.DataFrame, columns: list, period: int=1):
        for col in columns:
            data[col] = data[col].diff(period)
        data = data[period:]  # first period rows NaN
        return data
    
    
    def normalize_custom(self, data: pd.DataFrame, train_pct: float, columns: list):
        train_data_len = int(train_pct*data.shape[0])
        params = {}
        for col in columns:
            minimum = min(data[col][:train_data_len])
            maximum = max(data[col][:train_data_len])
            data[col] =(data[col] - minimum)/(maximum - minimum)
            params[col] = (minimum, maximum)
        return data, params
