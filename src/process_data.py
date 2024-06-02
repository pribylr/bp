import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

    def summed_price_movement(self, arr: np.array):  # [batch_size, seq_len]
        return np.sum(arr, axis=1)

    def create_binary_classification_classes(self, values: list):
        return [1 if value >= 0 else -1 for value in values]

    def create_ternary_classification_classes(self, values: list, threshold: float):
        return [1 if value > threshold else 0 if -threshold <= value <= threshold else -1
        for value in values]

    def create_metrics_from_classes_binary(self, real_classes: list, pred_classes: list):
        accuracy = accuracy_score(real_classes, pred_classes)
        precision = precision_score(real_classes, pred_classes)
        recall = recall_score(real_classes, pred_classes)
        f1 = f1_score(real_classes, pred_classes)
        return accuracy, precision, recall, f1

    def create_metrics_from_classes_ternary(self, real_classes: list, pred_classes: list):
        accuracy = accuracy_score(real_classes, pred_classes)    
        precision = precision_score(real_classes, pred_classes, labels=[-1, 0, 1], average='macro')
        recall = recall_score(real_classes, pred_classes, labels=[-1, 0, 1], average='macro')
        f1 = f1_score(real_classes, pred_classes, labels=[-1, 0, 1], average='macro')
        return accuracy, precision, recall, f1

    def calculate_value_movement_pct(self, value: float, value_move: float):
        return (value_move / value) * 100

    def find_pct_movements(self, input_sequences: np.array, real_values: pd.DataFrame, price_movements: list, pip_factor: int):
        """
        function finds last data point of each input sequence
        then calculates percentage change of an asset's price
        returns list of all percentage changes
        """
        test_size = input_sequences.shape[0]
        result = []
        for i in range(test_size):
            input_seq_last_data_point_time = input_sequences[i,-1,0]  # i: batch, -1: last data point in sequence, 0: date
            close_value_at_that_time = real_values[real_values['time'] == input_seq_last_data_point_time]['close'].item()
        
            price_movement_at_that_time = price_movements[i]
            price_movement_at_that_time /= pip_factor
            price_movement_at_that_time_pct = self.calculate_value_movement_pct(close_value_at_that_time, price_movement_at_that_time)
            result.append(price_movement_at_that_time_pct)
        return result
    
        