import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class DataProcessor():
    def __init__(self):
        pass
    
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

    def discard_little_price_movements(self, pred_classes, real_price_movement_pct, pred_price_movement_pct):
        """
        pred_classes contain -1, 0, 1.
        method deletes data for which zero is predicted (from pred_classes)
        and corresponding data from all other lists
        
        returns lists containing data for which significant rise/drop in price is predicted
        """
        pred_classes = [val2 for val2 in pred_classes if val2 != 0]
        real_price_movement_pct = [val1 for val1, val2 in zip(real_price_movement_pct, pred_classes) if val2 != 0]
        pred_price_movement_pct = [val1 for val1, val2 in zip(pred_price_movement_pct, pred_classes) if val2 != 0]
        return pred_classes, real_price_movement_pct, pred_price_movement_pct

        