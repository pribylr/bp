import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer():
    def __init__(self):
        pass
    
    def plot_four_predictions(self, predictions: list, real_data: np.array, idx0: int, idx1: int, idx2: int, idx3: int, target_idx: int):
        
        # real_data: [batch_size, output_seq_len, features]
        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.plot(predictions[0][0, :, target_idx], c='magenta', linestyle=':')
        ax1.plot(real_data[idx0, :, target_idx], c='black')
        ax2.plot(predictions[1][0, :, target_idx], c='magenta', linestyle=':')
        ax2.plot(real_data[idx1, :, target_idx], c='black')
        ax3.plot(predictions[2][0, :, target_idx], c='magenta', linestyle=':')
        ax3.plot(real_data[idx2, :, target_idx], c='black')
        ax4.plot(predictions[3][0, :, target_idx], c='magenta', linestyle=':')
        ax4.plot(real_data[idx3, :, target_idx], c='black')

    """
    display four line plots
    each plot has vertical line in middle
    left of the line - input sequence (actual price of an asset)
    right of the line - 1) real output sequence that follows the input and 2) predicted output sequence
    """
    def plot_real_predicted_sequences(self, col: str, real_candles: list, pred_candles: list, seq_len: int):
        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(list(real_candles[0][col]), c='black', alpha=0.7)
        ax1.plot(list(pred_candles[0][col]), c='magenta', linestyle='--')
        ax2.plot(list(real_candles[1][col]), c='black', alpha=0.7)
        ax2.plot(list(pred_candles[1][col]), c='magenta', linestyle='--')
        ax3.plot(list(real_candles[2][col]), c='black', alpha=0.7)
        ax3.plot(list(pred_candles[2][col]), c='magenta', linestyle='--')
        ax4.plot(list(real_candles[3][col]), c='black', alpha=0.7)
        ax4.plot(list(pred_candles[3][col]), c='magenta', linestyle='--')
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(seq_len-1, color='red', linestyle='--', linewidth=0.8)

        for i in range(2*seq_len):
            if i == seq_len-1:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.axvline(seq_len-1, color='red', linestyle='--', linewidth=0.8)
            else:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.axvline(i, color='blue', linestyle='--', linewidth=0.4, alpha=0.5)

    def pie_number_of_price_movements_binary(self, real_price_move: list, pred_price_move: list):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        def absolute_value_real(value):
            a = int(np.round(value/100.*len(real_price_move), 0))
            return a
        def absolute_value_pred(value):
            a = int(np.round(value/100.*len(pred_price_move), 0))
            return a
        ax1.pie([real_price_move.count(1), real_price_move.count(-1)],
            labels=['price go up', 'price go down'],
            autopct=absolute_value_real,
            colors=['#00DEFF', '#818181'],
            startangle=95,
            labeldistance=1.00);
        ax2.pie([pred_price_move.count(1), pred_price_move.count(-1)],
            labels=['price go up', 'price go down'],
            autopct=absolute_value_pred,
            colors=['#00DEFF', '#818181'],
            startangle=95,
            labeldistance=1.00);
        ax1.set_title('How many real price movements');
        ax2.set_title('How many predicted price movements');

    def plot_two_histograms(self, real_change_pct, pred_change_pct):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        k = 16
        ax1.hist(real_change_pct, bins=k, edgecolor='black', density=True);
        ax2.hist(pred_change_pct, bins=k, edgecolor='black', density=True);
        
        ax1.set_xlim(min(real_change_pct + pred_change_pct), max(real_change_pct + pred_change_pct));
        ax2.set_xlim(min(real_change_pct + pred_change_pct), max(real_change_pct + pred_change_pct));
        ax1.set_ylim(0,3.8)
        ax2.set_ylim(0,3.8)
        ax1.set_title('Real percentage price movements');
        ax2.set_title('Predicted percentage price movements');

    def pie_number_of_price_movements_ternary(self, real_price_move: list, pred_price_move: list):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        def absolute_value_real(value):
            a = int(np.round(value/100.*len(real_price_move), 0))
            return a
        def absolute_value_pred(value):
            a = int(np.round(value/100.*len(pred_price_move), 0))
            return a
        
        ax1.pie([real_price_move.count(1), real_price_move.count(-1), real_price_move.count(0)],
               labels=['price go up', 'price go down', 'nothing'],
               autopct=absolute_value_real,
               colors=['#00DEFF', '#818181', '#D0D0D0'],
               startangle=90,
               labeldistance=1.00);
        ax2.pie([pred_price_move.count(1), pred_price_move.count(-1), pred_price_move.count(0)],
               labels=['price go up', 'price go down', 'nothing'],
               autopct=absolute_value_pred,
               colors=['#00DEFF', '#818181', '#D0D0D0'],
               startangle=90,
               labeldistance=1.00);
        ax1.set_title('How many real price movements');
        ax2.set_title('How many predicted price movements');

    def create_metrics_df(self, accuracy, precision, recall, f1):
        return pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1]
        })