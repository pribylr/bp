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

    def plot_real_predicted_sequences(self, col: str, real_candles: list, pred_candles: list, seq_len: int):
        """
        display four line plots
        each plot has vertical line in middle
        left of the line - input sequence (actual price of an asset)
        right of the line - 1) real output sequence that follows the input and 2) predicted output sequence
        """
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
            labels=['price increased', 'price decreased'],
            autopct=absolute_value_real,
            colors=['#00C98D', '#E9413A'],
            startangle=95,
            labeldistance=1.00);
        ax2.pie([pred_price_move.count(1), pred_price_move.count(-1)],
            labels=['price increased', 'price decreased'],
            autopct=absolute_value_pred,
            colors=['#00C98D', '#E9413A'],
            startangle=95,
            labeldistance=1.00);
        ax1.set_title('How many real price changes');
        ax2.set_title('How many predicted price changes');

    def plot_two_histograms(self, real_change_pct, pred_change_pct):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        k = 16
        counts1, bins1, patches1 = ax1.hist(real_change_pct, bins=k, edgecolor='black', density=True);
        counts2, bins2, patches2 = ax2.hist(pred_change_pct, bins=k, edgecolor='black', density=True);
        
        ax1.set_xlim(min(real_change_pct + pred_change_pct), max(real_change_pct + pred_change_pct));
        ax2.set_xlim(min(real_change_pct + pred_change_pct), max(real_change_pct + pred_change_pct));
        ax1.set_ylim(0, max(counts1 + counts2) + 0.05*max(counts1 + counts2))
        ax2.set_ylim(0, max(counts1 + counts2) + 0.05*max(counts1 + counts2))
        ax1.set_title('Real percentage price changes');
        ax2.set_title('Predicted percentage price changes');

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
               labels=['price increased', 'price decreased', 'nothing'],
               autopct=absolute_value_real,
               colors=['#00C98D', '#E9413A', '#D0D0D0'],
               startangle=90,
               labeldistance=1.00);
        ax2.pie([pred_price_move.count(1), pred_price_move.count(-1), pred_price_move.count(0)],
               labels=['price increased', 'price decreased', 'nothing'],
               autopct=absolute_value_pred,
               colors=['#00C98D', '#E9413A', '#D0D0D0'],
               startangle=90,
               labeldistance=1.00);
        ax1.set_title('How many real price changes');
        ax2.set_title('How many predicted price changes');

    def create_metrics_df(self, accuracy, precision, recall, f1):
        return pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1]
        })

    def plot_metrics_ternary(self, thresholds: list, accuracies: list, precisions: list, recalls: list, f1s: list):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.plot(thresholds, accuracies, label='accuracy', color='black')
        ax2.plot(thresholds, precisions, label='precision', c='#A315E6')
        ax2.plot(thresholds, recalls, label='recall', c='#FF9B00')
        ax2.plot(thresholds, f1s, label='f1 score', c='#66DA10')
        # ax1.axhline(y=0.333, label='0.333', linestyle='--', c='blue')
        # ax2.axhline(y=0.333, label='0.333', linestyle='--', c='blue')
        ax1.set_xlim(0, 1)
        ax2.set_xlim(0, 1)
        # ax1.set_ylim(min(accuracies+precisions+recalls+f1s)-0.02, max(accuracies+precisions+recalls+f1s)+0.02)
        # ax2.set_ylim(min(accuracies+precisions+recalls+f1s)-0.02, max(accuracies+precisions+recalls+f1s)+0.02)
        ax1.set_title('Accuracy for different price movement thresholds')
        ax2.set_title('Precision, recall, f1 score for different price movement thresholds')
        ax1.set_xlabel('Price move percentage threshold')
        ax2.set_xlabel('Price move percentage threshold')
        ax1.set_ylabel('Accuracy value')
        ax2.set_ylabel('Metrics values')
        ax1.legend();
        ax2.legend();

    def plot_metrics_binary(self, thresholds: list, accuracies: list, precisions: list, recalls: list, f1s: list, original_accuracy: float, original_precision: float, original_recall: float, original_f1: float):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.plot(thresholds, accuracies, label='accuracy', color='black')
        ax2.plot(thresholds, precisions, label='precision', alpha=0.8, c='#A315E6')
        ax2.plot(thresholds, recalls, label='recall', alpha=0.8, c='#FF9B00')
        ax2.plot(thresholds, f1s, label='f1 score', alpha=0.8, c='#66DA10')
        ax1.axhline(y=original_accuracy, label=f'accuracy with no threshold considered ({original_accuracy:.2f})', c='blue', linestyle='--')
        ax2.axhline(y=original_precision, label=f'precision with no threshold considered ({original_precision:.2f})', c='#A315E6', linestyle='--', alpha=0.7)
        ax2.axhline(y=original_recall, label=f'recall with no threshold considered ({original_recall:.2f})', c='#FF9B00', linestyle='--', alpha=0.7)
        ax2.axhline(y=original_f1, label=f'f1 score with no threshold considered ({original_f1:.2f})', c='#66DA10', linestyle='--', alpha=0.7)
        ax1.set_title('Accuracy for different price movement thresholds')
        ax2.set_title('Precision, recall, f1 score for different price movement thresholds')
        ax1.set_xlabel('Price move percentage threshold')
        ax2.set_xlabel('Price move percentage threshold')
        ax1.set_ylabel('Accuracy value')
        ax2.set_ylabel('Metrics values')
        ax1.set_xlim(0, 1)
        ax2.set_xlim(0, 1)
        # ax1.set_ylim(min(accuracies+precisions+recalls+f1s)-0.02, max(accuracies+precisions+recalls+f1s)+0.02)
        # ax2.set_ylim(min(accuracies+precisions+recalls+f1s)-0.02, max(accuracies+precisions+recalls+f1s)+0.02)
        ax1.legend(loc='upper left');
        ax2.legend(loc='upper left', fontsize=7);