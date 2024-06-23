import numpy as np
import pandas as pd
from vanilla_transformer import Transformer
from autoformer import Autoformer
import tensorflow as tf

class ModelTrainer():
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

    def load_model_transformer(self, path: str, Xtest: np.array, config: dict):
        model = Transformer(config)
        _ = model(Xtest[:1], target=None, training=False)
        model.load_weights(path)
        return model

    def load_model_autoformer(self, path: str, Xtest: np.array, config: dict):
        model = Autoformer(config)
        _ = model(Xtest[:1], training=False)
        model.load_weights(path)
        return model

    def train_model_transformer(self, Xtrain, ytrain, Xval, yval, model_path, epochs, model, lr, config, Xtest):
        train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain)).batch(config['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices((Xval, yval)).batch(config['batch_size'])
        
        #model = Transformer(config)
        
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        
        best_val_loss = float('inf')
        best_epoch = -1
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            train_loss.reset_states()
            val_loss.reset_states()
    
            for batch, (xbatch, ybatch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    predictions = model(xbatch, ybatch, training=True)
                    loss = loss_fn(ybatch, predictions)
    
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss.update_state(loss)
            #model.save_weights(model_path)
            for xbatch, ybatch in val_dataset:
                predictions = model(xbatch, None, training=False)
                v_loss = loss_fn(ybatch, predictions)
                val_loss.update_state(v_loss)
            
            current_val_loss = val_loss.result().numpy()
            better_val_loss = False
            if current_val_loss < best_val_loss:
                better_val_loss = True
                best_val_loss = current_val_loss
                best_epoch = epoch
                model.save_weights(model_path)
    
            train_losses.append(train_loss.result().numpy())
            val_losses.append(current_val_loss)
            
            
            #print(f'Epoch: {epoch}, Loss: {train_loss.result().numpy():.3f}')
            if better_val_loss:
                print(f'Epoch: {epoch}, Loss: {train_loss.result().numpy()}, Val Loss: {current_val_loss}, Val loss improved')
            else:
                print(f'Epoch: {epoch}, Loss: {train_loss.result().numpy()}, Val Loss: {current_val_loss}')
    
        best_model = Transformer(config)
        _ = best_model(Xtest[:1], target=None, training=False)  # 'build' the model architecture so weights can be loaded
        best_model.load_weights(model_path)
        print(f'Model with the best validation loss at epoch {best_epoch} is reloaded')
        return best_model, best_epoch, train_losses, val_losses

    def train_model_autoformer(self, Xtrain, ytrain, Xval, yval, model_path, epochs, model, lr, config, Xtest):
        train_losses = []
        val_losses = []
        
        train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain)).batch(config['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices((Xval, yval)).batch(config['batch_size'])
        
        #model = Autoformer(config)
        
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        
        best_val_loss = float('inf')
        best_epoch = -1
        #model_path = 'saved_models/test0.h5'
        for epoch in range(epochs):
            train_loss.reset_states()
            val_loss.reset_states()
    
            for batch, (xbatch, ybatch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    predictions = model(xbatch, training=True)
                    loss = loss_fn(ybatch, predictions)
    
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss.update_state(loss)
            for xbatch, ybatch in val_dataset:
                predictions = model(xbatch, training=False)
                v_loss = loss_fn(ybatch, predictions)
                #print('vloss:', v_loss)
                val_loss.update_state(v_loss)
            
            current_val_loss = val_loss.result().numpy()
            #current_val_loss = 0
            better_val_loss = False
            if current_val_loss < best_val_loss:
                better_val_loss = True
                best_val_loss = current_val_loss
                best_epoch = epoch
                model.save_weights(model_path)
    
    
            train_losses.append(train_loss.result().numpy())
            val_losses.append(current_val_loss)
            if better_val_loss:
                print(f'Epoch: {epoch}, Loss: {train_loss.result().numpy()}, Val Loss: {current_val_loss}, Val loss improved')
            else:
                print(f'Epoch: {epoch}, Loss: {train_loss.result().numpy()}, Val Loss: {current_val_loss}')
        
        best_model = Autoformer(config)
        _ = best_model(Xtest[:1])
        best_model.load_weights(model_path)
        print(f'Model with the best validation loss at epoch {best_epoch} is reloaded')
        return best_model, best_epoch, train_losses, val_losses