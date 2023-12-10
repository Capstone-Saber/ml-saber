# ============================================================================================
# Haikal's Playground
# Model, with Python Blyad
# ============================================================================================

import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import to_datetime
from sklearn.preprocessing import MinMaxScaler

# Callback function to stop the training
# Callback function untuk menghentikan training secara otomatis
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('mae') < 0.2:
            print("MAE Condition Fulfilled")
            self.model.stop_training = True

# Create Sequences
def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data[i:(i + n_past), 0])
        y.append(data[(i + n_past):(i + n_past + n_future), 0])
    return np.array(X), np.array(y)

# To Test The Model
# ==========================================================================

# ==========================================================================

def my_Model():
    # Preprocessing The Data
    # ======================
    data = pd.read_csv('MBC GO Energy Consumption Milestone 2.csv')
    data['Date'] = to_datetime(data['Date'], errors='coerce')
    energy_data = data[['Date', 'day_energy']].dropna()
    daily_energy = energy_data.groupby('Date').sum().reset_index()
    expected_dates = pd.date_range(start=daily_energy['Date'].min(), end=daily_energy['Date'].max())
    missing_dates = expected_dates.difference(daily_energy['Date'])
    missing_dates_list = missing_dates.to_list()
    
    # Normalize The Series
    # ====================
    daily_energy['day_of_week'] = daily_energy['Date'].dt.dayofweek
    scaler = MinMaxScaler()
    daily_energy['scaled_energy'] = scaler.fit_transform(daily_energy[['day_energy']])
    
    # Number of past days to use for predicting the future
    n_past = 30
    # Number of future days to predict
    n_future = 7
    # Prepare input and output sequences
    x_train, y_valid = create_sequences(daily_energy[['scaled_energy']].values, n_past, n_future)
    # Split data into training and testing sets
    train_size = int(len(x_train) * 0.8)
    X_train, X_test = x_train[:train_size], x_train[train_size:]
    y_train, y_test = y_valid[:train_size], y_valid[train_size:]

    # SCATCH
    BATCH_SIZE = 32

    # Define The Model
    model = tf.keras.models.Sequential([
        # Whatever your first layer is, the input shape will be (N_PAST = 30, N_FEATURES = 7)
        tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, strides=1,
            activation="relu",input_shape=(n_past, 1)
        ),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.summary()
    
    # Compile and Train The Model
    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE, verbose=1, callbacks=[myCallback()])

    mae = history.history['mae']
    loss = history.history['loss']
    epochs = range(len(mae))

    # Plot mae & validation mae
    plt.figure()
    plt.plot(epochs, mae, '--b')
    plt.plot(epochs, loss, '--r')
    plt.xlabel("Epochs")
    plt.legend(["mae", "loss"])
    plt.show()

    return model

# The code below is to save your model as a .h5 file.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = my_Model()
    model.save("saved_model/model_py.h5")

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    tflite_model_path = "saved_model/model_py.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)