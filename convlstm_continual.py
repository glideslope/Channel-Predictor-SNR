import os
from scipy.io import loadmat
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Load data
data_dir = 'C:/Users/netlab/Desktop/2nd Paper/Code/Normarlized'
files = os.listdir(data_dir)
log_dir = "C:/Users/netlab/Desktop/2nd Paper/Code/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# SNR values
snr_values = [0, 5, 10, 15, 20, 25]  # SNR 값들

# Define the model
model = Sequential()

# Encoder
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(None, 624, 14, 2)))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

# Decoder
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=2, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.compile(optimizer='adam', loss='mse')


for snr in snr_values:
    files_snr = [f for f in files if "snr_" + str(snr) + "_" in f]  
    X = []
    Y = []
    seq_length = 30

    for i in range(len(files_snr) - seq_length*2):
        X_seq = []
        Y_seq = []
        for j in range(seq_length):
            X_seq.append(loadmat(os.path.join(data_dir, files_snr[i + j])).get('chan_Nor'))
            Y_seq.append(loadmat(os.path.join(data_dir, files_snr[i + seq_length + j])).get('chan_Nor'))
        X.append(X_seq)
        Y.append(Y_seq)

    # Convert list to numpy array
    X = np.array(X)  # Shape: (num_samples, 30, 624, 14, 2)
    Y = np.array(Y)  # Shape: (num_samples, 30, 624, 14, 2)

    # Train the model
    model.fit(X, Y, epochs=10, verbose=1, callbacks=[tensorboard_callback])