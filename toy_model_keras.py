import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import LearningRateScheduler
import keras_tuner as kt
import math
import matplotlib.pyplot as plt
import numpy as np
from utils_mrf import read_mrf_dict
from datetime import datetime
#tf.compat.v1.enable_eager_execution()

# learning rate schedule
def step_decay(epoch, initial_lrate=0.1, drop=0.5, epochs_drop=10.0):
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def build_and_compile_model(n_outputs, kernel_regularizer=None, dropout_rate=None, learning_rate=0.0001,
                            activation="relu"):
    if dropout_rate is None:

        model = keras.Sequential([
            layers.Dense(100, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dense(40, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dense(30, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dense(30, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dense(n_outputs, kernel_regularizer=kernel_regularizer),
        ])

    else:
        model = keras.Sequential([
            layers.Dense(100, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(40, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(30, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(30, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(n_outputs, kernel_regularizer=kernel_regularizer),
            layers.Dropout(dropout_rate)
        ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model


def build_and_compile_model_simple():
    global n_outputs
    model = keras.Sequential([
        layers.Dense(100, activation="elu", name="Dense_1"),
        layers.Dense(40, activation="elu", name="Dense_2"),
        layers.Dense(30, activation="elu", name="Dense_3"),
        layers.Dense(30, activation="elu", name="Dense_4"),
        layers.Dense(n_outputs, name="Dense_Out"),
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.01))
    return model


def build_and_compile_model_BN(n_outputs):
    model = keras.Sequential([
        layers.Dense(100),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(40),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(30),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(30),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(n_outputs)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.0001))
    return model


def build_and_compile_model_hp(hp):
    global n_outputs
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_dropout_rate = hp.Choice('dropout_rate', values=[0.0])
    hp_activation = hp.Choice('activation', values=["elu", "relu", "tanh"])
    return build_and_compile_model(n_outputs, dropout_rate=hp_dropout_rate, learning_rate=hp_learning_rate,
                                   activation=hp_activation)


def plot_loss(history, epoch_start=0):
    ymax_train = np.max(history.history['loss'][epoch_start:])
    ymax_val = np.max(history.history['val_loss'][epoch_start:])
    ymax = np.maximum(ymax_train, ymax_val)
    plt.plot(history.history['loss'][epoch_start:], label='loss')
    plt.plot(history.history['val_loss'][epoch_start:], label='val_loss')
    plt.ylim([0, ymax])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)


FF_list=list(np.arange(0.,1.05,0.05))
dictfile = "mrf175_SimReco2.dict"
keys,signal=read_mrf_dict(dictfile,FF_list)

Y_TF = np.array(keys)
real_signal = signal.real
imag_signal = signal.imag

X_TF = np.concatenate((real_signal,imag_signal),axis=1)

from sklearn.preprocessing import MinMaxScaler,StandardScaler

input_scaler  =StandardScaler()
output_scaler = MinMaxScaler()

X_TF = np.transpose(input_scaler.fit_transform((X_TF)))
Y_TF = np.transpose(output_scaler.fit_transform(Y_TF))




tuner = kt.Hyperband(build_and_compile_model_hp,
                    objective='val_loss',
                    max_epochs=1500,
                    factor=3,
                    directory='my_dir',
                    project_name='intro_to_kt',overwrite=True)
n_outputs=Y_TF.shape[0]

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(np.transpose(X_TF), np.transpose(Y_TF), validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal dropout rate is {best_hps.get('dropout_rate')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}. The optimal activation is {best_hps.get('activation')}
""")


callbacks_list = []

model_keras = tuner.get_best_models()[0]

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks_list.append(tensorboard_callback)

history = model_keras.fit(
    np.transpose(X_TF), np.transpose(Y_TF),batch_size=1024,shuffle=True,
    validation_split=0.2,
    verbose=1, epochs=1000,callbacks=callbacks_list)

output_scaler.inverse_transform(model_keras.predict(X_TF[:,10].reshape(1,-1)))
