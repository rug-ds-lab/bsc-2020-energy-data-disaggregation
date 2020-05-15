from tensorflow_core.python import keras
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, Reshape


# TODO: setdense layer to -3
def DAE_model(sequence_length):
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dropout(0.2))
    model.add(Dense((sequence_length - 0) * 8, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense((sequence_length - 0) * 8, activation='relu'))

    model.add(Dropout(0.2))

    # 1D Conv
    model.add(Reshape(((sequence_length-0), 8)))
    model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model



def RNN_model(sequence_length):
    '''Creates the RNN module described in the paper
	'''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))

    # Bi-directional LSTMs
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat'))

    # Fully Connected Layers
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model