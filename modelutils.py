import os
from tensorflow.keras.layers import Conv3D, Dropout, BatchNormalization, Activation, MaxPool3D, LSTM, Dense, TimeDistributed, Bidirectional, Reshape
from tensorflow.keras.models import Sequential
from utils import char_to_num


def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(BatchNormalization())  
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    model.add(TimeDistributed(Reshape((-1,))))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('models', 'new_best_weights2.weights.h5'))
    return model
