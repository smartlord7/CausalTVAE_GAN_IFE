from keras import Sequential, Input
from keras.src.layers import Dropout, Dense


def create_gan_discriminator(shape, dropout_rate=0.3, dense_units=(64, 32)):
    discriminator = Sequential([
        Input(shape=(shape[1],)),
        Dropout(dropout_rate),
        Dense(dense_units[0], activation='relu'),
        Dropout(dropout_rate),
        Dense(dense_units[1], activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')

    return discriminator