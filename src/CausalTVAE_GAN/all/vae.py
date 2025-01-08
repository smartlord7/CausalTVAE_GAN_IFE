import keras
import numpy as np
from keras import Model, Input
from keras.src.optimizers import Adam
from tensorflow.keras.layers import Dense, Reshape, Flatten, Lambda
import tensorflow as tf
from causal_aggregation_layer import CausalAggregationLayer
from causal_discovery import pad_adjacency_matrix
from transformer_encoder import transformer_encoder
from vae_loss_layer import VAELossLayer


def create_vae_model(shape, causal_graph, num_heads=4, ff_dim=64, dropout_rate=0.1, latent_dim=8):
    input_dim = shape[1]

    # Define larger dense units for encoder dynamically based on input_dim
    dense_units_encoder = [
        max(1024, input_dim * 16),  # 16x the input dimension or 1024, whichever is larger
        max(512, input_dim * 8),    # 8x the input dimension or 512, whichever is larger
        max(256, input_dim * 4),    # 4x the input dimension or 256, whichever is larger
        max(128, input_dim * 2),    # 2x the input dimension or 128, whichever is larger
        max(64, input_dim)          # Same as input dimension or 64, whichever is larger
    ]

    # Define larger dense units for decoder dynamically based on input_dim
    dense_units_decoder = [
        max(64, input_dim // 2),    # Half of the input dimension or 64, whichever is larger
        max(128, input_dim),        # Same as input dimension or 128, whichever is larger
        max(256, input_dim * 2),    # 2x the input dimension or 256, whichever is larger
        max(512, input_dim * 4),    # 4x the input dimension or 512, whichever is larger
        max(1024, input_dim * 8)    # 8x the input dimension or 1024, whichever is larger
    ]

    inputs = Input(shape=(input_dim,))

    # Initial Dense Layers in Encoder
    h = Dense(dense_units_encoder[0], activation='relu')(inputs)
    h = Dense(dense_units_encoder[1], activation='relu')(h)
    h = Dense(dense_units_encoder[2], activation='relu')(h)
    h = Dense(dense_units_encoder[3], activation='relu')(h)
    h = Dense(dense_units_encoder[4], activation='relu')(h)

    # Causal Aggregation Layer
    adj_matrix_padded = pad_adjacency_matrix(causal_graph, dense_units_encoder[4])
    h = CausalAggregationLayer(adj_matrix_padded)(h)

    # Determine the dynamic shape for Transformer Encoder
    seq_len = int(np.sqrt(h.shape[-1]))
    feature_dim = seq_len

    # Reshape for Transformer Encoder
    h = Reshape((1, seq_len, feature_dim))(h)

    # Apply Transformer Encoder
    h = transformer_encoder(h, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)

    # Flatten and Continue
    h = Flatten()(h)
    h = Dense(dense_units_encoder[4], activation='relu')(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    w = int(dense_units_decoder[0] / 2)
    h = int(dense_units_decoder[0] / w)
    decoder_h = Dense(dense_units_decoder[0], activation='relu')(z)
    decoder_h = Reshape((1, w, h))(decoder_h)

    # Apply Transformer Encoder in Decoder
    decoder_h = transformer_encoder(decoder_h, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
    decoder_h = Flatten()(decoder_h)
    decoder_h = Dense(dense_units_decoder[1], activation='relu')(decoder_h)
    decoder_h = Dense(dense_units_decoder[2], activation='relu')(decoder_h)
    decoder_h = Dense(dense_units_decoder[3], activation='relu')(decoder_h)
    decoder_h = Dense(dense_units_decoder[4], activation='relu')(decoder_h)
    outputs = Dense(input_dim, activation='linear')(decoder_h)

    outputs = VAELossLayer()([inputs, outputs, z_mean, z_log_var])

    vae = Model(inputs, outputs)
    vae.compile(optimizer=Adam())

    return vae


@keras.saving.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
