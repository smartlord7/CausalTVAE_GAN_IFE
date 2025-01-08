import keras
from keras import Layer
import tensorflow as tf

@keras.saving.register_keras_serializable()
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        original_inputs, reconstructed_outputs, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(tf.square(original_inputs - reconstructed_outputs), axis=-1)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss, axis=-1) * -0.5
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return reconstructed_outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1]
