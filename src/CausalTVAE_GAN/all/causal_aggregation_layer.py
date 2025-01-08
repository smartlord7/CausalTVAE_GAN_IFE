import keras
from keras import Layer
import tensorflow as tf

@keras.saving.register_keras_serializable()
class CausalAggregationLayer(Layer):
    def __init__(self, adj_matrix, **kwargs):
        super(CausalAggregationLayer, self).__init__(**kwargs)
        self.adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

    def call(self, inputs):
        projected_inputs = tf.matmul(inputs, self.adj_matrix)
        return projected_inputs

    def get_config(self):
        config = super(CausalAggregationLayer, self).get_config()
        config.update({"adj_matrix": self.adj_matrix})
        return config

    @classmethod
    def from_config(cls, config):
        adj_matrix = config.pop("adj_matrix")['config']['value']
        return cls(adj_matrix=adj_matrix, **config)