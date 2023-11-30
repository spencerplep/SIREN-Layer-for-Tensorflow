
import tensorflow as tf
import keras
from siren_init import SirenInitializer

class Siren(keras.layers.Layer):
    def __init__(self, units=32, first_layer_init=False):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=SirenInitializer,
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        
        return tf.math.sin(tf.matmul(inputs, self.w) + self.b)