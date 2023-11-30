
import tensorflow as tf
import keras

class SirenInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, fan_in=32):
        return tf.random.uniform(shape, -tf.math.sqrt(6/float(fan_in)),
                                 tf.math.sqrt(6/float(fan_in)),
                                 dtype)
