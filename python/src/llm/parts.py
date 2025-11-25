import tensorflow as tf
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocabSize, dModel):
        super().__init__()
        self.dModel = dModel
        self.embedding = tf.keras.layers.Embedding(vocabSize, dModel, mask_zero = True)
        self.posEncoding = self.positional_encoding(length = vocabSize, depth = dModel)

    def positional_encoding(self, length, depth):
        depth /= 2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth

        angleRates = 1 / (10000**depths)
        angleRads = positions * angleRates

        posEncoding = np.concatenate([np.sin(angleRads), np.cos(angleRads)], axis=-1)

        return tf.cast(posEncoding, dtype=tf.float32)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.dModel, tf.float32))
        x = x + self.posEncoding[tf.newaxis, :length, :]

        return x


class AddAndNormalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        return self.layerNorm(tf.add_n(inputs))