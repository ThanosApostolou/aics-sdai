import tensorflow as tf
from classes.transformer.GlobalSelfAttention import GlobalSelfAttention
from classes.transformer.FeedForward import FeedForward


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                  d_model=d_model)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
