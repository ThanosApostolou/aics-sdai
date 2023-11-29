import tensorflow as tf
from tensorflow.keras import layers

class Attention(layers.Layer):
    def __init__(self, decoder_dim, attention_units):
        super(Attention, self).__init__()
        self.decoder_dim = decoder_dim
        self.attention_units = 128
        self.W_a = layers.Dense(self.attention_units)
        self.W_b = layers.Dense(self.attention_units)
        self.V = layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
        score = self.V(tf.nn.tanh(self.W_a(decoder_hidden_with_time_axis) + self.W_b(encoder_outputs)))
        attention_scores = tf.nn.softmax(score, axis=1)
        context_vector = attention_scores * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_scores
