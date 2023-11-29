import tensorflow as tf
from tensorflow.python.keras import layers

class DotProductAttention(layers.Layer):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def call(self, decoder_hidden, encoder_outputs):
        attention_scores = tf.matmul(decoder_hidden, encoder_outputs, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=2)
        context_vector = tf.matmul(attention_weights, encoder_outputs)
        return context_vector, attention_weights