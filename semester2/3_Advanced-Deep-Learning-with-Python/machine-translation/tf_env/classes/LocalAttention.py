import tensorflow as tf
from tensorflow.keras.layers import layers

class LocalAttention(layers.Layer):
    def __init__(self, hidden_size, window_size):
        super(LocalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.attn = layers.Dense(hidden_size)
        self.v = layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        energy = self.v(tf.tanh(self.attn(decoder_hidden + encoder_outputs)))
        attention_scores = tf.nn.softmax(energy, axis=1)

        # Apply local window to attention scores
        start_idx = max(0, decoder_hidden.shape[1] - self.window_size // 2)
        end_idx = min(seq_len, start_idx + self.window_size)
        local_attention_scores = attention_scores[:, start_idx:end_idx, :]

        context_vector = tf.reduce_sum(local_attention_scores * encoder_outputs[:, start_idx:end_idx, :], axis=1)
        return context_vector, local_attention_scores
