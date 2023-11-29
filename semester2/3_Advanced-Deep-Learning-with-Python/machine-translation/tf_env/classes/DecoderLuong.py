import tensorflow as tf
from classes.LuongAttention import LuongAttention

import tensorflow as tf


class DecoderLuong(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(DecoderLuong, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = LuongAttention(hidden)(enc_output, hidden)


        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # ------------ COMPUTING EQUATION (3) ----------------------#
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_decoder_state(self, encoder_outputs, initial_state):
        # You should define how to initialize the decoder state here
        # Use the encoder_outputs and initial_state as needed
        # Example: You can simply return the initial_state
        decoder_state = initial_state
        return decoder_state
