import tensorflow as tf
from classes.DotProductAttention import DotProductAttention


class DecoderDotProduct(tf.keras.Model):
    def __init__(self, vocab_size, timesteps_num, embedding_dim, decoder_dim, **kwargs):
        super(DecoderDotProduct, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim

        # Embedding layer to convert input tokens to embeddings
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   input_length=timesteps_num)

        # RNN layer (GRU) with return sequences and return state
        self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True,
                                       return_state=True)

        self.attention = DotProductAttention()

        # Dense layer for generating output logits
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, encoder_outputs):
        # Embed the input tokens
        x = self.embedding(x)

        # Pass through the RNN layer
        x, state = self.rnn(x, initial_state=state)

        # Apply DotProductAttention using the encoder outputs
        context_vector, attention_weights = self.attention(x, encoder_outputs)

        # Concatenate the context vector with the decoder outputs
        x = tf.concat([x, context_vector], axis=-1)

        # Pass through the dense layer to generate logits
        output_logits = self.dense(x)

        return output_logits, state

    def initialize_decoder_state(self, encoder_outputs, initial_state):
        # You should define how to initialize the decoder state here
        # Use the encoder_outputs and initial_state as needed
        # Example: You can simply return the initial_state
        decoder_state = initial_state
        return decoder_state
