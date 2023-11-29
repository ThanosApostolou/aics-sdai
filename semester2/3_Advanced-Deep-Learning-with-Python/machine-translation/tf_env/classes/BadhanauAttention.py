# Define BahdanauAttention (you can implement it in a separate class):
import tensorflow as tf
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        # Calculate attention scores
        query = self.W1(decoder_hidden)
        values = self.W2(encoder_outputs)
        scores = self.V(tf.nn.tanh(query + values))

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=1)

        # Calculate the context vector
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Modify DecoderDotProduct class:
class DecoderBahdanau(tf.keras.Model):
    def __init__(self, vocab_size, timesteps_num, embedding_dim, decoder_dim, attention_units, **kwargs):
        super(DecoderBahdanau, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim

        # Embedding layer to convert input tokens to embeddings
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   input_length=timesteps_num)

        # RNN layer (GRU) with return sequences and return state
        self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True,
                                       return_state=True)

        # BahdanauAttention mechanism
        self.attention = BahdanauAttention(attention_units)

        # Dense layer for generating output logits
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, encoder_outputs):
        # Embed the input tokens
        x = self.embedding(x)

        # Pass through the RNN layer
        x, state = self.rnn(x, initial_state=state)

        # Apply BahdanauAttention using the encoder outputs
        context_vector, attention_weights = self.attention(x, encoder_outputs)

        # Concatenate the context vector with the decoder outputs
        x = tf.concat([x, context_vector], axis=-1)

        # Pass through the dense layer to generate logits
        output_logits = self.dense(x)

        return output_logits, state
