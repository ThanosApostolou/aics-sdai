import tensorflow as tf

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim):
        super(LuongAttention, self).__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        # Get the dimensions of the query and value tensors
        _, self.query_seq_len, self.query_dim = input_shape

    def call(self, query, values):
        # Calculate the score by computing the dot product between query and values
        query_with_time_axis = tf.expand_dims(query, 1)
        scores = tf.matmul(query_with_time_axis, values, transpose_b=True)
        scores = tf.squeeze(scores, axis=1)  # Remove the extra dimension

        # Compute attention weights using the softmax function
        attention_weights = tf.nn.softmax(scores, axis=1)

        # Calculate the context vector by weighted sum of values
        context_vector = tf.reduce_sum(values * tf.expand_dims(attention_weights, axis=1), axis=1)

        return context_vector, attention_weights
