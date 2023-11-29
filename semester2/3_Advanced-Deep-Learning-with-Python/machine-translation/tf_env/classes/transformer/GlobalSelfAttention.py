import tensorflow as tf

class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.1, **kwargs):
        super(GlobalSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model  # Correct the argument name
        self.dropout_rate = dropout_rate
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                      dropout=dropout_rate, key_dim=512)  # Correct the argument names
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = tf.keras.layers.add([x, attn_output])
        x = self.layer_norm(x)
        return x
