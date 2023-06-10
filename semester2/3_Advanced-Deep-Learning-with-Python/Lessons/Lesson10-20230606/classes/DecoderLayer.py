# This class file provides the fundamental code functionality for the 
# implementation of the Decoder component which lies at the core of the 
# Transformer Model. In particular, this functional component realizes the 
# Decoder Layer which consists of two Multi-Head Attention layers, one 
# Position-Wise Feed-Forward layer, and three Layer Normalization layers.

# The DecoderLayer class initializes with the required input parameters and the 
# following structural components:
# [i]:    One Multi Head Attention module for masked self-attention.
# [ii]:   One Multi Head Attention module for cross-attention.
# [iii]:  One Position-Wise Feed Forward Neural Network module.
# [iv]:   Three Layer Normalization modules.
# [v]:    A Dropout Layer.

# The forward method computes the final decoder layer output by performing the
# following sequential computational operations:
# [i]:   Calculate the masked self-attention output and add it to the current 
#        input tensor, followed by the dropout and layer normalization
#        activities.
# [ii]:  Compute the cross-attention output between the decoder and encoder 
#        outputs and add it to the normalized self-attention output, followed 
#        by the dropout and layer normalization activities.
# [iii]: Calculate the position-wise feed-forward neural network output and
#        combine it with the normalized cross-attention output, followed by the
#        dropout and layer normalization activities.
# [iv]:  Ruturn the processed tensor.

# The aforementioned operations enable the decoder to generate target sequences
# based on each instance of the input sequence and the corresponding encoder
# output.

# The required input parameters for the class constructor are the following:
# [i]:   The dimensionality d_model of each instance of the input sequence.
# [ii]:  The number num_heads of the attention heads realized by this 
#        component.
# [iii]: The intrinsic dimensionality d_ff of the combined linear 
#        transformation layer which is realized by this component.
# [iv]:  The percentage of connections that will be eliminated when information
#        flows through each dropout layer.


# Import required libraries.
import torch.nn as nn
from classes.MultiHeadAttention import MultiHeadAttention
from classes.PositionWiseFeedForward import PositionWiseFeedForward
 
class DecoderLayer(nn.Module):
    
    # Class Constructor.
    def __init__(self, d_model, num_heads, d_ff, dropout):
        # Call the super class constructor.
        super(DecoderLayer, self).__init__()
        # Add the Masked Self Attention structural component of the Decoder.
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Add the Cross Attention structural component of the Decoder.
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # Add the Point-Wise Feed Forward Neural Network component of the 
        # Decoder.
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # Add the first Normalization Layer component of the Decoder.
        self.norm1 = nn.LayerNorm(d_model)
        # Add the second Normalization Layer component of the Decoder.
        self.norm2 = nn.LayerNorm(d_model)
        # Add the third Normalization Layer component of the Decoder.
        self.norm3 = nn.LayerNorm(d_model)
        # Add the Dropout Layer of the Decoder.
        self.dropout = nn.Dropout(dropout)
        
    # Forward Pass Function
    def forward(self, x, enc_output, source_mask, target_mask):
        # Take into consideration that the current input information is 
        # repeated for each one of the fundamental matrices Q, K and V of the 
        # multi-head masled self-attention layer. This is the self-attention 
        # layer which operates based on the target mask parameter.
        attn_output = self.self_attn(x, x, x, target_mask)
        # The first normalization layer performs dropout on the multi-head 
        # self-attention output and the final normalization output is generated
        # by considering a residual link which is directly connected to the 
        # currently presented input tensor.
        x = self.norm1(x + self.dropout(attn_output))
        # Compute the output of the cross-attention layer between the decoder
        # and encoder outputs. This attention layer operates based on the 
        # source mask parameter. Take into consideration that the current input
        # information is feeded exclusively as input for the Q matrix while the
        # current encoder output is feeded as input for the K and V matrices of
        # this attention layer.
        attn_output = self.cross_attn(x, enc_output, enc_output, source_mask)
        # The second normalization layer performs dropout on the multi-head 
        # cross-attention output and the final normalization output is 
        # generated by considering a residual link which is directly connected
        # to the currently presented input tensor.
        x = self.norm2(x + self.dropout(attn_output))
        # Compute the output of point-wise feed-forward neural-network layer.
        ff_output = self.feed_forward(x)
        # The third normalization layer performs dropout on the feed-forward
        # network output and the final normalization output is once again 
        # generated by considering a residual link which is directly connected 
        # to the currently presented input tensor.
        x = self.norm3(x + self.dropout(ff_output))
        return x