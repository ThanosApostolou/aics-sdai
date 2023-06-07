# This class file provides the fundamental code functionality for the 
# implementation of the Mutli-Head Attention mechanism which lies at the 
# core of the Transformer Model. This functional component computes the 
# attention between each pair of positions within the given sequence. It is 
# composed of multiple attention heads which aim at capturing different aspects
# of the input sequence.

# The MultiHeadAttention class initializes this structural module with input
# parameters and linear transformation layers. In particular, it performs the 
# the following series of computational operations:
# [i]:   Calculates the attention scores.
# [ii]:  Reshapes the input tensor in multiple heads.
# [iii]: Combines the attention outputs from all heads.

# The forward method computes the multi-head self attention, allowing the model
# to focus on some different aspects of the input sequence.

# The required input parameters for the class constructor are the following:
# [i]:  The dimensionality d_model of each instance of the input sequence.
# [ii]: The number num_heads of the attention heads realized by this component.

# Import required libraries.
import torch
import torch.nn as nn
import math

# Class Definition
class MultiHeadAttention(nn.Module):
    
    # Class Constructor.
    def __init__(self, d_model, num_heads):
        # Call the super class constructor.
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Set internal module parameters.
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Set the linear trandformation layers that will correspond to the
        # weight matrices Qw,Kw and Vw. An additional weight matrix Ow is 
        # required in order to provide the final output of the multi-head 
        # attention layer.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    # This method computes the scaled dot product-based attention scores.
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute the square matrix storing the pairwise scalded ttention 
        # scores.
        # The associated transposition operation on matrix K is applied in the
        # order of dimensions appearing as input arguments.
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            # All zero values will be replaced by -1e9 so that after the 
            # application of the softmax transfer function the corresponding
            # attention probability will be zero.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # Row-wise application of the softmax transfer funtion.    
        attn_probs = torch.softmax(attn_scores,-1)
        # Compute the final output of the attention layer which is calculated 
        # by multiplying with matrix V.
        output = torch.matmul(attn_probs, V)
        return output
    
    # This function splits the content of the input sequence to the various 
    # heads. It is assumed that the training input sequences are organized into
    # batches of batch size where each training sequence is composed of seq_len
    # instances so that each instance is of d_model dimensionality.
    # The operation that is being conducted in this function performes the 
    # following transformation to the input batch sequence:
    # [batch_size x seq_len x d_model]         ==> 
    # [batch_size x seq_len x num_heads x d_k] ==>
    # [batch_size x num_heads x seq_len x d_k]
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
    
    # This function combines the contents of the various heads into a batch of
    # training input sequences. In fact, it performs the inverse operation with
    # respect to split_heads method implemented above. 
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k  = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
    
    
    # Forward Pass Function.
    def forward(self, Q, K, V, mask=None):
        # Compute the contents of matrices Q, K and V after being passed through
        # the corresponding linear layers and the split heads function.
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        # Compute the pairwise attention values for each head.
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # Compute the final output of the attention layer.
        output = self.W_o(self.combine_heads(attn_output))
        return output