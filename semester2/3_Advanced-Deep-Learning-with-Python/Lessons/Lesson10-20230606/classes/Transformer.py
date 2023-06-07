# This class file provides the fundamental code functionality for the
# implementation of the complete Transformer Model by combining the previously
# defined modules.

# The Transformer class is initialized with all the required input parameters.
# At the same time the class constructor initializes the various buildibg
# blocks of the transformer architecture including:
# [i]:   Embedding Layers for source and target sequences.
# [ii]:  Positional Encoding Layer.
# [iii]: Encoder Layer
# [iv]:  Decoder Layer
# [v]:   Linear Layer for projecting the decoder output.
# [vi]:  Dropout Layer.

# Structural components [i] - [iv] are stacked together to create the composite
# transformer components.

# The generate_mask method creates binary masks for source and target
# sequences to ignore padding tokens and prevent the decoder from attending to
# future tokens.

# The forward method computes the final output of the transformer model by
# performing the following sequential computational operations:
# [i]:  Generate source and target masks utilizing the generate_mask class
#       method.
# [ii]  Compute the embedding for each source and target sequence instance.
#       This step is completed by applying positional encoding and dropout.
# [iii] Process the source sequence through the encoder layers, updating the
#       encoder output tensor.
# [iv]  Process the target sequence through the decoder layers utilizing the
#       the previously computed encoder output and generated masks, updating
#       the decoder output tensor.
# [v]   Apply the linear projection layer to the decoder output in order to
#       obtain output logits.

# These computational steps enable the Transformer Model to process input
# sequences and generate the desired output sequences based on the combined
# functionality of its components.

# The required input parameters for the class constructor are the following:
# [i]:    The source vocabulary size.
# [ii]:   The target vocabulary size.
# [iii]:  The dimensionality d_model of each instance of the input sequence.
# [iv]:   The number num_heads of the attention heads realized by this
#         component.
# [v]:    The number num_layers of stacked encoder and decoder components that
#         will ultimately form the transformer architecture.
# [vi]:   The intrinsic dimensionality d_ff of the combined linear
#         transformation layer which is realized by this component.
# [vii]:  The maximum sequence length.
# [viii]: The percentage of connections that will be eliminated when
#         information flows through each dropout layer.

# Import required libraries.
import torch
import torch.nn as nn
from classes.EncoderLayer import EncoderLayer
from classes.DecoderLayer import DecoderLayer
from classes.PositionalEncoding import PositionalEncoding


class Transformer(nn.Module):

    # Class Constructor.
    def __init__(self, source_vocab_size, target_vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_length, dropout):
        # Call the super class constructor.
        super(Transformer, self).__init__()
        # Add the Embedding Layer of the Encoder.
        self.encoder_embedding = nn.Embedding(source_vocab_size, d_model)
        # Add the Embedding Layer of the Decoder.
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        # Add the Positional Encoding Layer that will be used for both the
        # Encoder and the Decoder component of the Transformer.
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Add the Encoder Layer components of the Transformer.
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Add the Deocder Layer components of the Transformer.
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Add the Linear Layer component that will project the decoder output
        # to symbols of the target vocabulary.
        self.fc = nn.Linear(d_model, target_vocab_size)

        # Add the Dropout Layer that will be applied after the computation of
        # the positional encoding either for the encoder or for the decoder.
        self.dropout = nn.Dropout(dropout)

    # Mask Generator Function.
    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        seq_length = target.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        target_mask = target_mask & nopeak_mask
        return source_mask, target_mask

    # Forward Pass Function.
    def forward(self, source, target):
        # Compute the source and target masks.
        source_mask, target_mask = self.generate_mask(source, target)
        # Compute the embedding for the source sequence.
        source_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(source)))
        # Compute the embedding for the target sequence.
        target_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(target)))
        # Compute the final encoder output.
        # Initially set the encoder output equal to the embedded version of the
        # source patterns. Subsequently, pass the information through all the
        # available encoder layers.
        encoder_output = source_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)

        # Compute the final decoder output.
        # Initially set the decoder output equal to the embedded version of the
        # target patterns. Subsequently, pass the information through all the
        # available decoder layers.
        decoder_output = target_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output, encoder_output, source_mask, target_mask)

        # Pass the final decoder output through the linear projection layer.
        output = self.fc(decoder_output)

        # Return the final decoder output.
        return output
