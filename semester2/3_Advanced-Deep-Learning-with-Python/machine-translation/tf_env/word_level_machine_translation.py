# This is the main script file facilitating all the available functionality for
# the machine translation task at the word level.

# Import all required Python frameworks
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from classes.BadhanauAttention import BahdanauAttention
from classes.encoder import Encoder
from classes.decoder import Decoder
from classes.data_preparation import DataPreparation
from classes.transformer.DecoderLayer import DecoderLayer
from classes.transformer.EncoderLayer import EncoderLayer
from classes.transformer.Transformer import Transformer
from classes.decoderLSTM import DecoderLSTM
from classes.encoderLSTM import EncoderLSTM
from classes.decoderDotProduct import DecoderDotProduct


# -----------------------------------------------------------------------------
#                       FUNCTIONS DEFINITION
# -----------------------------------------------------------------------------
# This function reports fundamental tensor shape configurations for the encoder
# and decoder objects.
def report_encoder_decoder():
    for encoder_in, decoder_in, decoder_out in data_preparation.train_dataset:
        encoder_state = encoder.init_state(data_preparation.batch_size)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state)
        break
    print("=======================================================")
    print("Encoder Input:           :{}".format(encoder_in.shape))
    print("Encoder Output:          :{}".format(encoder_out.shape))
    print("Encoder State:           :{}".format(encoder_state.shape))
    print("=======================================================")
    print("Decoder Input:           :{}".format(decoder_in.shape))
    print("Decoder Output           :{}".format(decoder_pred.shape))
    print("Decoder State            :{}".format(decoder_state.shape))
    print("=======================================================")


# -----------------------------------------------------------------------------
# This function defines the loss function to be minimized during training.
# Given the fact that input and target sequences for both languages have been
# padded in order to reflect the maximum sequence length for the respective
# language, it is imperative to avoid considering equality of pad words between
# the true labels and the estimated predictions. To this end, the utilized loss
# function masks the estimated predictions with the true labels, so that padded
# positions on the label are also removed from the predictions. Thus, the loss
# function is actually evaluated exclusively on the non-zero elements of both
# labels and predictions.
def loss_fn(ytrue, ypred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(ytrue, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = scce(ytrue, ypred, sample_weight=mask)
    return loss, loss


# -----------------------------------------------------------------------------
# This function implements the actual training process for the neural model
# according to the Teacher Forcing technique where the input to the decoder
# is the actual ground truth output instead of the prediction from the previous
# timestep.
# -----------------------------------------------------------------------------
@tf.function
def train_step(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state)
        loss = loss_fn(decoder_out, decoder_pred)
    variables = (encoder.trainable_variables + decoder.trainable_variables)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


# ---------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# This function is used to randomly sample a single English sentence from the
# dataset and used the model trained so far to predict the French sentence. Mind
# that the sampling process does not discriminate between training and testing
# patterns.
# -----------------------------------------------------------------------------
def predict(encoder, decoder, type):
    random_id = np.random.choice(len(data_preparation.input_english_sentences))
    print("Input Sentence: {}".format(" ".join(data_preparation.input_english_sentences[random_id])))
    print("Target Sentence: {}".format(" ".join(data_preparation.target_french_sentences[random_id])))
    encoder_in = tf.expand_dims(data_preparation.input_data_english[random_id], axis=0)
    encoder_state = encoder.init_state(1)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state
    decoder_in = tf.expand_dims(tf.constant([data_preparation.french_word2idx["BOS"]]), axis=0)
    pred_sent_fr = []
    while True:
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state)
        decoder_pred = tf.argmax(decoder_pred, axis=-1)
        pred_word = data_preparation.french_idx2word[decoder_pred.numpy()[0][0]]
        pred_sent_fr.append(pred_word)
        if pred_word == "EOS" or len(pred_sent_fr) >= data_preparation.french_maxlen:
            break
        decoder_in = decoder_pred
    print("Predicted Sentence: {}".format(" ".join(pred_sent_fr)))


# Define the prediction function
def predict_with_bahdanau_attention(encoder, decoder):
    attention_layer = BahdanauAttention(units=256)

    random_id = np.random.choice(len(data_preparation.input_english_sentences))
    print("Input Sentence: {}".format(" ".join(data_preparation.input_english_sentences[random_id])))
    print("Target Sentence: {}".format(" ".join(data_preparation.target_french_sentences[random_id])))
    encoder_in = tf.expand_dims(data_preparation.input_data_english[random_id], axis=0)

    hidden = [tf.zeros((1, 1024))]
    enc_out, enc_hidden = encoder(encoder_in, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([data_preparation.french_word2idx['BOS']], 0)

    result = []

    for t in range(data_preparation.french_maxlen):
        context_vector, _ = attention_layer(dec_hidden, enc_out)

        dec_input = tf.concat([tf.cast(tf.expand_dims(context_vector, 1), tf.int32), dec_input], axis=-1)

        predictions, dec_hidden = decoder(dec_input, enc_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()
        if len(predicted_id) == 1:
            predicted_id = predicted_id.item()
            # The predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)
            if predicted_id in data_preparation.french_idx2word:
                result.append(data_preparation.french_idx2word[predicted_id])
            else:
                # Handle out-of-vocabulary (OOV) words by using 'UNK' or other suitable token
                result.append('UNK')
            print(f'Predicted id:  {predicted_id}')

        else:
            # Handle the case where `predicted_id` has more than one element, if needed.
            # You can choose to take the first element, for example.
            predicted_id = predicted_id[0].item()
            if predicted_id in data_preparation.french_idx2word:
                result.append(data_preparation.french_idx2word[predicted_id])
            else:
                # Handle out-of-vocabulary (OOV) words by using 'UNK' or other suitable token
                result.append('UNK')

            # result.append(data_preparation.french_idx2word[predicted_id])
            print(predicted_id)
            # The predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)


# -----------------------------------------------------------------------------
# This function computes the BiLingual Evaluation Understudy (BLEU) score between
# the label and the prediction across all the records in the test set. BLUE
# scores are generally used where multiple ground truth labels exist (in this
# there exists only one), but compares up to 4-grams in both reference and
# candidate sentences.
def evaluate_bleu_score(encoder, decoder, type):
    bleu_scores = []
    smooth_fn = SmoothingFunction()
    for encoder_in, decoder_in, decoder_out in data_preparation.test_dataset:
        # Initialize the encoder state accroding to the selected batchsize.
        encoder_state = encoder.init_state(data_preparation.batch_size)
        # Get the encoder final state and output for the given the encoder input
        # according to the initialized encoder state.
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        # Set the initial state of the decoder to be equal to the final state
        # of the encoder.
        decoder_state = encoder_state
        # Get the decoder final state and output for the given decoder input
        # according to the initialized decoder state.
        if (type == 'lstm'):
            decoder_pred, decoder_state = decoder(decoder_in, decoder_state)

        else:
            decoder_pred, decoder_state = decoder(decoder_in, decoder_state, encoder_out)
        # Connver the expected decoder output to a nunpy array.
        decoder_out = decoder_out.numpy()
        # Get the maximum index for each element of the decoder_pred tensor.
        # decoder_pred is initialy of shape:
        # [batch_size x french_maxlen x french_vocabulary_size] and will be
        # converted to a tensor of shape [batch_size x french_maxlen]
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()
        # Loop through the various patterns in the current decoder_out batch.
        for i in range(decoder_out.shape[0]):
            # Compose the correct sequence of target french words as a list of
            # strings.
            target_sent = [data_preparation.french_idx2word[j]
                           for j in decoder_out[i].tolist() if j > 0]
            # Compose the estimated sequence of target french words as a list of
            # strings.
            pred_sent = [data_preparation.french_idx2word[j] for j in
                         decoder_pred[i].tolist() if j > 0]
            # Remove trailing EOS tokens from both target and predicted
            # sentences. Mind that for predicted sentences during the earlier
            # training stages an EOS token may have not been generated.
            target_sent = target_sent[0:-1]
            pred_sent = pred_sent[0:-1]
            bleu_score = sentence_bleu([target_sent], pred_sent,
                                       smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)
        return np.mean(np.array(bleu_scores))


# -----------------------------------------------------------------------------
# This function cleans the older checkpoint files from the corresponding
# directory.
def clean_checkpoints(CHECKPOINT_DIRECTORY):
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    # Retrieve the prefix of the latest checkpoint file.
    if last_checkpoint is not None:
        prefix = last_checkpoint.split("/")[-1]
    # Get the list of files contained within the checkpoint directory.
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIRECTORY)]
    # Remove all file that do not contain the prefix of the latest checkpoint
    # file.
    for file in checkpoint_files:
        status = file.find(prefix)
        if status == -1:
            if file != 'checkpoint':
                remove_file = os.path.join(CHECKPOINT_DIRECTORY, file)
                os.remove(remove_file)


# -----------------------------------------------------------------------------
# This function provides the actual training process for the neural model.
def train_model(num_epochs, delta_epochs, encoder, decoder, type, CHECKPOINT_DIRECTORY, checkpoint_prefix):
    # Initialize the list of evaluation scores for the current training session.
    eval_scores = []
    # Initialize the list of losses for the current training session.
    losses = []
    # Initialize the starting training epoch.
    start_epoch = 1
    # Retrieve the starting training epoch from the last checkpoint file.
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    if last_checkpoint:
        clean_checkpoints(CHECKPOINT_DIRECTORY)
        start_epoch = int(last_checkpoint.split("-")[-1])
        checkpoint.restore(last_checkpoint)
    # Set the ending training epoch.
    finish_epoch = min(num_epochs, start_epoch + delta_epochs)
    for epoch in range(start_epoch, finish_epoch + 1):
        encoder_state = encoder.init_state(data_preparation.batch_size)
        # Loop through the contents of the training dataset.
        for encoder_in, decoder_in, decoder_out in data_preparation.train_dataset:
            loss = train_step(encoder_in, decoder_in, decoder_out, encoder_state)
            print("Training Epoch: {0} Current Loss: {1}".format(epoch, loss[0].numpy()))
            # Get the average evaluation score for the testing dataset.
            eval_score = evaluate_bleu_score(encoder, decoder, type)
            print("Eval Score (BLEU): {}".format(eval_score))
            eval_scores.append(eval_score)
            losses.append(loss[0].numpy())
        # Call the prediction function.
        if type == 'badhanau':
            predict_with_bahdanau_attention(encoder, decoder)
            checkpoint.save(file_prefix=checkpoint_prefix)
            clean_checkpoints(CHECKPOINT_DIRECTORY)
        else:
            predict(encoder, decoder, type)
            checkpoint.save(file_prefix=checkpoint_prefix)
            clean_checkpoints(CHECKPOINT_DIRECTORY)
    return eval_scores, losses


# -----------------------------------------------------------------------------
# Train and Evaluate GRU-based Seq2Seq Model
# -----------------------------------------------------------------------------
def train_evaluate_gru(num_epochs, delta_epochs, encoder, decoder, CHECKPOINT_DIRECTORY, CHECKPOINT_PREFIX):
    print("Starting to train Badhanau GRU")
    # Initialize the list of evaluation scores for the current training session.
    gru_eval_scores = []
    # Initialize the list of losses for the current training session.
    gru_losses = []
    # Initialize the starting training epoch.
    start_epoch = 1
    # Retrieve the starting training epoch from the last checkpoint file.
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    if last_checkpoint:
        clean_checkpoints(CHECKPOINT_DIRECTORY)
        start_epoch = int(last_checkpoint.split("-")[-1])
        checkpoint.restore(last_checkpoint)
    # Set the ending training epoch.
    finish_epoch = min(num_epochs, start_epoch + delta_epochs)
    for epoch in range(start_epoch, finish_epoch + 1):
        encoder_state = encoder.init_state(data_preparation.batch_size)
        # Loop through the contents of the training dataset.
        for encoder_in, decoder_in, decoder_out in data_preparation.train_dataset:
            loss = train_step(encoder_in, decoder_in, decoder_out, encoder_state)
            print("Training Epoch: {0} Current Loss: {1}".format(epoch, loss[0].numpy()))
            # Get the average evaluation score for the testing dataset.
            eval_score = evaluate_bleu_score(encoder, decoder)
            print("Eval Score (BLEU): {}".format(eval_score))
            gru_eval_scores.append(eval_score)
            gru_losses.append(loss[0].numpy())
        # Call the prediction function.
        predict_with_bahdanau_attention(encoder, decoder)
        checkpoint.save(file_prefix=CHECKPOINT_PREFIX)
        clean_checkpoints(CHECKPOINT_DIRECTORY)
    return gru_eval_scores, gru_losses


# -----------------------------------------------------------------------------
#                       MAIN PROGRAM
# -----------------------------------------------------------------------------
# Set the path to the data folder.
DATAPATH = "datasets"
# Set the name of the data file.
DATAFILE = "fra.txt"
# Set the number of English-French sentence pairs to be retrieved.
SENTENCE_PAIRS = 15000
# Set the bath size for training.
BATCH_SIZE = 64
# Set the portion of the available data to be used for testing.
TESTING_FACTOR = 10
# Set the checkpoints' directory.
CHECKPOINT_DIRECTORY = "checkpoints"
# Set the checkpoint's directory for LSTM
CHECKPOINT_DIRECTORY_LSTM = "checkpoints_lstm"
# Set the checkpoint's directory for Bad
CHECKPOINT_DIRECTORY_BAD = "checkpoints_bad"
# Set the checkpoint's directory for Bad
CHECKPOINT_DIRECTORY_LUONG = "checkpoints_luong"
# Set the checkpoint's directory for Bad
CHECKPOINT_DIRECTORY_TRANS = "checkpoints_trans"

# Set the total number of training epochs.
EPOCHS_NUMBER = 250
# Set the number of training epochs to be conducted during the current session.
DELTA_EPOCHS = 30
# Instantiate the DataPreparation class.
data_preparation = DataPreparation(DATAPATH, DATAFILE, SENTENCE_PAIRS, BATCH_SIZE,
                                   TESTING_FACTOR)

# Set the embedding dimension for the encoder and the decoder.
EMBEDDING_DIM = 256
# Set the encoder and decoder RNNs hidden dimensions.
ENCODER_DIM, DECODER_DIM = 1024, 1024
# Instantiate the encoder class.
encoder = Encoder(data_preparation.english_vocabulary_size + 1, EMBEDDING_DIM,
                  data_preparation.english_maxlen, ENCODER_DIM)
# Instantiate the decoder class.
decoder = Decoder(data_preparation.french_vocabulary_size + 1, EMBEDDING_DIM,
                  data_preparation.french_maxlen, DECODER_DIM)
# Note that vocabulary sizes for both languages was extended by one in order to
# take into account the fact that a PAD character was added during the call to
# the pad_sequences() method.
# report_encoder_decoder()
# Set the optimizer to be used during the training process.
optimizer = tf.keras.optimizers.legacy.Adam()
# Set the checkpoint directory prefix.
checkpoint_prefix = os.path.join(CHECKPOINT_DIRECTORY, "checkpoint/lstm")
# Setup a checkpoint so that the neural model can be saved every 10 epochs.
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder,
                                 decoder=decoder)

# -----------------------------------------------------------------------------
# Instantiate LSTM and train
# -----------------------------------------------------------------------------
# Instantiate the encoder class.
encoderLSTM = EncoderLSTM(data_preparation.english_vocabulary_size + 1, EMBEDDING_DIM,
                          data_preparation.english_maxlen, ENCODER_DIM)
# Instantiate the decoder class.
decoderLSTM = DecoderLSTM(data_preparation.french_vocabulary_size + 1, EMBEDDING_DIM,
                          data_preparation.french_maxlen, DECODER_DIM)
# Note that vocabulary sizes for both languages was extended by one in order to
# take into account the fact that a PAD character was added during the call to
# the pad_sequences() method.
report_encoder_decoder()

# Set the optimizer to be used during the training process.
optimizer = tf.keras.optimizers.Adam()
# Set the checkpoint directory prefix.
checkpoint_prefix_lstm = os.path.join(CHECKPOINT_DIRECTORY_LSTM, "ckpt_lstm")
# Set up a checkpoint so that the neural model can be saved every 10 epochs.
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoderLSTM,
                                 decoder=decoderLSTM)

# Call the training function.

# eval_scores, losses = train_model(EPOCHS_NUMBER, DELTA_EPOCHS, encoderLSTM, decoderLSTM, type='lstm',
#                                   CHECKPOINT_DIRECTORY=CHECKPOINT_DIRECTORY_LSTM,
#                                   checkpoint_prefix=checkpoint_prefix_lstm)
# ----------------------------------------------------------------------------------
# BADHANAU
checkpoint_prefix_bad = os.path.join(CHECKPOINT_DIRECTORY_BAD, "ckpt_bad")

# Instantiate the encoder and decoder with BadhanauAttention
encoder_badhanau = Encoder(data_preparation.english_vocabulary_size + 1, EMBEDDING_DIM,
                           data_preparation.english_maxlen, ENCODER_DIM)
decoder_badhanau = DecoderDotProduct(data_preparation.french_vocabulary_size + 1, EMBEDDING_DIM,
                                     data_preparation.french_maxlen, DECODER_DIM)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder_badhanau,
                                 decoder=decoder_badhanau)

# Train and evaluate the model
# gru_badhanau_eval_scores, gru_badhanau_losses = train_model(EPOCHS_NUMBER, DELTA_EPOCHS, encoder_badhanau,
#                                                             decoder_badhanau, type='badhanau',
#                                                             CHECKPOINT_DIRECTORY=CHECKPOINT_DIRECTORY_BAD,
#                                                             checkpoint_prefix=checkpoint_prefix_bad)

# -----------------------------------------------------------------
# LUONG
# Instantiate the encoder and decoder with LuongDotProductAttention
checkpoint_prefix_luong = os.path.join(CHECKPOINT_DIRECTORY_LUONG, "ckpt_luong")
encoder_luong = Encoder(data_preparation.english_vocabulary_size + 1, EMBEDDING_DIM,
                        data_preparation.english_maxlen, ENCODER_DIM)
decoder_luong = Decoder(data_preparation.french_vocabulary_size + 1, EMBEDDING_DIM,
                        data_preparation.french_maxlen, DECODER_DIM)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder_luong,
                                 decoder=decoder_luong)

# Train and evaluate the model
gru_luong_eval_scores, gru_luong_losses = train_model(EPOCHS_NUMBER, DELTA_EPOCHS, encoder_luong,
                                                      decoder_luong, type='lstm',
                                                      CHECKPOINT_DIRECTORY=CHECKPOINT_DIRECTORY_LUONG,
                                                      checkpoint_prefix=checkpoint_prefix_luong)

# -----------------------------------------------------------------------------
#                       TRANSFORMER
# -----------------------------------------------------------------------------
transformer = Transformer(
    num_layers=4,
    d_model=1024,
    num_heads=8,
    dff=512,
    input_vocab_size=data_preparation.english_vocabulary_size + 1,
    target_vocab_size=data_preparation.french_vocabulary_size + 1,
    dropout_rate=0.1)


def calculate_bleu_score_transformer(transformer, data_preparation, test_dataset):
    bleu_scores = []
    smooth_fn = SmoothingFunction()
    for batch_data in test_dataset:
        context, x, decoder_out = batch_data
        # Generate predictions from the Transformer model
        predictions = transformer((context, x), training=False)

        # Process the predictions and true target sequences
        for i in range(len(decoder_out)):
            target_sent = [data_preparation.french_idx2word[j]
                           for j in decoder_out[i].numpy() if j > 0]
            pred_sent = [data_preparation.french_idx2word[j]
                         for j in tf.argmax(predictions[i], axis=-1).numpy() if j > 0]

            # Remove trailing EOS tokens from both target and predicted sentences
            target_sent = target_sent[:-1]
            pred_sent = pred_sent[:-1]

            bleu_score = sentence_bleu([target_sent], pred_sent,
                                       smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)

    return np.mean(bleu_scores)


def train_transformer(num_epochs, delta_epochs, transformer, train_dataset, optimizer, checkpoint_path):
    print('Training Transformer Model')
    # Initialize the list of evaluation scores for the current training session.
    eval_scores = []
    # Initialize the list of losses for the current training session.
    losses = []

    # Initialize the starting training epoch.
    start_epoch = 1
    # Retrieve the starting training epoch from the last checkpoint file.
    last_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if last_checkpoint:
        start_epoch = int(last_checkpoint.split("-")[-1])
        checkpoint.restore(last_checkpoint)

    # Set the ending training epoch.
    finish_epoch = min(num_epochs, start_epoch + delta_epochs)

    for epoch in range(start_epoch, finish_epoch + 1):
        total_loss = 0.0
        print(f'Current epoch: {epoch}')

        for (batch, batch_data) in enumerate(train_dataset):
            # Access the context, x, and targets using tuple indices
            context, x, targets = batch_data

            with tf.GradientTape() as tape:
                predictions = transformer((context, x), training=True)
                loss = loss_function(targets, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            total_loss += loss
            eval_score = calculate_bleu_score_transformer(transformer, data_preparation, data_preparation.test_dataset)
            eval_scores.append(eval_score)
            print(f"Eval Score (BLEU): {eval_score:.4f}")
            #
            # if batch % 50 == 0:
            #     print(f"Training Epoch: {epoch}, Batch {batch}, Loss: {float(loss.numpy()[0]):.4f}")

        avg_loss = total_loss / (batch + 1)
        print(f"Training Epoch: {epoch}, Average Loss: {avg_loss:.4f}")

        losses.append(avg_loss)

        # Call the prediction function here if needed.

        # Save model checkpoints
        checkpoint.save(file_prefix=checkpoint_path)

    return eval_scores, losses


# Example usage
# Replace train_dataset with your dataset
train_dataset = data_preparation.train_dataset

# Define your optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# Define the Transformer model
transformer = Transformer(
    num_layers=4,
    d_model=1024,
    num_heads=8,
    dff=512,
    input_vocab_size=data_preparation.english_vocabulary_size + 1,
    target_vocab_size=data_preparation.french_vocabulary_size + 1,
    dropout_rate=0.1)

# Define a checkpoint path where model checkpoints will be saved
checkpoint_path_transformer = "checkpoints_transformer/"

# Train the Transformer model
eval_scores, losses = train_transformer(num_epochs=EPOCHS_NUMBER, delta_epochs=DELTA_EPOCHS, transformer=transformer,
                                        train_dataset=train_dataset, optimizer=optimizer,
                                        checkpoint_path=checkpoint_path_transformer)
