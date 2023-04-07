# This class file provides fundamental computational functionality for addressing
# the problem of time series classification through the utilization of a 
# Transformer Model. The main architecture of the transformer model is described
# in the "Attention Is All You Need" paper by Vaswani et al. (2017).

# The dataset was originally used in a competition in the IEEE World Congress
# on Computational Intelligence, 2008. The underlying pattern classification
# task was to diagnose whether a certain symptom exists or not within a given
# automotive subsystem. Each case consists of 500 measurements of engine noise 
# and the associated classification label.

# Import required Python modules.
import os
from scipy.io import arff
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class TimeSeriesTransformer:
    
    def __init__(self,datadirectory,train_file,test_file):
        self.datadirectory = datadirectory
        self.trainpath = os.path.join(self.datadirectory,train_file)
        self.testpath = os.path.join(self.datadirectory,test_file)
        self.load_train_test_data()
    
    def initialize_model_parameters(self,input_shape,head_size,num_heads,
                                    ff_dim,num_transformer_blocks,mlp_units,
                                    mlp_dropout,dropout,patience_epochs,
                                    training_epochs,validation_split,
                                    batch_size,trained_model):
        self.input_shape = input_shape
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout
        self.patience_epochs = patience_epochs
        self.training_epochs = training_epochs
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.trained_model = trained_model
        
    
    def load_train_test_data(self):
        # Load the training data patters.
        train_data = arff.loadarff(self.trainpath)
        self.train_df = pd.DataFrame(train_data[0])
        # Ensure that the training data targets are integers.
        self.train_df.target = self.train_df.target.astype(int)
        # Convert the train dataframe to object to numpy array.
        self.traip_np = self.train_df.to_numpy()
        # Get the training features as the numpy array x_train.
        self.x_train = self.traip_np[:,:-1]
        # Reshape the contents of the x_train array so that each training 
        # pattern is a sequence of observations where each observation is a
        # distinct array element.
        self.x_train = self.x_train.reshape((self.x_train.shape[0],
                                            self.x_train.shape[1],
                                            1))
        # Get the training labels as the numpy array y_train.
        self.y_train = self.traip_np[:,-1]
        
        # Shuffle the contents of x_train and y_train accordingly.
        train_idx = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[train_idx]
        self.y_train = self.y_train[train_idx]
        # Replace -1 labels with zeros.
        self.y_train[self.y_train==-1] = 0
        
        # Get the number of classes.
        self.classes_number = len(np.unique(self.y_train))
        
        # Load the testing data patterns.
        test_data = arff.loadarff(self.testpath)
        self.test_df = pd.DataFrame(test_data[0])
        # Ensure that the testing data targets are integers.
        self.test_df.target = self.test_df.target.astype(int)
        # Convert the test dataframe object to numpy array.
        self.test_np = self.test_df.to_numpy()
        # Get the testing features as the numpy array x_test.
        self.x_test = self.test_np[:,:-1]
        # Reshape the contents of the x_test array so that each testing 
        # pattern is a sequence of observations where eacj observation is a 
        # distinct array element.
        self.x_test = self.x_test.reshape((self.x_test.shape[0],
                                           self.x_test.shape[1],
                                           1))
        # Get the testing labels as the numpy array y_test.
        self.y_test = self.test_np[:,-1]
        
        # Shuffle the contents of x_test and y_test accordingly.
        test_idx = np.random.permutation(len(self.x_test))
        self.x_test = self.x_test[test_idx]
        self.y_test = self.y_test[test_idx]
        # Replace the -1 labels with zeros.
        self.y_test[self.y_test==-1] = 0
    
    # This function implements the main architectural block of the utilized 
    # neural model. Multiple instances of these transformer_encoder blocks
    # can be stack together before adding the final multi-layer perceptron
    # classfication head.
    def transformer_encoder(self,inputs):
        # This function defines the fundamental building block of the utilized
        # neural model. The ongoing neural computation processes  a tensor of 
        # shape (batch_size, sequence_length, features) where sequence_length
        # is the number of time steps and features is its input timeseries.
        
        # Set the Normalization and Attention Layers. Residual connections and
        # dropout are included.
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=self.head_size,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout)(x,x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs
        
        # Set the Feed Forward Part. The projection layers wil be implemented
        # through a one-dimensional convolution layer.
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim,kernel_size=1,activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1],kernel_size=1)(x)
        return x + res
        
    # This function stacks multiple instances of the transformer_encoder block
    # before proceeding to add the final multi-layer perceptron classification
    # head. It is important to note that appart from stacking a sequence of
    # dense layers, it is required to reduce the dimensionality of the output
    # tensor which is produced by the transformer-encoder part of the neural 
    # model. This tensor should be reduced down to a vector of features for
    # each data point in the current batch. Such an operation can be achieved
    # through the utilization of a pooling layer.
    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim,activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        outputs = layers.Dense(self.classes_number,activation='softmax')(x)
        self.model = keras.Model(inputs,outputs)
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer = keras.optimizers.Adam(learning_rate=1e-4),
                           metrics = ["sparse_categorical_crossentropy"])
        self.model.summary()
        
    # This function performs the actual training of the model by incorporating
    # an early stopping criterion that defines a patience level of training
    # epochs before terminating the training process.
    def train_model(self):
        # Set the early stopping callback.
        callbacks = [keras.callbacks.EarlyStopping(patience=self.patience_epochs,
                                                   restore_best_weights=True)]
        # Check the existence of the MODEL_NAME directory. In case of existence, then the
        # neural network model is assumed to be already trained.
        model_directory_status = os.path.isdir(TRAINED_MODEL)
        if not model_directory_status:
            # Train the model.
            self.model.fit(self.x_train,self.y_train,validation_split=self.validation_split,
                           epochs=self.training_epochs,
                           batch_size=self.batch_size,callbacks=callbacks)
            # Save the neural network model.
            self.model.save(TRAINED_MODEL)
            
    # This function performs the actual testing of the neural model.
    def test_model(self):
        self.model.evaluate(self.x_test,self.y_test,verbose=1)
# =============================================================================
#                                          MAIN  PROGRAM
# =============================================================================
if __name__ == "__main__":
    
    # Set the data directory.
    datadirectory = "datasets"
    # Set the name of the training data file.
    train_file = "FordA_TRAIN.arff"
    # Set the name of the testing data file.
    test_file = "FordA_TEST.arff"
    # Instantiate the time series transformer class.
    time_series_transformer = TimeSeriesTransformer(datadirectory,train_file,test_file)
    
    # Set the internal model parameters.
    INPUT_SHAPE = time_series_transformer.x_train.shape[1:]
    HEAD_SIZE = 256
    NUM_HEADS = 4
    FF_DIM = 4
    NUM_TRANSFORMER_BLOCKS = 4
    MLP_UNITS = [128]
    MLP_DROPOUT = 0.4
    DROPOUT = 0.25
    PATIENCE_EPOCHS = 10
    TRAINING_EPOCHS = 200
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 32
    TRAINED_MODEL = "transformer_model"
    
    # Intialize internal model parameters.
    time_series_transformer.initialize_model_parameters(INPUT_SHAPE,HEAD_SIZE,
                                                        NUM_HEADS,FF_DIM,
                                                        NUM_TRANSFORMER_BLOCKS, 
                                                        MLP_UNITS,MLP_DROPOUT,
                                                        DROPOUT,PATIENCE_EPOCHS,
                                                        TRAINING_EPOCHS,
                                                        VALIDATION_SPLIT,
                                                        BATCH_SIZE,
                                                        TRAINED_MODEL)
    
    # Build Transformer Model.
    time_series_transformer.build_model()
    
    # Train the model.
    time_series_transformer.train_model()
    
    # Test the model.
    time_series_transformer.test_model()