#!/usr/bin/env python3

'''	
Denoising of EEG Signals based on Deep Stacked Autoencoders
'''

import numpy as np
from keras.layers import Input, Dense
from keras import optimizers
from keras.models import Model
from keras import regularizers  # for sparsity constraint (Stacked Sparse Auto-Encoder
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
from math import sqrt

import data_handling as data_handling


# pip install xmltodict

class DLN_Denoising:
    '''     data_train_DLN   numpy matrix 	size = no_train_DLN x n        NOEs 
            data_test_DLN   numpy matrix 	size = no_train_DLN x n
            SN_min           float 
            SN_max           float 
            SN               refrence NOE Signal
            rho              sparsity parameter
            beta             spasity regularization
            '''

    def __init__(self, structure: np.array, rho: float, beta: float,
                 dataset: data_handling.Dataset, trainInput: np.array, trainOutput:np.array):  # Take DLN with a structure of 100(should be exactly equal to n)-80-100-80-100 for example,
        self.structure = structure
        self.rho = rho
        self.beta = beta
        self.dataset = dataset
        self.learning_rate = 0.001
        self.n_epochs = 500
        self.trainInput = trainInput
        self.trainOutput = trainOutput

    def build_network(self):
        # extract parameters
        num_Layers = self.structure.__len__()
        # Build Deep Stacked Sparse Autoencoders using Keras Package
        n = self.structure[0]
        # build Sparse Autoencoders for Layer-wise Pretraining
        AEs = []
        Encoders = []
        for layer in range(0, num_Layers - 1):
            input_dim = self.structure[layer]
            encoding_dim = self.structure[layer + 1]
            input_layer = Input(shape=(input_dim,))

            if (layer == 0):
                input_eeg = input_layer

            encoder_layer = Dense(encoding_dim, activity_regularizer=regularizers.l1(self.beta))(input_layer)
            decoder_layer = Dense(input_dim)(encoder_layer)
            if (layer == 0):
                decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)
            # optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)
            AE = Model(input_layer, decoder_layer)
            AE.compile(optimizer=optimizer, loss='mean_squared_error')
            AEs.insert(layer, AE)
            # optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)
            Encoder = Model(input_layer, encoder_layer)
            Encoder.compile(optimizer=optimizer, loss='mean_squared_error')
            Encoders.insert(layer, Encoder)

        # build Unfolded Deep Sparse Autoencoder for Fine-tune Stage
        # Stack encoders
        for layer in range(0, num_Layers - 1):
            input_dim = self.structure[layer]
            encoding_dim = self.structure[layer + 1]
            # stack desire encoders
            if (layer == 0):
                stacked_AEs = Dense(encoding_dim, activity_regularizer=regularizers.l1(self.beta))(input_eeg)
            else:
                stacked_AEs = Dense(encoding_dim, activity_regularizer=regularizers.l1(self.beta))(stacked_AEs)
        # Stack decoders on the final encoder
        for layer in range(num_Layers - 1, 0, -1):
            decoding_dim = self.structure[layer - 1]
            if (layer == 1):
                stacked_AEs = Dense(decoding_dim, activation='sigmoid')(stacked_AEs)
            else:
                stacked_AEs = Dense(decoding_dim)(stacked_AEs)

        stacked_AEs = Model(input_eeg, stacked_AEs)
        # Mean Squared Error Loss function
        # optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)
        stacked_AEs.compile(optimizer=optimizer, loss='mean_squared_error')

        self.Encoders = Encoders
        self.AEs = AEs
        self.stacked_AEs = stacked_AEs

    def train(self, n_epochs: int, learning_rate: float):

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.build_network()
        # Testing for parameters to be valid
        n = self.dataset.data_train_DLN.shape[1]  # n = 100 structure[0] === n
        if (n != self.structure[0]):
            print("Error: the dimension of input and Input layer must be same")
            self.structure[0] = n
            print("Structure[0] has been automatically changed to ensure the consistency ... ")

        # You can Plot DLN Structure by this line, it would need graphviz installation
        # SVG(model_to_dot(self.Deep_SAE).create(prog='dot', format='svg'))

        # Data Normalization
        # (Section 2.3.1) Each training sample should be normalized before training
        data_train_DLN_input = self.dataset.normalize(self.trainInput)
        data_train_DLN_output = self.dataset.normalize(self.trainOutput)
        # Table 2. DLN Training Algorithm which is implemented using keras
        # Stage 1: Layer-wise pretraining of Sparse Autoencoders
        print(
            "======== LAYER_WISE Pre-training Stage ==============================================================================")
        train_layer = data_train_DLN_input
        out_layer = data_train_DLN_output
        for layer in range(0, self.structure.__len__() - 1):
            print("======== Pre-training of AE[" + str(layer + 1) + "]")
            self.AEs[layer].fit(train_layer, out_layer,
                                epochs=self.n_epochs,
                                batch_size=32,
                                shuffle=True,
                                validation_split=0.2)  # this will be used to validate model during training
            self.Encoders[layer].layers[1].set_weights(self.AEs[layer].layers[1].get_weights())
            train_layer = self.Encoders[layer].predict(train_layer)
            out_layer = train_layer
            # set the result for final DLN
            self.stacked_AEs.layers[layer + 1].set_weights(self.AEs[layer].layers[1].get_weights())
            self.stacked_AEs.layers[-1 - layer].set_weights(self.AEs[layer].layers[2].get_weights())
        # Stage 2: Fine-tune of Overall Deep Network
        print(
            "======== Fine-tune Stage              ==============================================================================")
        history = self.stacked_AEs.fit(data_train_DLN_input, data_train_DLN_output,
                                       epochs=self.n_epochs,
                                       batch_size=32,
                                       shuffle=True,
                                       validation_split=0.2)  # this will be used to validate model during training

        # Visualize Training Status loss values
        fig, (ax1) = plt.subplots()
        ax1.plot(history.history['loss'])
        ax1.set_title('Training Loss for each epoch')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend('Training Loss')
        plt.show(block=False)

    def predict_ctn(self, EEGs_cnt: np.array) -> np.array:
        # Take data_test_DLN as EEGs_ctn
        EEGs_std = self.dataset.standardize(EEGs_cnt)
        EEGs_otp = self.stacked_AEs.predict(EEGs_std)
        EEGs_crt = self.dataset.de_standardize(EEGs_otp, EEGs_cnt)
        return EEGs_crt

    def predict(self, EEGs_inp: np.array) -> np.array:
        EEGs_otp = self.stacked_AEs.predict(EEGs_inp)
        return EEGs_otp
