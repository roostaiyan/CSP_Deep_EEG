#!/usr/bin/env python3

''' 
Model for common spatial pattern (CSP) feature calculation and classification for EEG data
This source code is implemented based on below paper:
@article{YANG2018148,
title = {Automatic ocular artifacts removal in EEG using deep learning},
journal = {Biomedical Signal Processing and Control},
volume = {43},
pages = {148-158},
year = {2018},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2018.02.021},
url = {https://www.sciencedirect.com/science/article/pii/S1746809418300521},
author = {Banghua Yang and Kaiwen Duan and Chengcheng Fan and Chenxiao Hu and Jinlong Wang},
keywords = {Ocular artifacts (OAs) removal, Electroencephalogram (EEG), Deep learning network (DLN), Independent component analysis (ICA), Shallow network},
abstract = {Ocular artifacts (OAs) are one the most important form of interferences in the analysis of electroencephalogram (EEG) research. OAs removal/reduction is a key analysis before the processing of EEG signals. For classic OAs removal methods, either an additional electrooculogram (EOG) recording or multi-channel EEG is required. To address these limitations of existing methods, this paper investigates the use of deep learning network (DLN) to remove OAs in EEG signals. The proposed method consists of offline stage and online stage. In the offline stage, training samples without OAs are intercepted and used to train an DLN to reconstruct the EEG signals. The high-order statistical moments information of EEG is therefore learned. In the online stage, the trained DLN is used as a filter to automatically remove OAs from the contaminated EEG signals. Compared with the exiting methods, the proposed method has the following advantages: (i) nonuse of additional EOG reference signals, (ii) any few number of EEG channels can be analyzed, (iii) time saving, and (iv) the strong generalization ability, etc. In this paper, both public database and lab individual data for EEG analysis are used, we compared the proposed method with the classic independent component analysis (ICA), kurtosis-ICA (K-ICA), Second-order blind identification (SOBI) and a shallow network method. Experimental results show that the proposed method performs better even for very noisy EEG.}
}
'''
#
import numpy as np
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from math import sqrt, log10
import matplotlib.pyplot as plt
from scipy import signal
from mne.decoding import CSP
import DLN as DLN

from sklearn.decomposition import FastICA

# import self defined functions
import data_handling as data_handling


class DLN_CSP_SVM_Model:
    def __init__(self):
        self.data_path = 'BCICIV_1_mat/'
        self.dataset = data_handling.Dataset()  # should be assigned by calling load_data function
        # From Fig. 8 we can see that the RMSE obtains the minimum when rho = 0.4, beta = 0.00005.
        # Thus, rho = 0.4, beta = 0.00005 is taken in this paper
        self.structure = [100, 80, 100, 80, 100]  # to test Single Layer Autoencoder you should set this to [100,80]
        self.rho = 0.4
        self.beta = 0.00005
        self.n_epochs = 100
        self.lr = 0.0001

        #  paper (Section 4.2) Here, the common spatial pattern (CSP) is used
        #  paper (Section 4.2) Here, the common spatial pattern (CSP) is used
        # for feature extraction and the support vector machine (SVM) for classification.
        self.svm_kernel = 'linear'  # 'linear' 'sigmoid', 'rbf', 'poly'
        self.svm_gamma = 1
        self.svm_c = 10
        self.csp_components = 200
        self.dim_reduction = 20

        self.fs = 100.  # sampling frequency
        self.no_channels = 59  # number of EEG channels
        #  paper (Section 3.Data Description) In this paper, we selected Subject 1, 2, 6 and 7 to test the proposed method,
        #  because their EEG data are real.
        self.subjects = [2]  # [1,2,6,7] is tested in the paper
        self.subject = 1  # for each run it will be changed
        self.train_time = 0
        self.eval_time = 0

    def load_data(self):
        self.dataset = data_handling.Dataset()
        self.dataset.readDataset(self.subject, self.data_path)

    def run_dln(self):
        deep_model = DLN.DLN(self.structure, self.rho, self.beta, self.dataset)
        n_Epochs = self.n_epochs
        lr = self.lr
        deep_model.train(n_Epochs, lr)
        self.deep_model = deep_model
        # Test Trained Network for Demonising Capability in terms of RMSE measure
        # It is calculated in uV scale of the dataset, the given dataset is scaled before converting to int
        EEGs_NOE = 0.1 * self.dataset.data_test_DLN  # To convert it to uV values, use cnt= 0.1*double(cnt); http://bbci.de/competition/iv/desc_1.html
        EEGs_inp = self.dataset.normalize(EEGs_NOE)
        EEGs_otp = deep_model.predict(EEGs_inp)
        EEGs_crt = self.dataset.de_normalize(EEGs_otp, EEGs_NOE)
        rmse = calc_rmse(EEGs_crt, EEGs_NOE)
        print('RMSE of DLN for Subject ' + str(self.subject) + ' = ' + str(rmse))
        # Generate Figure. 9
        # Fig. 9. Reconstruction of test sample by DLN
        data_index = 100
        n = EEGs_otp.shape[1]
        EEG_original = EEGs_inp[data_index, :]
        EEG_recovered = EEGs_otp[data_index, :]
        fig, (ax1) = plt.subplots()
        fig.subplots_adjust(hspace=0.5)
        t1 = np.arange(1, n + 1, 1)
        ax1.plot(t1, EEG_original, c='blue', label='Test sample')
        ax1.plot(t1, EEG_recovered, c='red', label='Reconstruction of test sample')
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Sampling Point')
        ax1.grid(True)
        ax1.set_ylabel('Signal Amplitude')
        ax1.legend()
        ax1.set_title('Figure. 9')
        plt.show(block=False)

        # Fig. 11. Removal effect by using DLN.
        data_index = 0
        timeSignal = 10  # in second
        EEG_cnt = self.dataset.data_all[data_index:data_index + timeSignal, 0, :]  # size = (no_trains, no_channels, n)
        EEGs_crt = deep_model.predict_ctn(EEG_cnt)
        EEG_original = np.reshape(EEG_cnt, (EEG_cnt.shape[1] * timeSignal))
        EEG_recovered = np.reshape(EEGs_crt, (EEGs_crt.shape[1] * timeSignal))

        fig, (ax2) = plt.subplots()
        fig.subplots_adjust(hspace=0.5)
        t1 = np.arange(1, EEG_cnt.shape[1] * timeSignal + 1, 1)
        ax2.plot(t1, EEG_original, c='blue', label='Test sample')
        ax2.plot(t1, EEG_recovered, c='red', label='Reconstruction of test sample (DLN)')
        ax2.set_ylim(0, 3000)
        ax2.set_xlabel('Sampling Point')
        ax2.grid(True)
        ax2.set_ylabel('Signal Amplitude')
        ax2.legend()
        ax2.set_title('Figure. 11')
        plt.show(block=False)
        '''
        # Fig. 13. Removal effect by using ICA.
        transformer = FastICA(n_components=2)
        transformer.fit(EEG_cnt)
        EEGs_crt_ICA = transformer.inverse_transform(EEG_cnt)
        EEG_recovered_ICA = np.reshape(EEGs_crt_ICA,(EEG_cnt.shape[1]*timeSignal))
        fig, (ax3) = plt.subplots()
        fig.subplots_adjust(hspace=0.5)
        t1 = np.arange(1, n + 1, 1)
        ax3.plot(t1, EEG_original, c='blue', label='Test sample')
        ax3.plot(t1, EEG_recovered_ICA, c='red', label='Reconstruction of test sample (ICA)')
        ax3.set_ylim(0, 3000)
        ax3.set_xlabel('Sampling Point')
        ax3.grid(True)
        ax3.set_ylabel('Signal Amplitude')
        ax3.legend()
        ax3.set_title('Figure. 13')
        plt.show(block=False)
        '''

        # Figure. 16 Power Spectral Density
        freqs, psd_CNT = signal.welch(EEG_original, fs=self.fs, nperseg=100)
        freqs, psd_DLN = signal.welch(EEG_recovered, fs=self.fs, nperseg=100)
        # freqs, psd_ICA = signal.welch(EEG_recovered, fs=self.fs)
        fig, (ax4) = plt.subplots()
        fig.subplots_adjust(hspace=0.5)
        for i in range(0, psd_CNT.__len__()):
            psd_CNT[i] = 20 * log10(psd_CNT[i])
            psd_DLN[i] = 20 * log10(psd_DLN[i])
        ax4.plot(freqs, psd_CNT, c='blue', label='Contaminated EEG')
        ax4.plot(freqs, psd_DLN, c='red', label='Corrected by DLN')
        # ax4.plot(t1, psd_ICA, c='black', label='Corrected by DLN')
        ax4.set_xlabel('Frequency in Hz')
        ax4.set_ylabel('PSD')
        ax4.legend()
        ax4.set_title('Figure. 16')
        plt.show(block=False)

    def run_csp_svm(self):
        # Correct EEGs using Offline DLN
        data_train_crt = np.zeros(self.dataset.data_train.shape)
        for s in range(0, self.no_channels):
            data_train_crt[:, s, :] = (self.deep_model.predict_ctn(self.dataset.data_train[:, s, :]))
        data_test_crt = np.zeros(self.dataset.data_train.shape)
        for s in range(0, self.no_channels):
            data_test_crt[:, s, :] = (self.deep_model.predict_ctn(self.dataset.data_test[:, s, :]))

        start_train = time.time()
        # Apply CSP for Feature Extraction
        # Train CSP model for feature extraction

        self.csp = CSP(n_components=self.csp_components, reg=None, log=True, norm_trace=True)
        self.csp.fit(data_train_crt, self.dataset.class_train)
        data_train_features = self.csp.transform(data_train_crt)
        data_test_features = self.csp.transform(data_test_crt)

        data_train_features = normalize(data_train_features)
        data_test_features = normalize(data_test_features)

        dim_reduction = PCA(n_components=self.dim_reduction)
        dim_reduction.fit(data_train_features)

        data_train_features = normalize(dim_reduction.transform(data_train_features))
        data_test_features = normalize(dim_reduction.transform(data_test_features))

        # Train SVM Model
        if self.svm_kernel == 'linear':
            svm_classifier = LinearSVC(C=self.svm_c, intercept_scaling=1, max_iter=1000, multi_class='ovr'
                                       , random_state=1, tol=0.00001)
        else:
            svm_classifier = SVC(self.svm_c, self.svm_kernel, degree=3, gamma=self.svm_gamma, coef0=0.0, tol=0.001,
                                 max_iter=-1, decision_function_shape='ovr')
        svm_classifier.fit(data_train_features, self.dataset.class_train)

        end_train = time.time()
        self.train_time += end_train - start_train

        # Calculate Classification Accuracy
        start_eval = time.time()
        class_test_prediction = svm_classifier.predict(data_test_features)
        end_eval = time.time()
        self.eval_time += end_eval - start_eval
        classification_accuracy = np.mean(class_test_prediction == self.dataset.class_test) * 100

        print("Evaluation Time " + str(end_eval - start_eval))

        return classification_accuracy


def calc_rmse(eegs: np.array, dln_out: np.array) -> float:
    rmse = np.sqrt(np.mean(np.sqrt(np.mean(np.power(np.add(eegs, -dln_out), 2), 1))))  # root of mean squared error
    return rmse


def main():
    model = DLN_CSP_SVM_Model()
    # success rate sum over all subjects
    success_tot_sum = 0
    start = time.time()
    # Do This For each Subject
    for s_index in range(0, model.subjects.__len__()):
        model.subject = model.subjects[s_index]
        print("Training Procedure for Subject " + str(model.subject) + " ***************")
        # load data
        model.load_data()
        model.run_dln()
        success_rate = model.run_csp_svm()

        print('Classification Accuracy of DLN+CSP+SVM for Subject [' + str(model.subject) + ']  = ' + str(success_rate))
        success_tot_sum += success_rate
    ## Normalize Train Data for Deep Network as mentioned in the paper (Section 2.1.3)

    # Average success rate over all subjects
    print("Average success rate: " + str(success_tot_sum / model.subjects.__len__()))
    print("Average training time: " + str(model.train_time / model.subjects.__len__()))
    print("Average Evaluation time: " + str(model.eval_time / model.subjects.__len__()))

    end = time.time()

    print("Time elapsed [s] " + str(end - start))


if __name__ == '__main__':
    main()
