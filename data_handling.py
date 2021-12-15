#!/usr/bin/env python3

'''	
Loads the dataset 1 of the BCI Competition IV
available on http://www.bbci.de/competition/iv/#datasets
'''

import numpy as np
import scipy.io as sio
import scipy.stats as stats
import math as math
import matplotlib.pyplot as plt

class  Dataset:
    def __init__(self):
        # data_train_DLN: np.array, data_test_DLN: np.array, data_train: np.array, class_train: np.array, data_test: np.array,
        # class_test: np.array, SN: np.array, SN_min: float, SN_max: float
        self.data_train_DLN = np.array([1,1])
        self.data_test_DLN = np.array([1,1])
        self.data_train = np.array([1,1,1])
        self.class_train = np.array([1])
        self.data_test = np.array([1,1,1])
        self.class_test = np.array([1])
        self.SN = np.array([1])
        self.SN_min = 0.0
        self.SN_max = 0.0

    def readDataset(self,subject, PATH):
        '''	Loads the dataset 1 of the BCI Competition IV
        available on http://www.bbci.de/competition/iv/#dataset1
        
        Arguments:
        subject -- number of subject which is 1, 2, 3, 4, 5, 6 or 7
        In the paper, Subject 1, 2, 6 and 7 is used to test the proposed method, because their EEG data are real
        
        Return:	namedtupe. Dataset.(data_train_DLN, data_test_DLN, data_train, class_train, data_test, class_test, SN_min, SN_max, SN)
        
                data_train_DLN   numpy matrix 	size = no_train_DLN x n
                data_test_DLN   numpy matrix 	size = no_train_DLN x n
                data_train 	     numpy matrix 	size = (no_trains, no_channels, n)
                data_test	     numpy matrix 	size = (no_tests, no_channels, n)
                class_train 	 numpy matrix 	size =  no_trains
                class_test	     numpy matrix 	size =  no_trains
                SN_min           float 
                SN_max           float 
                SN               refrence NOE Signal
                
        
        page 153 Each subject has 200 trails (cues) of motor imagery and each trail lasts for more than 6 s.
        In this paper, n equals 100 input units are inputted into DLN
        For each subject, the EEG data is divided into two parts (Part 1 and
        Part 2), while each part consists of 100 trails. Part 1 is used to train
        and test the DLN, and Part 2 is used to remove OAs and test the proposed method
        '''
    # kv (Section 2.1.2) (Determine NOEs) If the EEG segment kurtosis is over threshold
    # kv, this segment is eliminated which is doom to contain ocular artifact.
    # (allSamples/100)*59 = X   (Filtering)=> 16520 = 280*59 (training NOEs) + 15458 = 262*59 (testing NOEs)
        kv = -0.1 # undefined in the paper
        # num of training cues in each subject
        no_cues = 200
        no_train_cues = 100
        no_test_cues = no_cues - no_train_cues # 100
        # page 153 -(EEG signals were recorded from 59 channels)
        no_channels = 59
        # 1s of 100Hz = 100 => n = 100
        segment_len_sc = 1.0
        num_segments = 5
        freq = 100
        n = int(segment_len_sc*freq) # 100
        # calculate num of train and test
        no_trains = no_train_cues*num_segments
        no_tests = no_test_cues*num_segments
        self.class_train = np.zeros(no_trains)
        self.class_test = np.zeros(no_tests)
        self.data_train = np.zeros((no_trains, no_channels, n))
        self.data_test = np.zeros((no_tests, no_channels, n))
        data_train_DLN = []
        data_test_DLN = []
        data_cnt = []
        # Load Dataset
        data = sio.loadmat(PATH + 'BCICIV_calib_ds1' + chr(ord('a') - 1 + subject) + '.mat')
        # test = sio.loadmat(PATH + 'BCICIV_eval_ds1' + chr(ord('a') - 1 + subject) + '.mat')
        # digitized at 100 Hz with 16bit (0.1 mu V) accuracy
        samples = (data['cnt']).astype(float) # samples x no_channels
        no_samples = samples.shape[0]
        # dataset description: (pos: vector of positions of the cue in the EEG signals given in uint sample, length cues)
        cue_Positions = data['mrk']['pos'][0][0][0]
        # label of each cue which will used for classification
        label_Cues = data['mrk']['y'][0][0][0]
        no_samples_Part1 = cue_Positions[no_train_cues-1]
        data.clear()
        # no_cues is 200
        num_Cues = label_Cues.size
        win_size = 200 # 2 sec
        num_wins = math.floor(no_samples_Part1/win_size)
        # read each windows of Part 1 and determine NOEs based on kurtosis
        for ii in range(0, num_wins):
            startIndex = ii*win_size
            endIndex = (ii+1)*win_size
            win = samples[startIndex:endIndex][:]
            winSamples = np.reshape(win,[win_size*no_channels])
            ks = stats.kurtosis(winSamples,fisher=True,bias=False)
            if(ks<kv):
                if(ii<(num_wins/2)): # take for train
                    data_train_DLN.extend(np.transpose(win).reshape(no_channels*int(win_size/n),n))
                else:                # take for test
                    data_test_DLN.extend(np.transpose(win).reshape(no_channels*int(win_size/n),n))
            else:
                if(data_cnt.__len__()<50): # keep first 50 contaminated EEGs to visualize purpose (Figure 6 and 7 of paper)
                    data_cnt.extend(np.transpose(win))
        self.data_train_DLN = np.array(data_train_DLN)
        self.data_test_DLN = np.array(data_test_DLN)
        data_cnt = np.array(data_cnt)
        # Refrence NOE
        self.SN = self.data_train_DLN[0,:]
        self.SN_min = np.min(self.SN) #np.min(self.data_train_DLN))
        self.SN_max = np.max(self.SN) #np.max(self.data_train_DLN))
        SN_nrm = self.normalize(self.SN)
        # plot Normalized Refrence NOE
        EEG_ctn = data_cnt[0]
        EEG_std = self.standardize(EEG_ctn)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.subplots_adjust(hspace=0.5)
        t1 = np.arange(1,n+1,1)
        t2 = np.arange(1,2*n+1,1)
        ax1.plot(t1,SN_nrm)
        ax2.plot(t2,EEG_std)
        ax1.set_ylim(0, 7)
        ax2.set_ylim(0, 7)
        ax1.set_xlabel('Sampling Point')
        ax1.grid(True)
        ax1.set_ylabel('Signal Amplitude')
        ax2.set_xlabel('Sampling Point')
        ax2.grid(True)
        ax2.set_ylabel('STD Amplitude')
        ax1.set_title('Figure. 6')
        ax2.set_title('Figure. 7')
        plt.show(block=False)
        # Obtain Train and Test data for classification purpose
        # Intercept first 5 s of EEG data for each trial, andmthen evenly divide these 5-s EEG data into 5 segments. Therefore,
        # we can obtain 500 labels in total and each label corresponds to 1 s of EEG data
        self.data_all = np.zeros((num_Cues*5,no_channels,n))
        class_all = np.zeros(num_Cues*5)
        for ii in range(0, num_Cues):
            startIndex = cue_Positions[ii]
            endIndex = startIndex + 5 * n # 5 s of each trial
            cue_samples = samples[startIndex:endIndex][:]
            for s in range(0,5):
                    seg_index = ii*5 + s
                    segment = np.transpose(cue_samples[s*n:(s+1)*n][:]) # no_channels X 100
                    self.data_all[seg_index][:][:] = segment
                    class_all[seg_index] = int((label_Cues[ii] + 3)/2) # convert -1 and +1 to 1 and 2
        # split train and test
        np.random.seed(0)
        shuffle = np.random.permutation(no_trains+no_tests)
        self.data_train[:,:,:] = self.data_all[shuffle[0:no_trains],:,:]
        self.class_train[:] = class_all[shuffle[0:no_trains]]
        self.data_test[:, :, :] = self.data_all[shuffle[no_trains:no_trains+no_tests], :, :]
        self.class_test[:] = class_all[shuffle[no_trains:no_trains+no_tests]]

        print('Dataset has been read Successfully ...')



    def normalize(self, EEGs: np.array) -> np.array:
        if(EEGs.shape.__len__()==1):
            EEGs.shape = (1,EEGs.shape[0])
        num_segments = EEGs.shape[0]
        EEGs_nrm = np.zeros(EEGs.shape)
        for ii in range(0,num_segments):
            EEG = EEGs[ii,:]
            minEEG = np.min(EEG)
            maxEEG = np.max(EEG)
            EEGs_nrm[ii,:] = (EEG - minEEG) / (maxEEG - minEEG)
        if (num_segments == 1):
            EEGs_nrm.shape = (EEGs_nrm.shape[1])
        return EEGs_nrm

    def de_normalize(self, EEGs_nrm: np.array, EEGs: np.array) -> np.array:
        if(EEGs.shape.__len__()==1):
            EEGs.shape = (1,EEGs.shape[0])
        num_segments = EEGs.shape[0]
        EEGs_dnrm = np.zeros(EEGs.shape)
        for ii in range(0,num_segments):
            EEG = EEGs[ii,:]
            minEEG = np.min(EEG)
            maxEEG = np.max(EEG)
            EEGs_dnrm[ii,:] = EEGs_nrm[ii,:] * (maxEEG - minEEG) + minEEG
        if (num_segments == 1):
            EEGs_dnrm.shape = (EEGs_dnrm.shape[1])
        return EEGs_dnrm

    def power_normalize(self, EEGs: np.array) -> np.array:
        if(EEGs.shape.__len__()==1):
            EEGs.shape = (1,EEGs.shape[0])
        num_segments = EEGs.shape[0]
        EEGs_nrm = np.zeros(EEGs.shape)
        for ii in range(0,num_segments):
            EEG = EEGs[ii,:]
            power_EEG = sum(np.power(EEG,2))
            EEGs_nrm[ii,:] = (EEG) / power_EEG
        if (num_segments == 1):
            EEGs_nrm.shape = (EEGs_nrm.shape[1])
        return EEGs_nrm

    def power_normalize_ctn(self, EEGs: np.array,CleanEEGs) -> np.array:
        if(EEGs.shape.__len__()==1):
            EEGs.shape = (1,EEGs.shape[0])
        num_segments = EEGs.shape[0]
        EEGs_nrm = np.zeros(EEGs.shape)
        for ii in range(0,num_segments):
            EEG = EEGs[ii,:]
            power_EEG = sum(np.power(CleanEEGs[ii,:],2))
            EEGs_nrm[ii,:] = (EEG) / power_EEG
        if (num_segments == 1):
            EEGs_nrm.shape = (EEGs_nrm.shape[1])
        return EEGs_nrm

    def de_power_normalize(self, EEGs_nrm: np.array, EEGs: np.array) -> np.array:
        if(EEGs.shape.__len__()==1):
            EEGs.shape = (1,EEGs.shape[0])
        num_segments = EEGs.shape[0]
        EEGs_dnrm = np.zeros(EEGs.shape)
        for ii in range(0,num_segments):
            EEG = EEGs[ii,:]
            power_EEG = sum(np.power(EEG, 2))
            EEGs_dnrm[ii, :] = EEGs_nrm[ii,:] * power_EEG
        if (num_segments == 1):
            EEGs_dnrm.shape = (EEGs_dnrm.shape[1])
        return EEGs_dnrm

    def standardize(self, EEGs_ctn: np.array) -> np.array:
        if(EEGs_ctn.shape.__len__()==1):
            EEGs_ctn.shape = (1,EEGs_ctn.shape[0])
        num_segments = EEGs_ctn.shape[0]
        EEGs_std = np.zeros(EEGs_ctn.shape)
        for ii in range(0, num_segments):
            EEG_std = (EEGs_ctn[ii,:] - self.SN_min) / (self.SN_max - self.SN_min)
            minRes = np.min(EEG_std)
            EEG_std = EEG_std - minRes
            EEGs_std[ii,:] = EEG_std
        if (num_segments == 1):
            EEGs_std.shape = (EEGs_ctn.shape[1])
        return EEGs_std


    def de_standardize(self, EEGs_otp: np.array, EEGs_ctn:np.array) -> np.array:
        if(EEGs_otp.shape.__len__()==1):
            EEGs_otp.shape = (1,EEGs_otp.shape[0])
        num_segments = EEGs_otp.shape[0]
        EEGs_crt = np.zeros(EEGs_otp.shape)
        for ii in range(0, num_segments):
            S_EEG = (EEGs_ctn[ii,:] - self.SN_min) / (self.SN_max - self.SN_min)
            minRes = np.min(S_EEG)
            EEGs_crt[ii,:] = np.multiply((EEGs_otp[ii,:] + minRes),(self.SN_max - self.SN_min)) + self.SN_min
        if (num_segments == 1):
            EEGs_crt.shape = (EEGs_crt.shape[1])
        return EEGs_crt
