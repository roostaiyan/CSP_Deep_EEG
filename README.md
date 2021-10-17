# CSP_Deep_EEG
This source code is implemented using keras library based on "Automatic ocular artifacts removal in EEG using deep learning" paper just for academics purposes, though a complete match with this paper is not guaranteed.
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

This project contains:
- Common Spatial Pattern for EEG Feature Extraction
- using Deep Stacked Sparse Autoencoders (Keras) for EEG Denoising
- Support Vector Machine for EEG Classification

Dependencies:
- Please download dataset 1 of the BCI Competition IV available on http://www.bbci.de/competition/iv/#datasets.
- sudo pip install keras
- sudo pip install tensorflow
- sudo pip install sklearn
- sudo pip install matplotlib
- sudo pip install mne
- sudo pip install IPython
