#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from scipy.signal import stft
import seaborn as sns
from os import listdir
import os.path as op
import glob
import pickle
import mne
from mne import Info
sns.set()


# In[25]:


def ERD_MEAN(exp_name):
    
    electrodes_list = ['EEG Fp1-Cz', 'EEG Fp2-Cz', 'EEG F3-Cz', 'EEG F4-Cz', 'EEG F7-Cz', 'EEG F8-Cz']
    
    raw_cal = mne.io.read_raw_edf(cal_name[0], preload=True)
    raw_exp = mne.io.read_raw_edf(exp_name[0], preload=True)
    sfreq = raw_exp.info['sfreq']

    # Data from specific channels
    eyes = raw_cal.copy().pick_channels(ch_names=electrodes_list);
    experiment = raw_exp.copy().pick_channels(ch_names=electrodes_list)

    # Filtering AC line noise with notch filter

    # eyes_filtered_data = mne.filter.notch_filter(x=eyes.get_data(), Fs=sfreq, freqs=[50, 100])
    # experiment_filtered_data = mne.filter.notch_filter(x=experiment.get_data(), Fs=sfreq, freqs=[50, 100])
    eyes_filtered_data = eyes.get_data()
    experiment_filtered_data = experiment.get_data()

    # Preparing data for plotting
    eyes_filtered = mne.io.RawArray(data=eyes_filtered_data,
                                    info=mne.create_info(ch_names=electrodes_list, sfreq=sfreq))
    experiment_filtered = mne.io.RawArray(data=experiment_filtered_data,
                                          info=mne.create_info(ch_names=electrodes_list, sfreq=sfreq))

    IAF_p = IAF(20)

    # Getting L1A, L2A, UA, Theta waves from eyes closed using FIR filtering. Also we take mean signal from all channels
    eyes_sub_bands = {}

    eyes_sub_bands['L1A'] = mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 4,
                                                   h_freq=IAF_p - 2, sfreq=sfreq, method="fir")
    eyes_sub_bands['L2A'] = mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 2,
                                                   h_freq=IAF_p, sfreq=sfreq, method="fir")
    

    # Getting L1A, L2A, UA, Theta waves from experiment data using FIR filtering. Also we take mean signal from all
    # channels
    experiment_sub_bands = {}

    experiment_sub_bands['L1A'] = mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                         l_freq=IAF_p - 4, h_freq=IAF_p - 2, sfreq=sfreq, method="fir")
    experiment_sub_bands['L2A'] = mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                         l_freq=IAF_p - 2, h_freq=IAF_p, sfreq=sfreq, method="fir")
    

    # Calculating calibration values. Consider mean value of all channels. Va;ue are given in microvolts
    calibration_values = {}

    for band in eyes_sub_bands:
        calibration_values[band] = np.mean(eyes_sub_bands[band], axis=0) * np.power(10, 6)

    # Performing STFT transform on experiment data for each sub-band. Window size is given in samples
    window = sfreq * 2
    fft = {}

    for band in experiment_sub_bands:
        fft[band] = stft(x=experiment_sub_bands[band], fs=sfreq, window=('kaiser', window), nperseg=1000)

    erd = np.vectorize(ERD)
    # Calculating ERD for experiment
    erd_mean = {}
    erd_all = {}

    for band in fft:
        erd_all[band] = erd(fft[band][2], calibration_values[band])
        erd_mean[band] = np.mean(erd_all[band], axis=0)
    
    return erd_mean


# In[26]:


def get_experiments_mean(load_dir):
    
    ERD_MEANS = []
    directories = listdir(load_dir)

    for directory in directories:
        
        path = os.path.join(load_dir, directory)
        cal_name = glob.glob(path+'\*Calibration.Opened.edf')
        exp_name1 = glob.glob(path+'\*.Data1.edf')
        exp_name2 = glob.glob(path+'\*.Data2.edf')

        erd_mean1 = ERD_MEAN(exp_name1)
        erd_mean2 = ERD_MEAN(exp_name2)
        data1_All_mean = (np.mean(erd_mean1['L1A']) +np.mean(erd_mean1['L2A'])) / 2
        data2_All_mean = (np.mean(erd_mean2['L1A']) +np.mean(erd_mean2['L2A'])) / 2
        ERD_MEANS.append((data1_All_mean + data2_All_mean)/2)
    
    return ERD_MEANS


# In[35]:


def print_results(ERD_MEANS, exp_name):
    
    print('\n\n')
    print('----------------------- ' + exp_name +  ' DATA MEAN -----------------------\n')
    print('ERD MEAN FOR WHOLE DATA: ' + str(np.mean(ERD_MEANS)))
    print('\n\n')

    plt.xlabel(exp_name + '-Experiments')
    plt.ylabel('Mean ERD')
    plt.plot(np.real(ERD_MEANS))
    plt.show()
    plt.savefig(obg_dir + '.png', dpi=100)
    


# In[33]:


# Function to calculate IAF
def IAF(age):
    return 11.95 - 0.053 * age


# Implementing function to calculate ERD
def ERD(f, cal):
    return 100 * (cal - f) / cal


load_dir_control = os.path.abspath(r'D:\\LIPS EEG EXPERIMENTS\\DATA WITHOUT ARTIFACTS\\Control')
load_dir_extra = os.path.abspath(r'D:\\LIPS EEG EXPERIMENTS\\DATA WITHOUT ARTIFACTS\\Extra')
obg_dir = os.path.abspath(r'D:\\LIPS EEG EXPERIMENTS\\DATA WITHOUT ARTIFACTS\\Dumps')

ERD_MEANS_CONTROL = get_experiments_mean(load_dir_control)   
ERD_MEANS_EXTRA = get_experiments_mean(load_dir_extra)    

print_results(ERD_MEANS_CONTROL)
print_results(ERD_MEANS_EXTRA)

