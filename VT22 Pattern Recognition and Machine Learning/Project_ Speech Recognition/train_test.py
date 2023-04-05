from PRClasses import DiscreteD, GaussD, HMM, MarkovChain
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import sounddevice as sd

from matplotlib import pyplot as plt
import numpy as np
import os

def GetSpeechFeatures(signal, fs, winlen, winstep, nfilt, nfft, winfunc):
    '''
    :param signal:
    :param fs:
    :param winlen:
    :param winstep:
    :param nfilt:
    :param nfft:
    :param winfunc:
    :return: the normalized MFCCs
    '''
    features_mfcc = mfcc(signal, fs,
                         winlen=winlen, winstep=winstep,
                         nfilt=nfilt,
                         nfft=nfft,
                         winfunc=winfunc)

    features_mfcc = features_mfcc.T
    features_mfcc_nor = np.zeros((features_mfcc.shape[0], features_mfcc.shape[1]))
    for i in range(features_mfcc.shape[0]):
        features_mfcc_nor[i, :] = features_mfcc[i, :] - np.mean(features_mfcc[i, :])
        features_mfcc_nor[i, :] = features_mfcc_nor[i, :] / np.std(features_mfcc_nor[i, :])

    return features_mfcc_nor


#%%
# Cross-validation 1: first 40% for training, last 30% for testing
num_s_all = np.array([6, 4, 6, 4, 4, 4, 6, 6, 4, 5])
wav_dir_all = ['train/audio/down', 'train/audio/go', 'train/audio/left',
               'train/audio/no', 'train/audio/off', 'train/audio/on',
               'train/audio/right', 'train/audio/stop', 'train/audio/up',
               'train/audio/yes']
fs = int(16000)
t = 1
mc_all = {}
sD_all = {}

# Train
for word_i in range(10):

    # The path of each wav clip
    wav_dir = wav_dir_all[word_i]
    wav_list = os.listdir(wav_dir)

    # Initialization of q, A and state-conditional distributions
    num_s = num_s_all[word_i]
    ndim = 13
    q = np.zeros(num_s)
    q[0] = 1
    A = np.zeros((num_s, num_s + 1))
    for i in range(A.shape[0]):
        A[i, i] = 0.5
        A[i, i + 1] = 0.5

    meanD = np.zeros(ndim)
    covD = np.eye(ndim)

    sD = [None] * num_s
    for i in range(num_s):
        sD[i] = GaussD(means=meanD, cov=covD)

    # Initialization of  MarkovChain and HMM model
    mc = MarkovChain(q, A)
    h = HMM(mc, sD)

    # Training the HMM model
    nsample = int(np.floor(len(wav_list)*0.4))  # the number of clips used for training
    sig = np.zeros((nsample, fs))
    num_train = 0
    mfcc_all = {}
    for i in range(nsample):
        wav_path = os.path.join(wav_dir, wav_list[i])
        _, sig_temp = wav.read(wav_path)
        if len(sig_temp) == sig.shape[1]:
            sig[i] = sig_temp
            # Calculating the MFCCs of each training audio
            mfcc_test = GetSpeechFeatures(sig[i], fs, 0.02, 0.02, 30, 320, np.hanning)
            _, px, _ = h.Get_px(mfcc_test)
            ifinf = np.isinf(px)
            ifnan = np.isnan(px)
            test_ovfl = np.count_nonzero(ifinf) + np.count_nonzero(ifnan)
            # If overflowed, skip this clip sample
            if test_ovfl > 0:
                continue
            # If not overflowed, train the HMM with the current clip
            else:
                mfcc_all[num_train] = mfcc_test
                num_train += 1

    A_new, cov_new = h.train(mfcc_all)

    # Updating the mc and HMM model
    mc = MarkovChain(q, A_new)
    sD_new = {}
    for i in range(num_s):
        sD_new[i] = GaussD(means=meanD, cov=cov_new[i])

    mc_all[word_i] = mc
    sD_all[word_i] = sD_new


# Test
C_rate = np.zeros((10, 10))
for word_i in range(10):

    # The path of each wav clip
    wav_dir = wav_dir_all[word_i]
    wav_list = os.listdir(wav_dir)

    # Test the HMM model
    nsample = int(np.floor(len(wav_list)*0.3))  # the number of clips used for training
    C_row = np.zeros(10)
    num_test = 0

    for i in range(len(wav_list)-nsample, len(wav_list)):
        wav_path = os.path.join(wav_dir, wav_list[i])
        _, sig_temp = wav.read(wav_path)
        if len(sig_temp) == fs:
            # Calculating the MFCCs of each training audio
            mfcc_test = GetSpeechFeatures(sig_temp, fs, 0.02, 0.02, 30, 320, np.hanning)

            prob = [None] * 10
            test_ovfl = 0
            for j in range(10):
                h_test = HMM(mc_all[j], sD_all[j])
                _, px, _ = h_test.Get_px(mfcc_test)
                ifinf = np.isinf(px)
                ifnan = np.isnan(px)
                test_ovfl += np.count_nonzero(ifinf) + np.count_nonzero(ifnan)

            # If overflowed, skip this clip sample
            if test_ovfl > 0:
                continue
                # If not overflowed, test the HMM with the current clip
            else:
                num_test += 1
                for k in range(10):
                    h_test = HMM(mc_all[k], sD_all[k])
                    prob[k] = h_test.logprob(mfcc_test)

                decision = prob.index(max(prob))
                for m in range(10):
                    if decision == m:
                         C_row[m] += 1

    C_rate[word_i, :] = C_row / num_test




 #%%
# Cross-validation 2: last 40% for training, first 30% for testing
num_s_all = np.array([6, 4, 6, 4, 4, 4, 6, 6, 4, 5])
wav_dir_all = ['train/audio/down', 'train/audio/go', 'train/audio/left',
               'train/audio/no', 'train/audio/off', 'train/audio/on',
               'train/audio/right', 'train/audio/stop', 'train/audio/up',
               'train/audio/yes']
fs = int(16000)
t = 1
mc_all = {}
sD_all = {}

# Train
for word_i in range(10):

    # The path of each wav clip
    wav_dir = wav_dir_all[word_i]
    wav_list = os.listdir(wav_dir)

    # Initialization of q, A and state-conditional distributions
    num_s = num_s_all[word_i]
    ndim = 13
    q = np.zeros(num_s)
    q[0] = 1
    A = np.zeros((num_s, num_s + 1))
    for i in range(A.shape[0]):
        A[i, i] = 0.5
        A[i, i + 1] = 0.5

    meanD = np.zeros(ndim)
    covD = np.eye(ndim)

    sD = [None] * num_s
    for i in range(num_s):
        sD[i] = GaussD(means=meanD, cov=covD)

    # Initialization of  MarkovChain and HMM model
    mc = MarkovChain(q, A)
    h = HMM(mc, sD)

    # Training the HMM model
    nsample = int(np.floor(len(wav_list)*0.4))  # the number of clips used for training
    # sig = np.zeros((nsample, fs))
    num_train = 0
    mfcc_all = {}
    for i in range(len(wav_list)-nsample, len(wav_list)):
        wav_path = os.path.join(wav_dir, wav_list[i])
        _, sig_temp = wav.read(wav_path)
        if len(sig_temp) == fs:
            # sig[i] = sig_temp
            # Calculating the MFCCs of each training audio
            mfcc_test = GetSpeechFeatures(sig_temp, fs, 0.02, 0.02, 30, 320, np.hanning)
            _, px, _ = h.Get_px(mfcc_test)
            ifinf = np.isinf(px)
            ifnan = np.isnan(px)
            test_ovfl = np.count_nonzero(ifinf) + np.count_nonzero(ifnan)
            # If overflowed, skip this clip sample
            if test_ovfl > 0:
                continue
            # If not overflowed, train the HMM with the current clip
            else:
                mfcc_all[num_train] = mfcc_test
                num_train += 1

    A_new, cov_new = h.train(mfcc_all)

    # Updating the mc and HMM model
    mc = MarkovChain(q, A_new)
    sD_new = {}
    for i in range(num_s):
        sD_new[i] = GaussD(means=meanD, cov=cov_new[i])

    mc_all[word_i] = mc
    sD_all[word_i] = sD_new


# Test
C_rate = np.zeros((10, 10))
for word_i in range(10):

    # The path of each wav clip
    wav_dir = wav_dir_all[word_i]
    wav_list = os.listdir(wav_dir)

    # Test the HMM model
    nsample = int(np.floor(len(wav_list)*0.3))  # the number of clips used for training
    C_row = np.zeros(10)
    num_test = 0

    for i in range(nsample):
        wav_path = os.path.join(wav_dir, wav_list[i])
        _, sig_temp = wav.read(wav_path)
        if len(sig_temp) == fs:
            # Calculating the MFCCs of each training audio
            mfcc_test = GetSpeechFeatures(sig_temp, fs, 0.02, 0.02, 30, 320, np.hanning)

            prob = [None] * 10
            test_ovfl = 0
            for j in range(10):
                h_test = HMM(mc_all[j], sD_all[j])
                _, px, _ = h_test.Get_px(mfcc_test)
                ifinf = np.isinf(px)
                ifnan = np.isnan(px)
                test_ovfl += np.count_nonzero(ifinf) + np.count_nonzero(ifnan)

            # If overflowed, skip this clip sample
            if test_ovfl > 0:
                continue
                # If not overflowed, test the HMM with the current clip
            else:
                num_test += 1
                for k in range(10):
                    h_test = HMM(mc_all[k], sD_all[k])
                    prob[k] = h_test.logprob(mfcc_test)

                decision = prob.index(max(prob))
                for m in range(10):
                    if decision == m:
                        C_row[m] += 1

    C_rate[word_i, :] = C_row / num_test



