#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(82018)
# --- split OASIS-3 into train/dev/test with 80%/5%/15% ---
df   = pd.read_csv("FS_scans_metadata.csv")
IDs  = list(df["ID"])
Ages = list(df["age"])
Disease = list(df["HC0/AD1"])

print("The max age for OASIS3 is {}".format(max(Ages)))

def oversample(Ages):
    frequencies = Ages

    frequencies = np.array(frequencies).round()  # quantize (essentially bin)
    ages, counts = np.unique(frequencies, return_counts=True)  # get age hist.
    prob = counts/counts.sum()  # get age probabilities

    # Get inverted probabilities:
    iprob = 1 - prob
    iprob = iprob/iprob.sum()  # renormalize

    ifrequencies = frequencies.copy()
    for i in range(len(ages)):
        idx = np.where(frequencies == ages[i])
        ifrequencies[idx] = iprob[i]

    ifrequencies = ifrequencies/ifrequencies.sum()

    return ifrequencies
ifrequencies = oversample(Ages)
train_idx = np.random.choice(len(IDs), round(len(IDs)*0.8),  replace=False, p=ifrequencies)
train_list = np.array(IDs)[train_idx]

left_IDs = [id for id in IDs if id not in train_list]
left_ages = []
for id in left_IDs:
    left_ages.append(Ages[IDs.index(id)])

ifrequencies = oversample(left_ages)
test_idx  = np.random.choice(len(left_IDs), round(len(left_IDs)*0.75), replace=False, p=ifrequencies)
test_list = np.array(left_IDs)[test_idx]

valid_list = [id for id in left_IDs if id not in test_list]
# plot
print("Split train:{}/valid:{}/test:{} according to ages distribution".format(len(train_list), len(valid_list), len(test_list)))

def plot_stacked_bar(Ages, Diseases, title):
    x = np.array(Ages).round()
    uni_x = np.unique(x)
    y1 = []
    y2 = []
    for i in uni_x:
        idexes = np.where(x == i)
        dis_i  = np.array(Diseases)[idexes]
        idx_hc = np.where(dis_i == 0)
        idx_ad = np.where(dis_i == 1)
        y1.append(idx_hc[0].shape[0])
        y2.append(idx_ad[0].shape[0])

    plt.rcParams.update({'font.size': 20})

    plt.bar(uni_x, y1, color='orange')
    plt.bar(uni_x, y2, bottom=y1, color='b')
    plt.xlabel("Ages")
    plt.ylabel("Count of scans")
    plt.legend(["HC", "AD"])
    # plt.title(title)
    plt.savefig('./figs/'+title + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

plot_stacked_bar(Ages, Disease, 'OASIS3 All scans (2366 images)')

def find_age_disease_for_given_subset(subset_list):
    ages = []
    diseases = []
    for id in subset_list:
        ages.append(Ages[IDs.index(id)])
        diseases.append(Disease[IDs.index(id)])
    return ages, diseases

plot_stacked_bar(*(find_age_disease_for_given_subset(train_list)), 'OASIS3 train scans (1893 images)')
plot_stacked_bar(*(find_age_disease_for_given_subset(test_list)), 'OASIS3 test scans (355 images)')
plot_stacked_bar(*(find_age_disease_for_given_subset(valid_list)), 'OASIS3 valid scans (118 images)')

plot_stacked_bar(*(find_age_disease_for_given_subset(valid_list+list(test_list))), 'OASIS3 test scans (473 images)')

csv_file = "FS_scans_metadata_with_split.csv"
def calcu_idx(list):
    idx = []
    for id in list:
        idx.append(IDs.index(id))
    print(len(idx))
    return idx
partition = np.empty(len(IDs), dtype="S5")
partition[train_idx] = "train"
partition[calcu_idx(test_list)] = "test"
partition[calcu_idx(valid_list)] = "dev"
partition = [p.decode("utf-8") for p in partition]
df['Partition'] = partition
# pd.DataFrame(df).to_csv(csv_file, index=False)