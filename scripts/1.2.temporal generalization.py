#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:50:28 2020

@author: nmei


Please view 1_1_Cross_Validation_an_EEG_example.ipynb for full details with 
MNE-python publish dataset.

In this script, we will try to show you how to perform cross validation
machine learning - K-fold and leave-one-subject-out

"""

import os
import re
import gc # to clear some memory for parallel processing during machine learning
from glob import glob

import mne

import numpy as np

from sklearn.utils import shuffle as sk_shuffle

working_dir = '../data'
working_data = glob(os.path.join(working_dir,
                                 '*-epo.fif'))

features = []
labels = []
groups = []


def chunk_average(epochs):
    """
    Because the number of trials is hugee, ~ 5000, I don't have the patience to 
    wait for the process and the cluster does not have the memory to process
    such a huge dataset, so I will average every 10 trials to boost the 
    signal-to-noise ratio and decrease the number of trials
    """
    # extract Numpy array format epoch data
    data = epochs.get_data()
    # extract Numpy array format epoch event data
    events = epochs.events
    # set up the random seed because we will perform shuffling
    np.random.seed(12345)
    # shuffle the data so that we average every 10 trials that are randomly
    # selected
    data,events = sk_shuffle(data,events)
    
    # put the data into chunks and average every chunk, and then concatenate them
    chunk_data = np.array([a.mean(0) for a in np.array_split(data, int(data.shape[0]/10))])
    chunk_event = np.array([a[0,:] for a in np.array_split(events, int(data.shape[0]/10))])
    # putting the averaged data back to an MNE-Epochs object, which is preferred
    # by me
    new_epochs = mne.EpochsArray(chunk_data,
                                 epochs.info,
                                 events = chunk_event,
                                 event_id = epochs.event_id,
                                 tmin = epochs.tmin,
                                 baseline = epochs.baseline,)
    
    return new_epochs
    
for f in working_data:
    subj = int(re.findall(r'\d+',f)[0])
    epochs = mne.read_epochs(f,preload = True)
    
    high = epochs[list(epochs.event_id.keys())[0]]
    low = epochs[list(epochs.event_id.keys())[1]]
    
    high = chunk_average(high)
    low = chunk_average(low)
    
    epochs = mne.concatenate_epochs([high,low])
    
    events = epochs.events
    data = epochs.get_data()
    # epochs.resample(256)
    
    del epochs
    
    features.append(data)
    labels.append(events[:,-1])
    groups.append([subj] * len(events))
    
    del data

features = np.concatenate(features)
labels = np.concatenate(labels)
groups = np.concatenate(groups)




features,labels,groups = sk_shuffle(features,labels,groups)

# the machine learning part - for detail comments, go to the .ipynb files
from mne.decoding import (Vectorizer,
                          SlidingEstimator,
                          cross_val_multiscore,
                          GeneralizingEstimator)
from sklearn.model_selection import (cross_validate,
                                     StratifiedShuffleSplit,
                                     LeaveOneGroupOut,
                                     permutation_test_score)
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# decode the whole epoch
##  shuffle K - fold
cv = StratifiedShuffleSplit(n_splits = 10, 
                            test_size = .4, 
                            random_state = 12345,
                            )
svc = LinearSVC(penalty = 'l2',# due to the massive number of features (n_channels x n_time), we need to control for overfitting
                loss = 'squared_hinge',
                dual = True,
                tol = 1e-3, 
                C = 1., 
                fit_intercept = False, 
                class_weight = 'balanced',
                random_state = 12345,
                )
pipeline = make_pipeline(Vectorizer(),
                         StandardScaler(),
                         svc)
res = cross_validate(estimator = pipeline,
                     X = features,
                     y = np.array(labels == 1,dtype = int),
                     groups = None,
                     scoring = 'roc_auc',
                     cv = cv,
                     n_jobs = -1,
                     verbose = 1, 
                     # return_estimator = True,
                     )
print(res['test_score'])
gc.collect()
score,purms,pval = permutation_test_score(estimator = pipeline,
                                          X = features,
                                          y = np.array(labels == 1, dtype = int),
                                          groups = None,
                                          scoring = 'roc_auc',
                                          cv = cv,
                                          n_jobs = -1,
                                          verbose = 1,
                                          )















