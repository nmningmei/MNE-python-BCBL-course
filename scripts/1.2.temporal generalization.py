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
from glob import glob

import mne

import numpy as np

working_dir = '../data'
working_data = glob(os.path.join(working_dir,
                                 '*-epo.fif'))

features = []
labels = []
groups = []
for f in working_data:
    subj = int(re.findall(r'\d+',f)[0])
    epochs = mne.read_epochs(f,preload = True)
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



from sklearn.utils import shuffle as sk_shuffle
features,labels,groups = sk_shuffle(features,labels,groups)

# the machine learning part - for detail comments, go to the .ipynb files
from mne.decoding import (Vectorizer,
                          SlidingEstimator,
                          cross_val_multiscore,
                          GeneralizingEstimator)
from sklearn.model_selection import (cross_validate,
                                     StratifiedShuffleSplit,
                                     LeaveOneGroupOut)
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
















