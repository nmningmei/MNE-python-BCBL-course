#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 08:20:28 2020

@author: nmei

Please view 1_1_Cross_Validation_an_EEG_example.ipynb for full details with 
MNE-python publish dataset.

In this script, we will try to replicate these using Piermatteo's data

"""

import re
import os
from glob import glob

import mne

import seaborn as sns
sns.set_context('poster') # huge font size

from matplotlib import pyplot as plt

working_dir = "../data"
figure_dir = '../figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
working_data = glob(os.path.join(working_dir,"*.fif"))

epochs_concat = []
for f in working_data:
    subj = int(re.findall(r'\d+',f)[0])
    
    # load the continueous data
    raw = mne.io.read_raw_fif(f,preload = True)
    print(raw.info)
    
    # look at the trigger time stamps
    events = mne.find_events(raw,
                             # stim_channel = 'STI201', # don't know why it does not work
                              initial_event = True, # added according to the warnings
                             min_duration = 0.01, # added according error message
                             )
    events = mne.pick_events(events,include = [1,2]) # according to Piermatteo's method document
    events[:,1] = subj
    event_ids = {
        "high frequency tone":1,
        "low frequency tone":2,
        # "Omission of high frequency tone":3,
        # "Omission of low frequency tone":4,
        }
    picks = mne.pick_types(raw.info,meg = True,eeg = False,)
    epochs = mne.Epochs(raw,
                        events = events,
                        event_id = event_ids,
                        tmin = -20/raw.info['sfreq'],
                        tmax = 343/raw.info['sfreq'],
                        baseline = (-20/raw.info['sfreq'],0),
                        picks = picks,
                        preload = True,
                        )
    del raw
    # filter at the epochs level
    epochs.filter(None,80,
                  n_jobs = -1)
    for event_name in event_ids.keys():
        evoked = epochs[event_name].average()
        fig = evoked.plot_joint(title = f'sub{subj} - {event_name}')
        fig[0].savefig(os.path.join(figure_dir,
                                 f'sub{subj}_{event_name}.jpeg'))
    # combine the epochs from different subjects
    """
    ValueError: epochs[0]['info']['dev_head_t'] 
    must match. The epochs probably come from different runs, 
    and are therefore associated with different head positions. 
    Manually change info['dev_head_t'] to avoid this message but 
    beware that this means the MEG sensors will not be properly 
    spatially aligned. See mne.preprocessing.maxwell_filter to 
    realign the runs to a common head position.
    """
    epochs.save(os.path.join(working_dir,
                             f'sub{subj}-epo.fif'),
                overwrite = True)
    # epochs_concat.append(epochs)

# epochs_concat = mne.concatenate_epochs(epochs_concat)


# epochs_concat.save(os.path.join(working_dir, "epochs_concat-epo.fif"))












