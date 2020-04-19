#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 08:20:28 2020

@author: nmei
"""

import os
from glob import glob

import mne

import seaborn as sns
sns.set_context('poster') # huge font size

working_dir = "../data"
working_data = glob(os.path.join(working_dir,"*.fif"))
