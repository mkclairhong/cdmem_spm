import os
import pandas as pd
from scipy.io import loadmat
import numpy as np


def pickle_pattern(subjID):
    """
    Takes the initial .mat pattern file to create a more structured csv data frame
    including trial, item name, condition, and category info along with patterns

    Parameters
    ----------
    subjID : str
        str of subject folder name to be found in baseDir (ie. 'cdcatmr005')

    Returns
    -------
    csv
        creates a csv file for each pattern file found in the subject directory

    """

    #%% set directories
    subjID = str(subjID)
    baseDir = '/scratch/polynlab/fmri/cdcatmr/'
    subjDir = baseDir + subjID + '/'
    patDir = subjDir + 'patterns'
    eventsDir = subjDir + 'events'
    resDir = patDir + '/pkl'
    if not os.path.isdir(resDir):
        os.mkdir(resDir)

    #%% read in events and create df
    os.chdir(eventsDir)
    events = pd.read_csv(subjID + 'events.csv')
    events = events.query('types=="stim_pres"')

    cols = ['trialN','index', 'item', 'condition', 'category']
    pattern_events = pd.DataFrame(columns=cols)
    for i in cols:
        pattern_events[i] = events[i]
    pattern_events.reset_index(drop=True, inplace=True)
    pattern_events.to_csv(patDir + '/' + subjID + 'pattern_info.csv', index=False)

    #%% read in pat mat files
    os.chdir(patDir)
    pat_files = os.listdir(patDir)
    pat_files = [x for x in pat_files if x.endswith('.mat')]
    pat_files.sort()

    for p in range(0, len(pat_files)):
        filename = pat_files[p][:-4]  # remove the .mat at the end to get the roi name
        print('Starting ' + filename)
        pat_file = loadmat(pat_files[p])

        pat_df = pd.DataFrame(columns=['pats'])
        for i in range(0, np.shape(pat_file['pats'])[0]):
            pat_df.at[i, 'pats'] = pat_file['pats'][i]
        assert len(pattern_events) == len(pat_df), "length do not match to 216"
        pat_df.reset_index(drop=True, inplace=True)

        new_df = pd.DataFrame(columns=cols + ['pats'])
        new_df = pd.concat([pattern_events, pat_df], axis=1)

        # new_df.to_csv(resDir + '/' + filename + '.csv', index=False)
        new_df.to_pickle(resDir + '/' + filename + '.pkl')
