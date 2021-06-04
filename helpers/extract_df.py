import pandas as pd
import os
from glob import glob
"""
Script to run through each subject's full events.csv and extract just the columns we want from a particular event type.
Specify the desired columns from the full columns from events.csv, and it will save out a more concise version.
"""


# set variables
desired_columns = ['trialN', 'index', 'item', 'condition', 'category']
dataDir = '/scratch/polynlab/fmri/cdcatmr'
extract_type = 'stim_pres'


os.chdir(dataDir)
ls = os.listdir(dataDir)
ls = [x for x in ls if os.path.isdir(x) and x.startswith('cdcatmr')]
ls.sort()

for i in range(0, len(ls)):
    subjID = ls[i]
    if os.path.exists(subjID + '/events'):
        try:
            events_csv = pd.read_csv(subjID + '/events/' + subjID + 'events.csv')

            # create a list of columns to be dropped out
            drop_list = []
            for column in events_csv.columns:
                if column not in desired_columns:
                    drop_list.append(column)

            # create a new df just for stim_pres
            new_df = pd.DataFrame(columns=events_csv.columns)

            new_df = new_df.append(events_csv[events_csv['types'] == extract_type])
            assert len(new_df) == 216, "total stim item does not match"

            new_df = new_df.drop(columns=drop_list)
            new_df.to_csv(subjID + '/events/' + subjID + '_pat_' + extract_type + '.csv', index=False)
            print('Saved to ' + subjID + '/events/' + subjID + '_' + extract_type + '.csv')

        except IOError:  # if events.csv file is missing, skip
            pass
