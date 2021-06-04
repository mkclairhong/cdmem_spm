import pandas as pd
import numpy as np

def create_recall_onsets(subjID):
    n_items = 27
    subjID = str(subjID).zfill(3)
    subjID = 'cdcatmr' + subjID
    saveDir = '/scratch/polynlab/fmri/cdcatmr/' + subjID + '/images/func/'
    events_file = '/scratch/polynlab/fmri/cdcatmr/' + subjID + '/events/' + subjID + 'events.csv'
    event_file = pd.read_csv(events_file)
    
    for t in np.unique(event_file.trialN):
        recalls = event_file.query('trialN==@t & types=="rec_word"')
        onsets = list(recalls.onset_ttl_start)
        onsets = list(map(int, onsets))

        save_file = list([0] * n_items)
        save_file[:len(onsets)] = onsets
        save_file = pd.Series(save_file)
        save_file.to_csv(saveDir + 'func' + str(t) + '/rec_onsets_' + str(t) + '.csv', header=False, index=False)
        print('saved trial ' + str(t))
