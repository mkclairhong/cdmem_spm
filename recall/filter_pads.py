import os
import pandas as pd
import numpy as np
from glob import glob
import pdb
# allPatterns = ['rFrontal_Med_Orb_lr.txt', 'rFusiform_lr.txt','rHippocampus_lr.txt','rParaHippocampal_lr.txt','rTemporal_Inf_lr.txt',\
                        # 'rTemporal_Mid_lr.txt','rTemporal_Pole_Mid_lr.txt','rTemporal_Pole_Sup_lr.txt', 'rTemporal_Sup_lr.txt']

def filter_pads(subjID):
    subjID = str(subjID).zfill(3)
    subjID = 'cdcatmr' + subjID
    subjDir = '/scratch/polynlab/fmri/cdcatmr/' + subjID
    imgDir = subjDir + '/images/func'
    patDir = subjDir + '/patterns/recall_pat'

    # read in all subject's rec_onset csv files from each run
    all_recs = [y for x in os.walk(subjDir) for y in glob(os.path.join(x[0], 'rec_onsets_*.csv'))]
    # sort so that its in ascending order
    all_recs.sort()

    # compile all run's recall onset csv files into one dataframe
    subj_rec = pd.DataFrame()
    for r in all_recs:
        temp_r = pd.read_csv(r, names=['onset'])  # we assign a temporary column name
        subj_rec = subj_rec.append(temp_r)
    subj_rec.reset_index(drop=True, inplace=True)

    # get the index with 0s
    rec_pads = subj_rec.index[subj_rec['onset'] == 0].tolist()
    rec_pads = [x for x in rec_pads if x < 215]

    # direct path to the pattern folder
    pat_files = os.listdir(patDir)
    pat_files = [x for x in pat_files if x.startswith('r') and x.endswith('.txt')]
    total_pat = len(pat_files)
    pat_files.sort()
    # pdb.set_trace()
    os.chdir(patDir)
    k = 0
    print('filtering out 0 padded rows')
    for p in pat_files:
        k += 1
        try:
            pat = pd.read_csv(p)
            # drop the index with
            pat = pat.drop(pat.index[rec_pads])
            # overwrite as a new file
            pat.to_csv(p, index=False)
            print(str(k) + '/' + str(len(pat_files)) + ' overwrote pattern ' + str(p))
        except:
            print(str(k) + '/' + str(len(pat_files)) + ' failed pattern ' + str(p))
            continue
