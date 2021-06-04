import warnings
import sys
import os
import pandas as pd
import numpy as np
import scipy.io
from scipy import stats
import matplotlib.pyplot as plts
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import seaborn as sns
from sklearn import datasets
from sklearn import svm
from scipy.io import loadmat
import pdb

def class_byroi(subjID):
    #%% set directories for ACCRE setup
    baseDir = '/scratch/polynlab/fmri/cdcatmr/'
    subjID = 'cdcatmr' + str(subjID).zfill(3)
    subjDir = baseDir + subjID
    n_trials = 8
    n_items = 216

    roiloop = []

    events = pd.read_csv(subjDir + '/events/' + subjID + 'events.csv')
    scans = pd.read_csv(baseDir + 'all_subjects/study_beta_scans.csv', header=None)
    # cond = pd.read_csv(subjDir + '/events/' + subjID + 'cond.csv', header=None)


    #%% data frame output
    def add_row(df, row):
        """
        Appends the given row list value to the last row of df
        ** make sure row values matches the desired df column
        """
        assert len(df.columns) == len(row), "length of input row value needs to match len of df columns"
        indx = len(df) + 1
        cols = df.columns
        for i in range(0, len(row)):
            df.at[indx, cols[i]] = row[i]
        return df
    df_columns = ['subjID', 'roi', 'class', 'cond', 'train_pos', 'all', 'cel', 'loc', 'obj']
    df = pd.DataFrame(columns=df_columns)


    #%% identify condition (distraction) for each trial
    cond = []
    for t in range(1, n_trials+1):
        x = events.query('trialN==@t').iloc[0]['condition']
        cond.append(int(x))
    assert len(cond) == n_trials
    cond = pd.DataFrame(np.array(cond))
    cond = pd.concat([cond]* 27, axis=1)
    cond = cond.stack()


    #%% read in patterns
    # loop all rois
    # allPatterns = os.listdir(subjDir + '/patterns/txt')
    # allPatterns = [x for x in allPatterns if x.endswith('.txt')]
    # allPatterns.sort()
    os.chdir(subjDir + '/patterns/txt')
    # loop only several
    allPatterns = ['rFrontal_Med_Orb_lr.txt', 'rFusiform_lr.txt','rHippocampus_lr.txt','rParaHippocampal_lr.txt','rTemporal_Inf_lr.txt',\
                        'rTemporal_Mid_lr.txt','rTemporal_Pole_Mid_lr.txt','rTemporal_Pole_Sup_lr.txt', 'rTemporal_Sup_lr.txt']


    #%% run classifiers for each roi file
    progress = 1
    skipped = []
    for p in range(0, len(allPatterns)):
        accuracy_LDA = np.zeros(n_items)
        accuracy_LR = np.zeros(n_items)
        accuracy_KNN = np.zeros(n_items)

        pred_LDA = np.zeros(n_items)
        pred_LR = np.zeros(n_items)
        pred_KNN = np.zeros(n_items)

        try:
            pats = pd.read_csv(allPatterns[p], header=None)
            print(str(progress) + '/' + str(len(allPatterns)) + ': ' + allPatterns[p] + ' started')
        except IOError or FileNotFoundError:
            warnings.warn('No ' + allPatterns[p] + ' file found. Skipping...')
            skipped.append(allPatterns[p])
            continue
        except pd.errors.EmptyDataError:
            warnings.warn('Empty file ' + allPatterns[p] + '. Skipping...')
            skipped.append(allPatterns[p])
            continue
        X = pats

        # target labels
        st = events[['trialN', 'types', 'item', 'category']].copy()
        stim_pres = st[st['types'] == 'stim_pres']

        y = stim_pres['category']
        y = y.reset_index()
        y = y['category']


        ####################################
        # feature selection
        ####################################
        try:
            selectVox = SelectKBest(score_func=f_classif, k=100)
            fit = selectVox.fit(X, y)
        except ValueError:
            selectVox = SelectKBest(score_func=f_classif, k='all')
            fit = selectVox.fit(X, y)
            print('k set to all for ' + allPatterns[p])
        # fit = selectVox.fit(X, y)

        set_printoptions(precision=3)
        # print(fit.scores_)
        X = fit.transform(X)

        cnt = 0;
        for t in range(0, n_items): # items

            scaler = StandardScaler()
            scaler.fit(X)
            X=scaler.transform(X)

            X_train     = X[np.arange(len(X))!=t]
            X_test      = X[np.arange(len(X))==t]

            y_train     = y[np.arange(len(y))!=t]
            y_test      = y[np.arange(len(y))==t]

            lda         = LinearDiscriminantAnalysis()
            lda         = lda.fit(X_train, y_train)
            y_pred_lda  = lda.predict(X_test)

            #print('LDA: ' + str(accuracy_score(y_test, y_pred)))
            accuracy_LDA[cnt] = accuracy_score(y_test, y_pred_lda)
            pred_LDA[cnt] = y_pred_lda[0]

            logreg = LogisticRegression(solver='newton-cg', multi_class='multinomial', max_iter=1000)
            logreg = logreg.fit(X_train, y_train)
            y_pred_lr = logreg.predict(X_test)
            #print('LR: ' + str(accuracy_score(y_test, y_pred)))
            accuracy_LR[cnt] = accuracy_score(y_test, y_pred_lr)
            pred_LR[cnt] = y_pred_lr[0]

            KNN = KNeighborsClassifier(n_neighbors=5, metric='hamming')
            KNN = KNN.fit(X_train, y_train)
            y_pred_KNN = KNN.predict(X_test)
            #print('LDA: ' + str(accuracy_score(y_test, y_pred)))
            accuracy_KNN[cnt] = accuracy_score(y_test, y_pred_KNN)
            pred_KNN[cnt] = y_pred_KNN[0]  # Jin: had to sepcify '[0]' because y_pred_KNN was an array and can't assign array as a single object inside pred_KNN

            cnt = cnt + 1
        cats = np.asarray(y)
        temp0 = cats[cond==0]
        temp1 = cats[cond==1]

        cat0 = temp0.reshape(36,3)
        cat1 = temp1.reshape(36,3)

        ###############################################
        # look at predicted categories based on train
        # position - integration analyses
        # KNN
        ##############################################
        roitemp = np.zeros([18,4])
        pred_KNN = pred_KNN.reshape(216)
        # comp = np.array(np.equal(y.astype(int),pred_KNN.astype(int)))
        # comp = np.array(np.equal(y,pred_KNN))
        comp = np.array(np.equal(np.array(y.astype(int)),pred_KNN.astype(int)))  # Jin: had to change it to match the array type of y and pred_KNN (otherwise, it only gave everything False)

        # comp
        KNN_0 = comp[cond==0]
        KNN_1 = comp[cond==1]
        KNN0 = KNN_0.reshape(36,3)
        KNN1 = KNN_1.reshape(36,3)
        KNN1.shape

        roitemp[0,0] = str(np.mean(KNN0[:,0]))
        roitemp[1,0] = str(np.mean(KNN0[:,1]))
        roitemp[2,0] = str(np.mean(KNN0[:,2]))
        roitemp[3,0] = str(np.mean(KNN1[:,0]))
        roitemp[4,0] = str(np.mean(KNN1[:,1]))
        roitemp[5,0] = str(np.mean(KNN1[:,2]))

        roitemp[0,1] = str(np.mean(KNN0[cat0[:,0] == '1', 0]))
        roitemp[1,1] = str(np.mean(KNN0[cat0[:,1] == '1', 1]))
        roitemp[2,1] = str(np.mean(KNN0[cat0[:,2] == '1', 2]))
        roitemp[3,1] = str(np.mean(KNN1[cat1[:,0] == '1', 0]))
        roitemp[4,1] = str(np.mean(KNN1[cat1[:,1] == '1', 1]))
        roitemp[5,1] = str(np.mean(KNN1[cat1[:,2] == '1', 2]))

        roitemp[0,2] = str(np.mean(KNN0[cat0[:,0] == '2', 0]))
        roitemp[1,2] = str(np.mean(KNN0[cat0[:,1] == '2', 1]))
        roitemp[2,2] = str(np.mean(KNN0[cat0[:,2] == '2', 2]))
        roitemp[3,2] = str(np.mean(KNN1[cat1[:,0] == '2', 0]))
        roitemp[4,2] = str(np.mean(KNN1[cat1[:,1] == '2', 1]))
        roitemp[5,2] = str(np.mean(KNN1[cat1[:,2] == '2', 2]))

        roitemp[0,3] = str(np.mean(KNN0[cat0[:,0] == '3', 0]))
        roitemp[1,3] = str(np.mean(KNN0[cat0[:,1] == '3', 1]))
        roitemp[2,3] = str(np.mean(KNN0[cat0[:,2] == '3', 2]))
        roitemp[3,3] = str(np.mean(KNN1[cat1[:,0] == '3', 0]))
        roitemp[4,3] = str(np.mean(KNN1[cat1[:,1] == '3', 1]))
        roitemp[5,3] = str(np.mean(KNN1[cat1[:,2] == '3', 2]))
        KNN1
        # currently, hacky+bulky+hardcoded way of adding to df
        # cond 0
        add_row(df, [subjID, allPatterns[p], 'knn', 0, 1, np.mean(KNN0[:,0]), \
                np.mean(KNN0[cat0[:,0] == '1', 0]), np.mean(KNN0[cat0[:,0] == '2', 0]), np.mean(KNN0[cat0[:,0] == '3', 0])])
        add_row(df, [subjID, allPatterns[p], 'knn', 0, 2, np.mean(KNN0[:,1]), \
                np.mean(KNN0[cat0[:,1] == '1', 1]), np.mean(KNN0[cat0[:,1] == '2', 1]), np.mean(KNN0[cat0[:,1] == '3', 1])])
        add_row(df, [subjID, allPatterns[p], 'knn', 0, 3, np.mean(KNN0[:,2]), \
                np.mean(KNN0[cat0[:,2] == '1', 2]), np.mean(KNN0[cat0[:,2] == '2', 2]), np.mean(KNN0[cat0[:,2] == '3', 2])])
        # cond 1
        add_row(df, [subjID, allPatterns[p], 'knn', 1, 1, np.mean(KNN1[:,0]), \
                np.mean(KNN1[cat1[:,0] == '1', 0]), np.mean(KNN1[cat1[:,0] == '2', 0]), np.mean(KNN1[cat1[:,0] == '3', 0])])
        add_row(df, [subjID, allPatterns[p], 'knn', 1, 2, np.mean(KNN1[:,1]), \
                np.mean(KNN1[cat1[:,1] == '1', 1]), np.mean(KNN1[cat1[:,1] == '2', 1]), np.mean(KNN1[cat1[:,1] == '3', 1])])
        add_row(df, [subjID, allPatterns[p], 'knn', 1, 3, np.mean(KNN1[:,2]), \
                np.mean(KNN1[cat1[:,2] == '1', 2]), np.mean(KNN1[cat1[:,2] == '2', 2]), np.mean(KNN1[cat1[:,2] == '3', 2])])


        ##############################################
        # LDA integration
        ##############################################
        pred_LDA = pred_LDA.reshape(216)
        comp = np.array(np.equal(y.astype(int),pred_LDA.astype(int)))
        LDA_0 = comp[cond==0]
        LDA_1 = comp[cond==1]

        LDA0 = LDA_0.reshape(36,3)
        LDA1 = LDA_1.reshape(36,3)

        roitemp[6,0] = str(np.mean(LDA0[:,0]))
        roitemp[7,0] = str(np.mean(LDA0[:,1]))
        roitemp[8,0] = str(np.mean(LDA0[:,2]))
        roitemp[9,0] = str(np.mean(LDA1[:,0]))
        roitemp[10,0] = str(np.mean(LDA1[:,1]))
        roitemp[11,0] = str(np.mean(LDA1[:,2]))

        roitemp[6,1] = str(np.mean(LDA0[cat0[:,0] == '1', 0]))
        roitemp[7,1] = str(np.mean(LDA0[cat0[:,1] == '1', 1]))
        roitemp[8,1] = str(np.mean(LDA0[cat0[:,2] == '1', 2]))
        roitemp[9,1] = str(np.mean(LDA1[cat1[:,0] == '1', 0]))
        roitemp[10,1] = str(np.mean(LDA1[cat1[:,1] == '1', 1]))
        roitemp[11,1] = str(np.mean(LDA1[cat1[:,2] == '1', 2]))

        roitemp[6,2] = str(np.mean(LDA0[cat0[:,0] == '2', 0]))
        roitemp[7,2] = str(np.mean(LDA0[cat0[:,1] == '2', 1]))
        roitemp[8,2] = str(np.mean(LDA0[cat0[:,2] == '2', 2]))
        roitemp[9,2] = str(np.mean(LDA1[cat1[:,0] == '2', 0]))
        roitemp[10,2] = str(np.mean(LDA1[cat1[:,1] == '2', 1]))
        roitemp[11,2] = str(np.mean(LDA1[cat1[:,2] == '2', 2]))

        roitemp[6,3] = str(np.mean(LDA0[cat0[:,0] == '3', 0]))
        roitemp[7,3] = str(np.mean(LDA0[cat0[:,1] == '3', 1]))
        roitemp[8,3] = str(np.mean(LDA0[cat0[:,2] == '3', 2]))
        roitemp[9,3] = str(np.mean(LDA1[cat1[:,0] == '3', 0]))
        roitemp[10,3] = str(np.mean(LDA1[cat1[:,1] == '3', 1]))
        roitemp[11,3] = str(np.mean(LDA1[cat1[:,2] == '3', 2]))

        # currently, hacky+bulky+hardcoded way of adding to df
        # cond 0
        add_row(df, [subjID, allPatterns[p], 'lda', 0, 1, np.mean(LDA0[:,0]), \
                np.mean(LDA0[cat0[:,0] == '1', 0]), np.mean(LDA0[cat0[:,0] == '2', 0]), np.mean(LDA0[cat0[:,0] == '3', 0])])
        add_row(df, [subjID, allPatterns[p], 'lda', 0, 2, np.mean(LDA0[:,1]), \
                np.mean(LDA0[cat0[:,1] == '1', 1]), np.mean(LDA0[cat0[:,1] == '2', 1]), np.mean(LDA0[cat0[:,1] == '3', 1])])
        add_row(df, [subjID, allPatterns[p], 'lda', 0, 3, np.mean(LDA0[:,2]), \
                np.mean(LDA0[cat0[:,2] == '1', 2]), np.mean(LDA0[cat0[:,2] == '2', 2]), np.mean(LDA0[cat0[:,2] == '3', 2])])
        # cond 1
        add_row(df, [subjID, allPatterns[p], 'lda', 1, 1, np.mean(LDA1[:,0]), \
                np.mean(LDA1[cat1[:,0] == '1', 0]), np.mean(LDA1[cat1[:,0] == '2', 0]), np.mean(LDA1[cat1[:,0] == '3', 0])])
        add_row(df, [subjID, allPatterns[p], 'lda', 1, 2, np.mean(LDA1[:,1]), \
                np.mean(LDA1[cat1[:,1] == '1', 1]), np.mean(LDA1[cat1[:,1] == '2', 1]), np.mean(LDA1[cat1[:,1] == '3', 1])])
        add_row(df, [subjID, allPatterns[p], 'lda', 1, 3, np.mean(LDA1[:,2]), \
                np.mean(LDA1[cat1[:,2] == '1', 2]), np.mean(LDA1[cat1[:,2] == '2', 2]), np.mean(LDA1[cat1[:,2] == '3', 2])])


        ##############################################
        # LR integration
        ##############################################
        pred_LR = pred_LR.reshape(216)
        comp = np.array(np.equal(y.astype(int),pred_LR.astype(int)))
        LR_0 = comp[cond==0]
        LR_1 = comp[cond==1]

        LR0 = LR_0.reshape(36,3)
        LR1 = LR_1.reshape(36,3)

        roitemp[12,0] = str(np.mean(LR0[:,0]))
        roitemp[13,0] = str(np.mean(LR0[:,1]))
        roitemp[14,0] = str(np.mean(LR0[:,2]))
        roitemp[15,0] = str(np.mean(LR1[:,0]))
        roitemp[16,0] = str(np.mean(LR1[:,1]))
        roitemp[17,0] = str(np.mean(LR1[:,2]))

        roitemp[12,1] = str(np.mean(LR0[cat0[:,0] == '1', 0]))
        roitemp[13,1] = str(np.mean(LR0[cat0[:,1] == '1', 1]))
        roitemp[14,1] = str(np.mean(LR0[cat0[:,2] == '1', 2]))
        roitemp[15,1] = str(np.mean(LR1[cat1[:,0] == '1', 0]))
        roitemp[16,1] = str(np.mean(LR1[cat1[:,1] == '1', 1]))
        roitemp[17,1] = str(np.mean(LR1[cat1[:,2] == '1', 2]))

        roitemp[12,2] = str(np.mean(LR0[cat0[:,0] == '2', 0]))
        roitemp[13,2] = str(np.mean(LR0[cat0[:,1] == '2', 1]))
        roitemp[14,2] = str(np.mean(LR0[cat0[:,2] == '2', 2]))
        roitemp[15,2] = str(np.mean(LR1[cat1[:,0] == '2', 0]))
        roitemp[16,2] = str(np.mean(LR1[cat1[:,1] == '2', 1]))
        roitemp[17,2] = str(np.mean(LR1[cat1[:,2] == '2', 2]))

        roitemp[12,3] = str(np.mean(LR0[cat0[:,0] == '3', 0]))
        roitemp[13,3] = str(np.mean(LR0[cat0[:,1] == '3', 1]))
        roitemp[14,3] = str(np.mean(LR0[cat0[:,2] == '3', 2]))
        roitemp[15,3] = str(np.mean(LR1[cat1[:,0] == '3', 0]))
        roitemp[16,3] = str(np.mean(LR1[cat1[:,1] == '3', 1]))
        roitemp[17,3] = str(np.mean(LR1[cat1[:,2] == '3', 2]))

        # currently, hacky+bulky+hardcoded way of adding to df
        # cond 0
        add_row(df, [subjID, allPatterns[p], 'lr', 0, 1, np.mean(LR0[:,0]), \
                np.mean(LR0[cat0[:,0] == '1', 0]), np.mean(LR0[cat0[:,0] == '2', 0]), np.mean(LR0[cat0[:,0] == '3', 0])])
        add_row(df, [subjID, allPatterns[p], 'lr', 0, 2, np.mean(LR0[:,1]), \
                np.mean(LR0[cat0[:,1] == '1', 1]), np.mean(LR0[cat0[:,1] == '2', 1]), np.mean(LR0[cat0[:,1] == '3', 1])])
        add_row(df, [subjID, allPatterns[p], 'lr', 0, 3, np.mean(LR0[:,2]), \
                np.mean(LR0[cat0[:,2] == '1', 2]), np.mean(LR0[cat0[:,2] == '2', 2]), np.mean(LR0[cat0[:,2] == '3', 2])])
        # cond 1
        add_row(df, [subjID, allPatterns[p], 'lr', 1, 1, np.mean(LR1[:,0]), \
                np.mean(LR1[cat1[:,0] == '1', 0]), np.mean(LR1[cat1[:,0] == '2', 0]), np.mean(LR1[cat1[:,0] == '3', 0])])
        add_row(df, [subjID, allPatterns[p], 'lr', 1, 2, np.mean(LR1[:,1]), \
                np.mean(LR1[cat1[:,1] == '1', 1]), np.mean(LR1[cat1[:,1] == '2', 1]), np.mean(LR1[cat1[:,1] == '3', 1])])
        add_row(df, [subjID, allPatterns[p], 'lr', 1, 3, np.mean(LR1[:,2]), \
                np.mean(LR1[cat1[:,2] == '1', 2]), np.mean(LR1[cat1[:,2] == '2', 2]), np.mean(LR1[cat1[:,2] == '3', 2])])

        roiloop.append(roitemp)
        progress += 1  # simply a counter for how many rois have been proccessed

    roiloop = np.asarray(roiloop)

    roi2save = roiloop.reshape((roiloop.shape[0]*roiloop.shape[1]), roiloop.shape[2])
    # print(roi2save)
    print('Completed: ' + str(progress) + '/' + str(len(allPatterns)))
    print('Skipped ' + str(len(skipped)) + ' ROIs')
    df.to_csv(subjDir + '/patterns/' + subjID + 'class.csv', index=False)
    print('successfully saved as ' + subjDir + '/patterns/' + subjID + 'class.csv')
