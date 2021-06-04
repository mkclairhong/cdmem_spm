from scipy import stats
from mi_accre import avg_mi_coreg, avg_mi_type
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')
sns.set_color_codes('pastel')


def mi_plot(subjID, type='rw', nScans=187, nRuns=8, resDir='/scratch/polynlab/fmri/cdcatmr/all_subjects/mutual_info', verbose=True):
    """
    uses the mi_accre functions to calculate the average mutual information, and shows box plot and distribution plot for each subject

    Parameters
    ----------
    subjID : str
        string of subject data (ie. 'cdcatmr011')
    type : str ['rw', 'coreg'] (default: 'rw')
        depending on the type, it will know which QA comparision is being done
        'rw' will compare T1 scan to resliced images, and weighted T1 scan to weighted images
        'coreg' will compare the coregisteration methods: register to first vs. register to mean methods
    verbose : boolean (default: False)
        if True, will print out which scan file is being worked on
    nScans : int (default: 187)
        integer of how many scans to expect per each run
    nRuns : int (default: 8)
        integer of how many runs to expect for a given subject
    resDir : str/abs path (default is set to scratch project dir)
        string of absolute path to save the resulted plot summary

    Returns
    -------
    plot : eps
        creates a combined graphic of box plot and distribution plot summarizing correlation coefficient and mutual info

    """
    print('running type: ' + str(type))
    subjID = str(subjID)
    subjID = 'cdcatmr' + subjID.zfill(3)
    if type == 'rw':
        cc1, mi1 = avg_mi_type(subjID, 'r', resDir=resDir, saveText=True, verbose=verbose)  # source image is rfunc
        cc2, mi2 = avg_mi_type(subjID, 'w', resDir=resDir, saveText=True, verbose=verbose)  # source image is wrfunc
        label1, label2 = 'rfunc', 'wrfunc'

    if type == 'coreg':
        cc1, mi1 = avg_mi_coreg(subjID, 'rtf', resDir=resDir, verbose=verbose)  # register to first scan
        cc2, mi2 = avg_mi_coreg(subjID, 'rtm', resDir=resDir, verbose=verbose)  # register to mean
        label1, label2 = 'rtf', 'rtm'

    # assert len(cc1) == len(cc2) == nScans * nRuns
    # assert len(mi1) == len(mi2) == nScans * nRuns

    cc_t, cc_p = stats.ttest_ind(cc1, cc2)
    mi_t, mi_p = stats.ttest_ind(mi1, mi2)

    # let's plot box-dist plot combined
    f, (ax_box1, ax_box2, ax_dist) = plt.subplots(3, sharex=True, gridspec_kw= {"height_ratios": (0.3, 0.3, 1)})

    cc1_mean = np.mean(cc1)
    cc2_mean = np.mean(cc2)

    sns.boxplot(cc1, ax=ax_box1, color='b')
    sns.boxplot(cc2, ax=ax_box2, color='r')
    ax_box1.axvline(cc1_mean, color='g', linestyle='--')
    ax_box2.axvline(cc2_mean, color='m', linestyle='--')
    plt.subplots_adjust(top=0.87)
    plt.suptitle('correlation coefficient n=' + str(len(cc1)), fontsize = 16)

    sns.distplot(cc1, ax=ax_dist, label=label1, color='b', norm_hist=True)
    sns.distplot(cc2, ax=ax_dist, label=label2, color='r', norm_hist=True)
    ax_dist.axvline(cc1_mean, color='g', linestyle='--')
    ax_dist.axvline(cc2_mean, color='m', linestyle='--')
    plt.legend()
    ax_box1.set(xlabel='')
    ax_box2.set(xlabel='')
    plt.savefig(resDir + '/' + subjID + '_coef_boxdist_plot.eps')

    # let's plot box-dist plot combined
    f, (ax_box1, ax_box2, ax_dist) = plt.subplots(3, sharex=True, gridspec_kw= {"height_ratios": (0.3, 0.3, 1)})

    mi1_mean = np.mean(mi1)
    mi2_mean = np.mean(mi2)

    sns.boxplot(mi1, ax=ax_box1, color='b')
    sns.boxplot(mi2, ax=ax_box2, color='r')
    ax_box1.axvline(mi1_mean, color='g', linestyle='--')
    ax_box2.axvline(mi2_mean, color='m', linestyle='--')
    plt.subplots_adjust(top=0.87)
    plt.suptitle('mutual information n=' + str(len(cc1)), fontsize = 16)

    sns.distplot(mi1, ax=ax_dist, label=label1, color='b', norm_hist=True)
    sns.distplot(mi2, ax=ax_dist, label=label2, color='r', norm_hist=True)
    ax_dist.axvline(mi1_mean, color='g', linestyle='--')
    ax_dist.axvline(mi2_mean, color='m', linestyle='--')
    plt.legend()
    ax_box1.set(xlabel='')
    ax_box2.set(xlabel='')
    plt.savefig(resDir + '/'+ subjID + '_mutualinfo_boxdist_plot.eps')


# mi_plot(5, verbose=True)
