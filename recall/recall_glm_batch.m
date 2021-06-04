function AnaDef = recall_glm_batch(subjID)

subj = num2str(subjID, '%03.f');
% subj = '004';

%load(['/Users/polynlab/data/fmri/cdcatmr/cdcatmr', subj, '/events/stim_names.mat']);


n_trials    = 8;
n_items     = 27;
n_cat       = 3;
disp(n_trials)
for n = 1:n_items
    names{n,1} = ['item_', num2str(n), '_'];
end


basco_path = fileparts(which('BASCO'));
AnaDef.Img                  = 'nii';
AnaDef.Img4D                = false;      % true: 4D Nifti
AnaDef.NumCond              = n_items;    % number of conditions
AnaDef.Cond                 = names; % names of conditions
AnaDef.units                = 'scans';    % unit 'scans' or 'secs'
AnaDef.RT                   = 2;          % repetition time in seconds
AnaDef.fmri_t               = 16;
AnaDef.fmri_t0              = 9;
AnaDef.OutDir               = 'recall_beta';  % output directory
AnaDef.Prefix               = 'wrfunc';
AnaDef.OnsetModifier        = 0; % subtract this number from the onset-matrix (unit: scans)  <===== 4 files deleted and starting at 0 !!!

AnaDef.VoxelAnalysis        = true;  
AnaDef.ROIAnalysis          = true; % ROI level analysis (estimate model on ROIs for network analysis)
AnaDef.ROIDir               = fullfile(basco_path,'rois','AALROI90'); % select all ROIs in this directory
AnaDef.ROIPrefix            = 'MNI_';
AnaDef.ROINames             = fullfile(basco_path,'rois','AALROI90','AALROINAMES.txt'); % txt.-file containing ROI names
AnaDef.ROISummaryFunction   = 'mean'; % 'mean' or 'median'

AnaDef.HRFDERIVS            = [0 0];  % temporal and disperion derivatives: [0 0] or [1 0] or [1 1]

% regressors to include into design
AnaDef.MotionReg            = true;
AnaDef.GlobalMeanReg        = false;

% name of output-file (analysis objects)
AnaDef.Outfile              = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/model/recall_single_est.mat'];

cSubj = 0; % subject counter

vp = {['cdcatmr', subj]};

data_dir = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/images/func']; % directory containing all subject folders

% all subjects
for i=1:length(vp)
    cSubj = cSubj+1;
    AnaDef.Subj{cSubj}.DataPath = fullfile(data_dir); 
    AnaDef.Subj{cSubj}.NumRuns  = 8;
    AnaDef.Subj{cSubj}.RunDirs  = {'func1','func2','func3','func4','func5','func6','func7','func8'};
    AnaDef.Subj{cSubj}.Onsets   = {'rec_onsets_1.csv','rec_onsets_2.csv','rec_onsets_3.csv','rec_onsets_4.csv','rec_onsets_5.csv','rec_onsets_6.csv','rec_onsets_7.csv','rec_onsets_8.csv',};
    AnaDef.Subj{cSubj}.Duration = repmat(2.5,1,27);
end

%
AnaDef.NumSubjects = cSubj;
end