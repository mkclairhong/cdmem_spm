function AnaDef = cat_glm_batch(subjID)

subj = num2str(subjID, '%03.f');
% betaseries analysis definition for BASCO

% subj = '010';

%load(['/Users/beckycutler/data/fmri/cdcatmr/cdcatmr', subj, '/SPM.mat'])


n_items     = 27;
n_cat       = 3;

% make this directory
data_dir = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/images/func'];
% if ~isdir([data_dir, '/cat_betaseries'])
%     mkdir([data_dir, '/cat_betaseries'])
% end

basco_path = fileparts(which('BASCO'));
AnaDef.Img                  = 'nii';
AnaDef.Img4D                = false;      % true: 4D Nifti
AnaDef.NumCond              = n_cat;    % number of conditions
AnaDef.Cond                 = {'cat1','cat2','cat3'};      % names of conditions
AnaDef.units                = 'scans';    % unit 'scans' or 'secs'
AnaDef.RT                   = 2;          % repetition time in seconds
AnaDef.fmri_t               = 16;
AnaDef.fmri_t0              = 9;
AnaDef.OutDir               = 'cat_betaseries';  % output directory
AnaDef.Prefix               = 'wrfunc';
AnaDef.OnsetModifier        = 0; % subtract this number from the onset-matrix (unit: scans)  <===== 4 files deleted and starting at 0 !!!

AnaDef.VoxelAnalysis        = true;  
AnaDef.ROIAnalysis          = true; % ROI level analysis (estimate model on ROIs for network analysis)
AnaDef.ROIDir               = fullfile(basco_path,'rois','AALROI90'); % AALROI90 AAL3; select all ROIs in this directory
AnaDef.ROIPrefix            = 'MNI_'; % 'MNI_' 'r'
AnaDef.ROINames             = fullfile(basco_path,'rois','AALROI90','AALROINAMES.txt'); % AALROINAMES.txt AAL3.nii.txt; txt.-file containing ROI names
AnaDef.ROISummaryFunction   = 'mean'; % 'mean' or 'median'

AnaDef.HRFDERIVS            = [0 0];  % temporal and disperion derivatives: [0 0] or [1 0] or [1 1]

% regressors to include into design
AnaDef.MotionReg            = true;
AnaDef.GlobalMeanReg        = false;

% name of output-file (analysis objects)
AnaDef.Outfile              = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/model/out_cat_est.mat'];
cSubj = 0; % subject counter

vp = {['cdcatmr', subj]};

% directory containing all subject folders
data_dir = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/images/func'];

% all subjects
for i=1:length(vp)
    cSubj = cSubj+1;
    AnaDef.Subj{cSubj}.DataPath = fullfile(data_dir); 
    AnaDef.Subj{cSubj}.NumRuns  = 8;
    AnaDef.Subj{cSubj}.RunDirs  = {'func1','func2','func3','func4','func5','func6','func7','func8'};
    AnaDef.Subj{cSubj}.Onsets   = repmat({'cat_onsets.csv'}, 1, 8);
    AnaDef.Subj{cSubj}.Duration = repmat(2.5,1,n_items);
end

%
AnaDef.NumSubjects = cSubj;

end