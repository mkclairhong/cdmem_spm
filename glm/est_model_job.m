%-----------------------------------------------------------------------
% Job saved on 29-Oct-2019 19:26:33 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7487)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------


function matlabbatch = est_model_job(subjID)
%   import events csv and convert to struct
%   creating duration.mat, cond.mat, stimOnset.mat in /events directory
%
%   Parameters
%   ----------
%   subjID : int
%        simply subject ID in integers (ie. 5, 10, 25)
%
%   Returns
%   -------
%   SPM.mat : mat
%        struct created from SPM
%   beta images
%        beta images created in /images/beta_study/


% function variables (Do not change unless you know what you are doing)
%---------------------------------------------------------------------%
baseDir = '/scratch/polynlab/fmri/cdcatmr/';  % scratch data dir
scanDir = '/scratch/polynlab/fmri/cdcatmr/all_subjects/scanN.mat';
spm('defaults','fmri');
spm_jobman('initcfg');
nScans  = 187;
nTrials = 8;

%---------------------------------------------------------------------%
number_tag = num2str(subjID, '%03.f');
subject = ['cdcatmr', number_tag];
subjDir = [baseDir, subject];


% read in stimuli onset
load(scanDir);
disp('scans loaded')

load([subjDir, '/events/stimOnset.mat']);
disp('stim loaded')

matlabbatch{1}.spm.stats.fmri_spec.dir = {[subjDir, '/images/beta_study']};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'scans';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

disp('stage 1 loaded')

for t = 1:nTrials
    
    for s = 1:nScans
        matlabbatch{1}.spm.stats.fmri_spec.sess(t).scans{s,1} = ([subjDir, '/images/func/func', num2str(t), '/wrfunc', num2str(t), '_', num2str(scanN{s}), '.nii']);

        matlabbatch{1}.spm.stats.fmri_spec.sess(t).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});

%       disp(stimOnset(t,:))  
        %matlabbatch{1}.spm.stats.fmri_spec.sess(t).cond.onset = scanOnset(t,:);
    end
    %%
%     matlabbatch{1}.spm.stats.fmri_spec.sess(t).cond.duration = 0;
%     matlabbatch{1}.spm.stats.fmri_spec.sess(t).cond.tmod = 0;
%     matlabbatch{1}.spm.stats.fmri_spec.sess(t).cond.pmod = struct('name', {}, 'param', {}, 'poly', {});
%     matlabbatch{1}.spm.stats.fmri_spec.sess(t).cond.orth = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(t).multi = {[subjDir, '/events/glm/run', num2str(t), '.mat']};
    
    R = textread([subjDir, '/images/func/func', num2str(t), '/rp_func', num2str(t), '_00001.txt']);
                                                        
    matlabbatch{1}.spm.stats.fmri_spec.sess(t).regress = struct('name', {'x' 'y1' 'z' 'r' 'p' 'y2'}, 'val', {R(:,1) R(:,2) R(:,3) R(:,4) R(:,5) R(:,6)});
    matlabbatch{1}.spm.stats.fmri_spec.sess(t).multi_reg = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess(t).hpf = 128;
    
end

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';

matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = {[subjDir, '/images/beta_study/SPM.mat']};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

disp("Let's run spm_jobman")
spm_jobman('run', matlabbatch);

disp('All Done')

end
