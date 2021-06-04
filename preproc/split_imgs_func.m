function split_imgs_func(subjID)
%   function to split 4D .nii files into 3D by unpacking into one image per TR
%
%   Parameters
%   ----------
%   subjID : int
%        simply subject ID in integers (ie. 5, 10, 25)
%
%   Returns
%   -------
%   functional scan files allocated to func/funcN directory


% function variables (Do not change unless you know what you are doing)
%---------------------------------------------------------------------%
baseDir = '/scratch/polynlab/fmri/cdcatmr/';  % scratch data dir
n_trials = 8;

%---------------------------------------------------------------------%

number_tag = num2str(subjID, '%03.f');
% set path
subjDir = [baseDir, 'cdcatmr', number_tag];
imgDir = [subjDir, '/images'];
rawDir = [imgDir, '/raw'];


cd(rawDir)

% grab all functional scans
raw = dir('*CDCATMR*.01.nii');

% trial loop
for t = 1:n_trials
        
        cur_file = [rawDir, '/', raw(t).name];
        
        fname = ['func' num2str(t) '.nii'];
        
        % copy and rename the functional scans
        % creates func1.nii, func2.nii, ..., func8.nii
        copyfile(string(cur_file), string(fname));
        
        % check if funcN folders exist in images/func folder
        funcDir = [imgDir, '/func'];
        if ~isdir([funcDir, '/func', num2str(t)])
            mkdir([funcDir, '/func', num2str(t)]);
        end
        funcDir = [funcDir, '/func', num2str(t)];
        
        % add files to images/func/funcN folder
        func_file = [rawDir, '/func', num2str(t), '.nii'];
        
        spm_file_split(func_file, funcDir);
end
