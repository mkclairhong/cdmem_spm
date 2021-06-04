function split_imgs_anat(subjID)
%   locates, moves and renames the anatomical scan files
%   see function variables below to refer to save out name
%
%   Parameters
%   ----------
%   subjID : int
%        simply subject ID in integers (ie. 5, 10, 25)
%
%   Returns
%   -------
%   anatomical scan files allocated to images/anat directory


% function variables (Do not change unless you know what you are doing)
%---------------------------------------------------------------------%
baseDir = '/scratch/polynlab/fmri/cdcatmr/';  % scratch data dir
t1_new_name = 'T1_anat.nii';
t2_new_name = 'T2_anat.nii';
b0e1_new_name = 'B0map_e1.nii';
b0e2_new_name = 'B0map_e2.nii';
blip_new_name = 'blip_anat.nii';
%---------------------------------------------------------------------%

number_tag = num2str(subjID, '%03.f');
% set path
subjDir = [baseDir, 'cdcatmr', number_tag];
imgDir = [subjDir, '/images'];
anatDir = [imgDir, '/anat/'];
rawDir = [imgDir, '/raw'];

cd(rawDir);

% let's grab scan files, copy and rename into /images/anat directory
% for T1 scan
T1 = dir('*T1W*.01.nii');
copyfile(T1.name, [anatDir, t1_new_name]);
    
% for T2 scan
T2 = dir('*T2W*.01.nii');
% check if subject has T2 scan, since not all subjects have T2
if ~isempty(T2)
    copyfile(T2.name, [anatDir, t2_new_name]);
end
    
% for B0 fieldmap
B0e1 = dir('*B0map.01_e1.nii*');
if ~isempty(B0e1)
    copyfile(B0e1.name, [anatDir, b0e1_new_name]);
        
    B0e2 = dir('*B0map.01_e2*.nii');
    copyfile(B0e2.name, [anatDir, b0e2_new_name]);
end
    
blip = dir('*fatshift.01.nii*');
if ~isempty(blip)
    copyfile(blip.name, [anatDir, blip_new_name]);
end
    
end

