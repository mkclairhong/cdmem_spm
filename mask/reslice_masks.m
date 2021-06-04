%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% script to reslice AAL3 masks to match dims of
% functional [121 x 145 x 121]
% reslice mask [91 x 109 x 91] -> [121 x 145 x 121]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('/home/jeonj1/matlab/spm12'));


% variables: 
% directory to all the raw masks are
mask_dir = '/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/raw_masks';
% make sure the funcPath file exists
funcPath = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr006/images/func/func1/wrfunc1_00001.nii'];
% sanity check to make sure funcPath file has the expected dimension
target_dim = [121 145 121];  


% first read in expected file
func = spm_vol(funcPath);
assert(all(func.dim == target_dim));
mask_dir = dir(mask_dir);
for j=1:length(mask_dir)
    % if the file name ends with .nii
    if ~mask_dir(j).isdir && strcmp(mask_dir(j).name(end-3:end), '.nii')
        roi_mask = [mask_dir(j).folder '/' mask_dir(j).name];
        
        % read volumes of mask
        roi_vol = spm_vol(roi_mask);
              
        % rigid body reslicing mask to first image
        if ~all(roi_vol.dim == func.dim)
            flags.mean = false;
            flags.which = 1;
            spm_reslice({func.fname,roi_mask},flags);
        end
        
        % update resliced
        [pathstr,name,ext] = fileparts(roi_mask);

        % add -r to nifti resliced image
        roi_vol = spm_vol(fullfile(pathstr, ['r' name ext]));
        assert(all(roi_vol.dim==target_dim));
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment below to reslice manually
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% roi_mask = '/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/VT.nii';
% roi_vol = spm_vol(roi_mask);
% funcPath = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr006/images/func/func1/wrfunc1_00001.nii'];
% func = spm_vol(funcPath)
% if ~all(roi_vol.dim == func.dim)
%     flags.mean = false;
%     flags.which = 1;
%     spm_reslice({func.fname,roi_mask},flags);
% end
% [pathstr,name,ext] = fileparts(roi_mask);
% roi_vol = spm_vol(fullfile(pathstr,['r' name ext]));
