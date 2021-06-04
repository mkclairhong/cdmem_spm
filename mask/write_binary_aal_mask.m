function vol = write_binary_aal_mask(aal_path, roi_num, out_path)
%WRITE_BINARY_AAL_MASK writes a binary aal mask
%   Inputs:     aal_path - string specifying location of aal.nii
%
%               roi_num - value of the ROI(s) of interest, can be a scalar
%               or vector (see aal.nii.txt for a full list of values)
%
%               out_path - string specifying output
%
%   Outputs:    vol - SPM volume info for the created mask
%
% Example: 
%               aal_path = 'aal.nii'; %on the MATLAB path
%               roi_num = [37 38]; %bilateral hippocampus
%               out_path = 'hippocampus_mask.nii';
%               write_binary_aal_mask(aal_path,roi_num,out_path);
%


aal_vol = spm_vol(aal_path);
aal_data = spm_read_vols(aal_vol);

mask = ismember(aal_data,roi_num);

out_vol = aal_vol;
out_vol.fname = out_path;
vol = spm_write_vol(out_vol,mask);


end

