function pats = beta2pat(SPM,subj,roi_mask,pos)

% read mask - check dim - resample

cd(['/Users/beckycutler/data/fmri/cdcatmr/cdcatmr' num2str(subj) '/images/beta_study'])

st_vols = SPM.Vbeta(pos);

roi_vol = spm_vol(roi_mask);

if ~all(roi_vol.dim==st_vols(1).dim)
    flags.mean = false;
    flags.which = 1;
    spm_reslice({st_vols(1).fname,roi_mask},flags);
end

% update resliced
[pathstr,name,ext] = fileparts(roi_mask);

roi_vol = spm_vol(fullfile(pathstr,['r' name ext]));
mask = spm_read_vols(roi_vol);

% unravel
mask = reshape(mask,size(mask,1),[]);

% initialize n-trials by m-voxels array
temp_pat = single(nan([length(st_vols) st_vols(1).dim]));
temp_pat(1,:,:,:) = spm_read_vols(st_vols(1));
temp_pat = reshape(temp_pat,size(temp_pat,1),[]);
pats = temp_pat(:,mask==1);

% grab one beta at a time - into pat
for i = 1:length(st_vols)
    temp_beta = spm_read_vols(st_vols(i));
    new_pat = reshape(temp_beta,size(temp_beta,1),[]);
    pats(i,:) = new_pat(mask==1);
end

end
