function preproc_cdcatmr(subjID, register_type)
%   locates, moves and renames the anatomical scan files
%   see function variables below to refer to save out name
%
%   Parameters
%   ----------
%   subjID : int
%        simply subject ID in integers (ie. 5, 10, 25)
%
%   register_type: int {0, 1} or string {'rtf', 'rtm'}
%        specifies registration to first scan image or mean
%        0 or 'rtm' means register to first scan image 
%        1 or 'rtf' will register to mean image
%
%   Returns
%   -------
%   anatomical scan files allocated to images/anat directory

% function variables (Do not change unless you know what you are doing)
%---------------------------------------------------------------------%
baseDir = '/scratch/polynlab/fmri/cdcatmr/';  % scratch data dir
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);
tpmDir = [fileparts(which('spm')), '/tpm/TPM.nii'];
%---------------------------------------------------------------------%

number_tag = num2str(subjID, '%03.f');
subject = ['cdcatmr', number_tag];
subjDir = [baseDir, subject];

% set default registration type to 'register to first image'
if nargin < 2
    register_type = 0
end

if register_type == 'rtm' | register_type == 1
    register_type = 1
elseif register_type == 'rtf' | register_type == 0
    register_type = 0
else
    register_type = 0
end


cnt = 1;

% by trial loop, initialize file variables
for r = 1:8
    fdir = [subjDir '/images/func/func' num2str(r)];
    func_imgs{cnt} = get_files(fdir, 'func', 'nii')';
    realign_imgs{cnt} = get_files(fdir, 'func', 'nii','r')';
    norm_imgs{cnt} = get_files(fdir, 'func', 'nii','wr')';
    cnt = cnt + 1;
end

% combine all images into one by vertically concatenating
all_imgs = {}; all_realign={}; all_norm={};
for k = 1:length(func_imgs)
    all_imgs = vertcat(all_imgs, func_imgs{k});
    all_realign = vertcat(all_realign, realign_imgs{k});
    all_norm = vertcat(all_norm, norm_imgs{k});
end


'start preproc'
% realign imgs within subject
if register_type == 0
    matlabbatch{1}.spm.spatial.realign.estimate.data = func_imgs;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.quality = 0.9;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.sep = 4;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.fwhm = 5;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.rtm = 0;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.interp = 2;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.weight = '';
elseif register_type == 1
    matlabbatch{1}.spm.spatial.realign.estwrite.data = func_imgs;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;  % rtm- realign to mean// 0 was default- realign to first run instead of mean
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';
end

% create tissue prob. maps (separate grey,white and CSF)
% check name of anat
matlabbatch{2}.spm.spatial.preproc.channel.vols = {[subjDir '/images/anat/T1_anat.nii']};
%matlabbatch{2}.spm.spatial.preproc.channel.vols = {[subjDir '/images/anat/T1_anat.nii']};
matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{2}.spm.spatial.preproc.channel.write = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {[tpmDir, ',1']};

matlabbatch{2}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {[tpmDir, ',2']};

matlabbatch{2}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {[tpmDir, ',3']};

matlabbatch{2}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {[tpmDir, ',4']};

matlabbatch{2}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{2}.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {[tpmDir, ',5']};

matlabbatch{2}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{2}.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {[tpmDir, ',6']};

matlabbatch{2}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{2}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{2}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{2}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{2}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{2}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{2}.spm.spatial.preproc.warp.write = [1 1];

matlabbatch{3}.spm.util.imcalc.input = {
                                        [subjDir '/images/anat/T1_anat.nii']                  %% t1
                                        [subjDir '/images/anat/c1T1_anat.nii']                %% ct1
                                        };
matlabbatch{3}.spm.util.imcalc.output = 't1_mask_g';
matlabbatch{3}.spm.util.imcalc.outdir = {[subjDir '/images/anat']};
matlabbatch{3}.spm.util.imcalc.expression = 'i1.*((i2)>.5)'; %'i1.*((i2+i3+i4)>.5)';
matlabbatch{3}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{3}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{3}.spm.util.imcalc.options.mask = 0;
matlabbatch{3}.spm.util.imcalc.options.interp = 1;
matlabbatch{3}.spm.util.imcalc.options.dtype = 4;

% matlabbatch{4}.spm.util.imcalc.input = {
%                                         [subjDir '/images/anat/struct_t1-0001.nii']
%                                         [subjDir '/images/anat/c1struct_t1-0001.nii']
%                                         [subjDir '/images/anat/c2struct_t1-0001.nii']
%                                         };
% matlabbatch{4}.spm.util.imcalc.output = 't1_mask_gw';
% matlabbatch{4}.spm.util.imcalc.outdir = {[subjDir '/images/anat/struct_t1/']};
% matlabbatch{4}.spm.util.imcalc.expression = 'i1.*((i2+i3)>.5)'; %'i1.*((i2+i3+i4)>.5)';
% matlabbatch{4}.spm.util.imcalc.var = struct('name', {}, 'value', {});
% matlabbatch{4}.spm.util.imcalc.options.dmtx = 0;
% matlabbatch{4}.spm.util.imcalc.options.mask = 0;
% matlabbatch{4}.spm.util.imcalc.options.interp = 1;
% matlabbatch{4}.spm.util.imcalc.options.dtype = 4;
%
% matlabbatch{5}.spm.util.imcalc.input = {
%                                         [subjDir '/images/struct_t1/struct_t1-0001.nii']
%                                         [subjDir '/session_0/images/struct_t1/c1struct_t1-0001.nii']
%                                         [subjDir '/session_0/images/struct_t1/c2struct_t1-0001.nii']
%                                         [subjDir '/images/struct_t1/c3struct_t1-0001.nii']
%                                         };
% matlabbatch{5}.spm.util.imcalc.output = 't1_mask_gwc';
% matlabbatch{5}.spm.util.imcalc.outdir = {[subjDir '/session_0/images/struct_t1/']};
% matlabbatch{5}.spm.util.imcalc.expression = 'i1.*((i2+i3+i4)>.5)'; %'i1.*((i2+i3+i4)>.5)';
% matlabbatch{5}.spm.util.imcalc.var = struct('name', {}, 'value', {});
% matlabbatch{5}.spm.util.imcalc.options.dmtx = 0;
% matlabbatch{5}.spm.util.imcalc.options.mask = 0;
% matlabbatch{5}.spm.util.imcalc.options.interp = 1;
% matlabbatch{5}.spm.util.imcalc.options.dtype = 4;

anat_struct = [subjDir '/images/anat/T1_anat.nii'];
anat_ref_g = [subjDir '/images/anat/t1_mask_g.nii'];

% anat_ref_gw = [subjDir '/session_0/images/struct_t1/t1_mask_gw.nii'];
% anat_ref_gwc = [subjDir '/session_0/images/struct_t1/t1_mask_gwc.nii'];

matlabbatch{6}.spm.spatial.coreg.estwrite.ref = {anat_ref_g};

if register_type == 0
    matlabbatch{6}.spm.spatial.coreg.estwrite.source = func_imgs{1}(1); % first func
elseif register_type == 1
    matlabbatch{6}.spm.spatial.coreg.estwrite.source = cellstr(strcat(fileparts(string(func_imgs{1}(1))), '/meanfunc1_00001.nii'));
end

matlabbatch{6}.spm.spatial.coreg.estwrite.other = [all_imgs(2:end)];
matlabbatch{6}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{6}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{6}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{6}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{6}.spm.spatial.coreg.estwrite.roptions.interp = 4;
matlabbatch{6}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{6}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{6}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

% coreg: register all imgs to deformation field
% anat_struct = [subjDir '/session_0/images/struct_t1/struct_t1-0001.nii'];
% anat_ref_g = [subjDir '/session_0/images/struct_t1/t1_mask_g.nii'];
% anat_ref_gw = [subjDir '/session_0/images/struct_t1/rt1_mask_gw.nii'];
% anat_ref_gwc = [subjDir '/session_0/images/struct_t1/rt1_mask_gwc.nii'];
matlabbatch{7}.spm.util.defs.comp{1}.idbbvox.vox = [2 2 2];
matlabbatch{7}.spm.util.defs.comp{1}.idbbvox.bb = [NaN NaN NaN
                                                   NaN NaN NaN];
matlabbatch{7}.spm.util.defs.comp{2}.def(1) = {[subjDir '/images/anat/y_T1_anat.nii']};
%matlabbatch{7}.spm.util.defs.out{1}.pull.fnames = [all_realign; {anat_struct}; {anat_ref_g}; {anat_ref_gw}; {anat_ref_gwc}];
matlabbatch{7}.spm.util.defs.out{1}.pull.fnames = [all_realign; {anat_struct}; {anat_ref_g}];
matlabbatch{7}.spm.util.defs.out{1}.pull.savedir.savesrc = 1;
matlabbatch{7}.spm.util.defs.out{1}.pull.interp = 4;
matlabbatch{7}.spm.util.defs.out{1}.pull.mask = 1;
matlabbatch{7}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
matlabbatch{7}.spm.util.defs.out{1}.pull.prefix = 'w';

% smooth all imgs w/ 4fwhm filter
% anat_struct = [subjDir '/session_0/images/struct_t1/wstruct_t1-0001.nii'];
% anat_ref_g = [subjDir '/session_0/images/struct_t1/wt1_mask_g.nii'];
% anat_ref_gw = [subjDir '/session_0/images/struct_t1/wt1_mask_gw.nii'];
% anat_ref_gwc = [subjDir '/session_0/images/struct_t1/wt1_mask_gwc.nii'];
%matlabbatch{8}.spm.spatial.smooth.data = [all_norm; {anat_struct}; {anat_ref_g}; {anat_ref_gw}; {anat_ref_gwc}];
%matlabbatch{8}.spm.spatial.smooth.data = [all_norm; {anat_struct}; {anat_ref_g}];
%matlabbatch{8}.spm.spatial.smooth.fwhm = [0 0 0];
%matlabbatch{8}.spm.spatial.smooth.dtype = 0;
%matlabbatch{8}.spm.spatial.smooth.im = 0;
%matlabbatch{8}.spm.spatial.smooth.prefix = 's';

% hacky way to remove empty cells due to removing a step above
% e.g. I removed steps 4 and 5 (additional tissue maps)

matlabbatch(4:5) = [];

spm_jobman('run',matlabbatch);

end
