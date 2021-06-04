function pats = beta2pat_loop(subjID)
%% for a given subject, loops all the mask files in a given directory
addpath(genpath('/home/jeonj1/matlab/spm12'))
% Set path and some variables
baseDir = '/scratch/polynlab/fmri/cdcatmr/';  % scratch data dir
mask_dir = '/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/resliced';

number_tag = num2str(subjID, '%03.f');
subject = ['cdcatmr', number_tag];
subjDir = [baseDir, subject];
betaDir = [subjDir, '/images/beta_study'];
saveDir = [subjDir, '/patterns/'];

% misc setting for cdcatmr-specific (27 stim in a trial + xyzryw)
% default beta images are named beta_0001.nii, 0002, 0272, and pos variable is the numbers we care about
pos = nan(1,216);
pos = [1:27,34:60,67:93,100:126,133:159,166:192,199:225,232:258];

%---------------------------------------------------------------------%
load([betaDir, '/SPM.mat'])

mask_dir = dir(mask_dir);
for j=1:length(mask_dir)
  if ~mask_dir(j).isdir && strcmp(mask_dir(j).name(end-3:end), '.nii')
    roi_mask = [mask_dir(j).folder '/' mask_dir(j).name];
    [x1, roi_name, x3] = fileparts(roi_mask);
    saveName = [subject, '_pat_', roi_name, '.mat'];
    
    cd(betaDir)
    st_vols = SPM.Vbeta(pos);
    roi_vol = spm_vol(roi_mask);
    
    % bit of hacking to match match the renamed beta images to original SPM version names ie. beta_000N.nii
    beta_renames = {};
    for n = 1:length(st_vols)
        % if original name isn't changed, do nothing
        if exist(st_vols(n).fname)
            % do nothing
        else
            % st_vol(n).descrip is string of info ie. spm_spm:beta (0036) - Sn(2) Florence*bf(1)
            % so we just extract 0036 and 2, which denote the original beta and trial numbers respectively
            temp_digits = str2double(regexp(st_vols(n).descrip, '\d*','Match'));
            beta_num = temp_digits(1);
            trial_num = temp_digits(2);

            item_index = temp_digits(1);
            if beta_num > 27
                item_index = mod(beta_num, (27 * (trial_num - 1) + 6 * (trial_num - 1)));
            end
            temp_name = strcat(string(trial_num), '_', string(item_index), '_');
            file_name = dir([char(temp_name), '*']);

            % sanity check that items match
            nii_name = strsplit(file_name.name(1:end-4), '_');  % get rid of .nii at the end
            spm_name = strsplit(st_vols(n).descrip, ' ');
            spm_name = spm_name(5:end);
            temp = char(spm_name(end));
            temp = temp(1:end-6);
            spm_name(end) = cellstr(temp);
            nii_name = nii_name(3:end);

            if ~ismember(nii_name, spm_name)  % if all items are different and no strings match...
                warning('renamed beta might not match with SPM.fname at index %s: %s1 != %s2', string(n), file_name.name, st_vols(n).descrip)
            end
            beta_renamed = true;
            beta_renames = [beta_renames, file_name.name];
        end
    end

    % add a temporary new struct field of renamed beta names
    if beta_renamed
        for j = 1:length(st_vols)
            st_vols(j).fname = char(beta_renames(j));
        end
    end

    if ~all(roi_vol.dim==st_vols(1).dim)
        flags.mean = false;
        flags.which = 1;
        spm_reslice({st_vols(1).fname, roi_mask}, flags);
        % if ~beta_renamed
        %     spm_reslice({st_vols(1).fname, roi_mask}, flags);
        % else
        %     spm_reslice({st_vols(1).rename, roi_mask}, flags);
        % end
    end


    % update resliced
    [pathstr,name,ext] = fileparts(roi_mask);

    roi_vol = spm_vol(fullfile(pathstr, [name ext]));
    % roi_vol = spm_vol(fullfile(pathstr, ['r' name ext]));
    mask = spm_read_vols(roi_vol);

    % unravel
    mask = reshape(mask, size(mask, 1), []);

    % initialize n-trials by m-voxels array
    temp_pat = single( nan([length(st_vols) st_vols(1).dim]));

    temp_pat(1,:,:,:) = spm_read_vols(st_vols(1));
    temp_pat = reshape(temp_pat, size(temp_pat, 1), []);
    pats = temp_pat(:,mask==1);


    % grab one beta at a time - into pat
    for i = 1:length(st_vols)
        temp_beta = spm_read_vols(st_vols(i));
        new_pat = reshape(temp_beta,size(temp_beta,1),[]);
        pats(i,:) = new_pat(mask==1);
    end

    newPats = pats(:,all(~isnan(pats)));

    % create images/pattern directory
    if ~isdir(saveDir)
        mkdir(saveDir)
    end

    save([saveDir, saveName], 'pats')
    end
  end
end