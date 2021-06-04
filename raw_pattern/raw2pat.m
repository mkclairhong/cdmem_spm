function raw2pat(subjID)

addpath(genpath('/home/jeonj1/matlab/spm12'));
% script to grab stimuli onset scans from raw images

baseDir = '/scratch/polynlab/fmri/cdcatmr/';
number_tag = num2str(subjID, '%03.f');
subject = ['cdcatmr', number_tag];
subjDir = [baseDir, subject];
resDir = [subjDir, '/patterns/raw/'];
maskDir = '/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/resliced';


load([subjDir, '/images/beta_study/SPM.mat']);
scans = [];
for s = 1:8
    trialS = str2double({SPM.Sess(s).U.ons});
    scans = [scans; trialS];
end

% shift by 2 TRs (4 secs) for the HRF
scans = scans + 2;

%scan count
cnt = 1;
for r = 1:8
    for s = scans(r,:)
        scanN = ['wrfunc', num2str(r), '_', num2str(s, '%05.f'), '.nii'];
        fileNames{cnt,1} = [subjDir, '/images/func/func', num2str(r), '/', scanN];
        cnt = cnt + 1;
    end
end
% fileNames
    
% save out a text file to list which wrfunc images are used for raw patterns
if ~isdir(resDir)
    mkdir(resDir)
end
fileID = fopen([resDir, 'wrfuncs.txt'],'w');
for j = 1:length(fileNames)
    fprintf(fileID, string(fileNames(j)));
    fprintf(fileID, '\n');
end
fclose(fileID);

% get masks
maskDir = dir(maskDir);
for j=1:length(maskDir)
    if ~maskDir(j).isdir && strcmp(maskDir(j).name(end-3:end), '.nii')
        roi_mask = [maskDir(j).folder '/' maskDir(j).name];
        [x1, roi_name, x3] = fileparts(roi_mask);
        saveName = [subject, '_RAWpat_', roi_name, '.txt'];
        
        mask = niftiread(roi_mask);
        
        % unravel
        mask = reshape(mask,size(mask,1),[]);
        
        pats = [];
        % grab one scan at a time - into pat
        for i = 1:length(fileNames)
            temp_beta = niftiread(char(fileNames(i)));
            new_pat = reshape(temp_beta,size(temp_beta,1),[]);
            pats(i,:) = new_pat(mask==1);
            % fprintf('item %f pattern complete \n', i)
        end
        
        %['/scratch/polynlab/fmri/cdcatmr/cdcatmr', num2str(subj, '%03.f'), '/images/pattern/', num2str(subj, '%03.f'), '_VT.txt']
        csvwrite([resDir, saveName], pats)
    end
end
end
