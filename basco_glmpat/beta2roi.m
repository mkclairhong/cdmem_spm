function beta2roi(subj)
% make ROIs from betaseries images
subj = num2str(subj,'%03.f');

imgDir = (['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/images/betaseries']);
% imgDir = (['/Users/beckycutler/data/fmri/cdcatmr/cdcatmr', subj, '/images/func/betaseries']);

files = dir(imgDir); files = {files.name}';

scanN = csvread('/scratch/polynlab/fmri/cdcatmr/all_subjects/study_beta_scans.csv');

betaidx = files(contains(files,'beta'));
betaFiles = betaidx(scanN);

maskDir = ('/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/resliced/');
masks = dir(maskDir); masks = {masks(4:end).name}';


for m = 1:size(masks,1)
    
    curMask = niftiread(['/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/resliced/', masks{m}]);


    for s = 1:216

        V = niftiread([imgDir, '/beta_', num2str(s,'%04.f'), '.nii']);

        tempROI = V(curMask==1);

        pats(s,:) = tempROI;

    end
    
    pats = pats(:,all(~isnan(pats)));
    
    [filepath,name,ext] = fileparts(['/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/resliced/', masks{m}]);
    
    % if ~isdir(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/patterns/txt/']);
        % mkdir ['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/patterns/txt/'];
    % end
    
    csvwrite(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/patterns/txt/', name, '.txt'], pats)
    
    clear tempROI pats
    
end
    
end
