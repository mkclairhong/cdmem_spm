function run_beta2pat(subj)


subj = '015';

subjdir = ['/Users/beckycutler/data/fmri/cdcatmr/cdcatmr' num2str(subj)];


load([subjdir '/events/SPM.mat'])


%%


pos = nan(1,216);

pos = [1:27,34:60,67:93,100:126,133:159,166:192,199:225,232:258];

%%

roi_mask = '/Users/beckycutler/data/fmri/cdcatmr/all_subjects/masks/infT.nii';


pats = beta2pat(SPM,subj,roi_mask,pos);



newPats = pats(:,all(~isnan(pats)));

%save('[subj '_MTL_words.mat']' 'pats')
save('cdcatmr015_infT.mat', 'pats')
end






