function recall_onsets(subjID)

% script to make 8 separate runs .mat files

% clear all
subj = num2str(subjID, '%03.f');

% subj = '007';

tblEvents = readtable(['/scratch/polynlab/fmri/cdcatmr/cdcatmr' subj '/events/cdcatmr' subj 'events.csv']);

events = table2struct(tblEvents);

duration = zeros(1,27); 
duration = num2cell(duration);

for t = 1:8
    % trial subset
    trial = events([events.trialN] == t);

    % stim pres subset
    recPres = trial(strcmp({trial.types}, 'rec_word'));
    names = {recPres.item};

    % scan N
    
    for c = 1:3
        cat = recPres(strcmp({recPres.category}, num2str(c)));
        catScans = {cat.onset_ttl_start}
        % ERROR HERE
        catOnset(c,:) = catScans;
    end
    onsets = catOnset;
    
    cell2csv(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/images/func/func', num2str(t), '/recall_onsets.csv'], onsets);
    cd(['/scratch/polynlab/fmri/cdcatmr/cdcatmr' subj '/events/recall']);
    
    if ~isdir('glm')
        mkdir glm
    end
    
    save(['/scratch/polynlab/fmri/cdcatmr/cdcatmr' subj '/events/glm/rec2_run' num2str(t) '.mat'], 'onsets','duration','names')
    
end
    
disp('Completed onset')


% init lists
all_item_onsets = [];
all_cat_onsets = [];
all_rp = [];
stim_names = [];

for r = 1:8 % trial loop
    load(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/events/glm/rec2_run', num2str(r), '.mat']) 
    
    names = names';
    
    stim_names = [stim_names; names];
    item = onsets';
         
    all_item_onsets = [all_item_onsets; item];
    
    % this saves item onsets to CSV - make same category
    cell2csv(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/images/func/func', num2str(r), '/rec_onsets_', num2str(r), '.csv'], item);
    
end
disp('Completed Item Loop')

% init lists
% all_item_onsets = [];
% all_cat_onsets = [];
% all_rp = [];
% stim_names = [];

% for r = 1:8 % trial loop
%     load(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/events/cat_glm/run', num2str(r), '.mat']) 
    
%     names = names';
    
%     stim_names = [stim_names; names];
%     item = onsets;
         
%     all_cat_onsets = [all_cat_onsets; catOnset];
    
%     % this saves item onsets to CSV - make same category
%     cell2csv(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', subj, '/images/func/func', num2str(r), '/cat_onsets.csv'], item);
    
% end
% disp('Completed Cat Loop')

% end
