function create_recall_events(subjID)
%   import events csv and convert to struct
%   creating duration.mat, cond.mat, stimOnset.mat in /events directory
%
%   Parameters
%   ----------
%   subjID : int
%        simply subject ID in integers (ie. 5, 10, 25)
%
%   Returns
%   -------
%   duration.mat : mat
%        cell of zeros with row/column size trial x list length
%   cond.mat : mat
%        cell of trial x 1 with 0 for light, 1 for heavy distraction conditions
%   stimOnset.mat : mat
%        cell of trial x list length of scan numbers for each stimuli presentation


% function variables (Do not change unless you know what you are doing)
%---------------------------------------------------------------------%
baseDir = '/scratch/polynlab/fmri/cdcatmr/';  % scratch data dir

%---------------------------------------------------------------------%

number_tag = num2str(subjID, '%03.f');
subject = ['cdcatmr', number_tag];
subjDir = [baseDir, subject];


% look for the final events file from annotation code
if isfile([subjDir, '/events/', subject, 'events.csv'])
    % import csv as table then convert to struct
    eventsTbl = readtable([subjDir, '/events/', subject, 'events.csv']);
    
    % read in column names
    colNames = eventsTbl.Properties.VariableNames;
    events = table2struct(eventsTbl);
    
    % colNames = {'trial','index','trainN','trainPos','item','cond','type',...
    %     'cat','recall','resp','rt','isi','duration','trialTime','trialTime_samp','startScan_samp',...
    %     'endScan_samp','scanN','endScan','globalTime','intrusion','onset',...
    %     'offset','onset_sample','offset_sample','onset_ttl_start','onset_ttl_end',...
    %     'offset_ttl_start','offset_ttl_end'};
    
    rec_names = {};
    for t = 1:8 % trial loop
        
        % trial subset
        trial = events([events.trialN] == t);
        
        % stim pres subset
        rec_event = trial(strcmp({trial.types}, 'rec_word'));
        
        % scan N
        scanOnset(t,:) = {rec_event(:).start_scan};
        cond(t,1) = rec_event(1).condition;
    end
    
    rec_all = events(strcmp({events.types}, "rec_word"));
    rec_names =  cellstr({rec_all.item});
    
    % save to csv files
    save([subjDir, '/events/recall_stimOnset.mat'], 'scanOnset');
    disp(['Successfully saved stimOnset as ', [subjDir, '/events/recall_stimOnset.mat']])
    
    save([subjDir, '/events/recall_cond.mat'], 'cond');
    disp(['Successfully saved cond as ', [subjDir, '/events/recall_cond.mat']])
    
    duration = zeros(8,27); 
    duration = num2cell(duration);
    save([subjDir, '/events/recall_duration.mat'], 'duration');
    disp(['Successfully saved duration as ', [subjDir, '/events/recall_duration.mat']])
else
    
    disp(['No events file found. Looked for ', [subjDir, '/events/', subject, 'events.csv']])
end 
    
end
    
    