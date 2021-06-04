function rename_beta(subjID)
%   simple script to rename beta files to corresponding stimuli trial and name
%
%   Parameters
%   ----------
%   subjID : int
%        simply subject ID in integers (ie. 5, 10, 25)
%
%   Returns
%   -------
%   renamed beta images in the same directory

% set path
number_tag = num2str(subjID, '%03.f');
baseDir = '/scratch/polynlab/fmri/cdcatmr/';  % scratch data dir
subjDir = [baseDir, 'cdcatmr', number_tag];
imgDir = [subjDir, '/images'];
betaDir = [imgDir, '/beta_study/'];

% read in SPM file to identify the corresponding stimuli
load([betaDir, 'SPM.mat']);

for i = 1:length(SPM.xX.name)
    
    % get trial number
    trial_info = SPM.xX.name{i}(4);
    
    % get serial position within trial
    index_info = string(mod(i, 33));  % 33 = 27 stimuli + 6 constants
    % if 33rd item, mod(33, 33) is 0 so we fix to 33
    if index_info == '0'
        index_info = '33';
    end

    % if name is xyzrpy
    if length(SPM.xX.name{i}) < 8  % misc number to filter out by name legnth
        stim_info = SPM.xX.name{i}(end);
    
	% if name is constant
    elseif contains(SPM.xX.name{i}, 'constant')
        stim_info = 'constant';
        index_info = '0';
        
    else
        % remove misc characters at front and end of SPM file anme 
        stim_info = SPM.xX.name{i}(7:end-6);

        for j = 1:length(stim_info)
            if isspace(stim_info(j))
                stim_info(j) = '_';
            end
        end
    end
    new_name = strcat(trial_info, '_', index_info, '_', stim_info, '.nii');
    disp(new_name)
    
    % beta study images automatically have 0s in front, so we need to check
    if 99 < i && i < 1000
        beta_index = strcat('0', string(i));
    elseif 9 < i  && i < 100
        beta_index = strcat('00', string(i));
    else
        beta_index = strcat('000', string(i));
    end
    
    % rename nii files
    movefile(strcat(betaDir, 'beta_', beta_index, '.nii'), strcat(betaDir, new_name));
end
