

% script to grab stimuli onset scans from raw images

%load('/Users/beckycutler/data/fmri/cdcatmr/cdcatmr006/images/beta_study/SPM.mat');

addpath(genpath('/home/jeonj1/matlab/spm12'));

for subj = 10:10

    load(['/scratch/polynlab/fmri/cdcatmr/cdcatmr', num2str(subj, '%03.f'), '/images/beta_study/SPM.mat']);


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
            
            scanN = ['wrfunc', num2str(r), '_', num2str(s, '%05.f'), '.nii']

            
            fileNames{cnt,1} = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr',...
                num2str(subj, '%03.f'), '/images/func_rtf/func', num2str(r), '/', scanN];
                
            cnt = cnt + 1;
                
        end
            
    end
    
    fileNames

    %%
    
    
    %rawPath = '/Users/beckycutler/data/fmri/cdcatmr/cdcatmr006/images/raw_study';
    rawPath = ['/scratch/polynlab/fmri/cdcatmr/cdcatmr', num2str(subj, '%03.f'), '/patterns/test_raw_study'];
    rawPath
    
    for n = 1:216
        
        scanPath = fileNames{n};
        
        copyfile(scanPath, rawPath);
        fprintf('file %i copied \n', n);
        
    end
    
end
    
    %%

