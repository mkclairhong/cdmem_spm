# cdcat_spm (Neuroimaging pipeline for cdcatmr)
- [Installation & Setup](#installation--setup)
- [Code Format](#code-format)
- [Pipeline Overview](#pipeline-overview)
- [Subject Data Folder Hierarchy](#subject-data-folder-tree)
- [Run Time](#approximate-run-time)
- [Filesize](#filesize)
- [Useful Links](#useful-links)

## Code Format
Code is formatted so that MATLAB scripts/functions run a single subject. For each step (as divided by each folder), Slurm script compiles the required functions, and submits a parallel job array for efficiency. Once you have SPM12 and Atlas setup, the only variables you need to change are within Slurm files (You do not have to change anything in MATLAB scripts). All Slurm scripts share the same sBatch directives, and the only important variables you need to change are paths, your MATLAB version on ACCRE (check via `ml` on ACCRE terminal), and misc info such as your email to receive job updates.

There are **three** important paths:
1. Your MATLAB/SPM12 path
2. Your cloned directory of cdcat_spm (to access custom MATLAB scripts)
3. Scratch data directory (On ACCRE, all MATLAB scripts are set to `/scratch/polynlab/fmri/cdcatmr` so the code knows where to find the data)

So these would be the only main variables that need to be changed in Slurm scripts.
- [See detail for Slurm scripts](https://github.com/vucml/cdcat_spm/wiki/Running-sBatch-Scripts)

## Pipeline Overview (as of 03/10/20)  
1. **Get raw images from gStudy** [[See detail](https://github.com/vucml/cdcat_spm/wiki/Using-gStudy)]  
  - scp and setup data on ACCRE scratch directory  
2. **Run `preproc.slurm` with switch option '1'**  
  - `split_imgs_anat.m`: redirects anat files to /images/anat/  
  - `split_imgs_func.m`: splits each functional run to 187 scan files (4D-> 3D)  
  - `preproc_cdcatmr.m`: the actual preprocessing  
    + realign within subject  
    + create tissue probability maps  
    + create grey matter mask  
    + coregister all functional images within trial to first functional scan  
    + register images to deformation field (calculate mutual information, transform to MNI space)  
    + smooth (optional - wrfunc are not smoothed)  
3. **Run 'glm/runGlm.slurm'**  
  - The previous glm code is not being used so this slurm file is temporarily used only to create relevant events mat files  
  - `createEvents.m`: creates 'duration.mat', 'cond.mat', 'stimOnset.mat' in /events/  
  - `makeRunMats.m`: creates files for each functional run in /events/glm/...  
4. **Run 'basco_glmpat/create_onsets.m'**  
  - Because this script can be run very quickly, simply launch matlab and run it manually
  ```
  >>> matlab  # will launch matlab in terminal
  # Now that you are in matlab
  >>> addpath(genpath('YOUR/PATH/TO/CDCAT_SPM'))
  >>> create_onsets(N)  # where N = integer of subj number
  # ie. create_onsets(29)
  >>>
  Completed onsetByCat
  Completed Item Loop
  Completed Cat Loop
  ```
5. **Create _Item-level_ beta Images by running `/basco_glmpat/item_glm_batch.slurm`**  
  - Simply change the SLURM array to specify the subject ID number
  - Takes ~2, 3 hours
  - This will create `/subjID/images/func/betaItem` directory with all the beta images  


6. **Create _category/train-level_ beta Images by running `/basco_glmpat/cat_glm_batch.slurm`**  
  - Simply change the SLURM array to specify the subject ID number  
  - Takes less than an hour  
  - This will create `/subjID/images/func/cat_betaseries` directory with all the beta images  


7. **Create patterns using the beta images by running `/basco_glmpat/beta2roi.slurm`**  
  - Simply change the SLURM array to specify the subject ID number  
  - Takes about ~4, 5 hours  
  - This will create `/subjID/patterns/txt` directory with all the patterns for each mask.nii files found in the given directory  


### Creating Recall Period Patterns
All the codes required are in `cdcat_spm/recall`. Since we use images that are preprocessed, creating patterns for recall period takes about ~5 hours per subject.
1. **Run `recall_basco.slurm`**  
  - This script calls in two scripts:
    - `create_recall_onsets.py`: which simply creates rec_onset_N.csv files in each run's `image/func/funcN` directory
    - `recall_glm_batch.m`: which does the actual BASCO codes for the recall period.


2. **Run `recall_beta2roi.slurm`**  
  - This script calls in two scripts:
    - `recall_beta2roi.m`: MATLAB script that creates the patterns based on resliced ROIs, and saves the files in the subject directory `/patterns/recall_pat`
    - `filter_pads.py`: because recall periods vary by size for each run, it removes any rows that are initially padded as 0s.
      - If you see the slurm output message, it will tell which ROIs have been correctly converted to remove the paddings.


### Previous pipeline
<details><summary>**Click here to see previous pipeline version**</summary>

1. **Get raw images from gStudy** [[See detail](https://github.com/vucml/cdcat_spm/wiki/Using-gStudy)]
  - scp and setup data on ACCRE scratch directory
2. **Run `preproc.slurm`**
  - `split_imgs_anat.m`: redirects anat files to /images/anat/
  - `split_imgs_func.m`: splits each functional run to 187 scan files (4D-> 3D)
  - `preproc_cdcatmr.m`: the actual preprocessing
    + realign within subject
    + create tissue probability maps
    + create grey matter mask
    + coregister all functional images within trial to first functional scan
    + register images to deformation field (calculate mutual information, transform to MNI space)
    + smooth (optional - wrfunc are not smoothed)
3. **Run `runGlm.slurm`**
  - `createEvents.m`: creates 'duration.mat', 'cond.mat', 'stimOnset.mat' in /events/
  - `makeRunMats.m`: creates files for each functional run in /events/glm/...
  - `est_model_job.m`: for study period classification, creates a regression model that has a regressor for each single stimulus presentation. Additional regressors are 6 motion parameters (x,y,z,r,p,w) and subject level constants
  - `rename_beta.m`: renames the created beta files in /images/beta_study/ from beta000N.nii to trial_index_item.nii
4. **Extracting Patterns from Beta Images; Run `runPat.slurm`**
  - `beta2pat_loop.m`:
    + reslices beta images based on given mask
    + extracts voxels based on binary mask (just 1's)
    + creates matrix where each row is an event (stimulus presentation), and each column is a voxel
    + value within a cell is the activation of a voxel at the time of study
    + these vectors (~11,000 elements) are the patterns that we use for classification (e.g. to predict category)
  - **Optional: Run `pickle_pattern.py`**
    + creates a csv file with pattern information, showing corresponding item name, category, condition, and etc
    + additionally creates `pkl` directory in patterns with Python-readable full data structure
    + **Reading Pickled Patterns**:
    You can read the pickled patterns in `/subjDir/patterns/pkl/roi_pattern.pkl` by using `pandas.read_pickle('pickle_name')` which will conserve the same pandas dataframe that was saved.
    Then on, you can use the read in pickle format as df instance.
</details>


### Creating ROI Masks from Atlas
all the masks reside in `/scratch/polynlab/fmri/cdcatmr/all_subjects/masks` so that `beta2pat.m` function knows where to grab the file. Run the following MATLAB codes. No need for SLURM jobs.
  - run `run_aal.m`: this code will read in the AAL3 atlas text file that specifies the numbers for different ROIs, and calls in `write_binary_aal_mask` to create the mask
  - run `reslice_masks.m`: this will reslice the mask dimensions to match with that of the preprocessed images. Will generated "rROI.nii" files.
    - Resliced masks are located in '/all_subjects/masks/resliced/'


## Subject Data Folder Tree
Working on ACCRE, we assume all subject data are within `/scratch/polynlab/fmri/cdcatmr/`. Example subject data structure on scratch is below:
```
subjectID
├── events            # subjIDevents.csv file with event details (used for glm code)
│   ├── glm
│   └── recall        # contains recall period onset files
├── images
│   ├── anat          # anatomical scan files
│   ├── func
│   │   ├── func1     # images broken down by each run
│   │   ├── ...       # images broken down by each run
│   │   └── func8     # images broken down by each run
│   ├── beta_study    # beta images created from runGlm.slurm > est_model.m (incorrect glm method/not recommended)
│   ├── betaseries    # beta images created from item_glm_batch.slurm
│   ├── cat_betaseries# beta images created from cat_glm_batch.slurm
│   └── raw           # raw files from gStudy
├── patterns          # subjectIDpattern_info about stim's order, category, etc
│   ├── txt           # patterns from the most recent glm code (USE THIS)
│   ├── recall_pat    # patterns for recall events (note that the size of rows varies by subjects)
│   ├── pkl           # patterns in pickle/serialized data frame for Python use (incorrect glm method/not recommended for use)
│   └── mat           # patterns in mat form with vectors only (incorrect glm method/not recommended for use)
├── trial1            # PsychoPy outputs
├── ...               # PsychoPy outputs
└── trial8            # PsychoPy outputs
```

## Installation & Setup
1. **On ACCRE, cd into your directory you want to clone the repository. Then type:**
```
git clone https://github.com/vucml/cdcat_spm.git
```

2. **Unzip the spm12.zip file to your matlab directory**
  - the zipped spm12 file contains three additional files/folders:
    - AAL3 atlas
    - BASCO for running glm + custom BASCO script to run in slurm batch
    - marsbar
```
tar -zxvf spm12.tar.gz
```

### Installing SPM12, AAL3, BASCO, and Marsbar manually

1. **Download SPM12 to your ACCRE** <br>
  a. Go to https://www.fil.ion.ucl.ac.uk/spm/software/download/, dropdown to SPM12, and hit download<br>
  b. Unzip folder if necessary, and scp to your ACCRE MATLAB directory or anywhere accessible. For example,
    ```
    scp -r YOUR/LOCAL/PATH/TO/SPM12 VUNETID@login7.accre.vanderbilt.edu:/home/VUNETID/PATH/TO/MATLAB
    ```
2. **Download AAL3**<br>
  You can refer to AAL user guide (http://www.gin.cnrs.fr/wp-content/uploads/aal3-user-guide-gin-imn.pdf) for details. Below is quick summary on setting it up on ACCRE:
    <br>
    a. Go to http://www.gin.cnrs.fr/en/tools/aal/, and download AAL3 (released 2019 Aug)
    <br>
    b. cd in terminal to where AAL3 file was downloaded (Downloads if Mac). Then, scp the file to ACCRE to wherever you have SPM12  organized:
    ```
    scp AAL3_for_SPM12.tar.gz VUNETID@login7.accre.vanderbilt.edu:/home/VUNETID/PATH/TO/SPM12
    ```
    <br><br>
    c. Now login to ACCRE; cd to where you moved the AAL file.
    <br><br>
    d. Gunzip and untar the archive. In terminal, type:
    `tar -zxvf AAL3_for_SPM12.tar.gz`
    <br><br>
    e. Make `/atlas` directory in your spm12 folder. For me, I would do:
    `mkdir /home/jeonj1/matlab/spm12/atlas`
    <br><br>
    f. Let's move the necessary AAL files to the atlas directory

    ```
    cp /YOUR/PATH/TO/AAL3/AAL3.nii /YOUR/PATH/TO/SPM12/atlas/
    cp /YOUR/PATH/TO/AAL3/AAL3.xml /YOUR/PATH/TO/SPM12/atlas/
    cp /YOUR/PATH/TO/AAL3/AAL3.1mm.nii /YOUR/PATH/TO/SPM12/atlas/
    cp /YOUR/PATH/TO/AAL3/AAL3.1mm.xml /YOUR/PATH/TO/SPM12/atlas/
    ```

    <br><br>
    g. Now you can run `write_binary_aal_mask.m`, specifying where AAL file is located, region of interest (See AAL3.nii.txt for corresponding region-numbers), and output name


## Approximate Run Time
The process so far is broken down to:
1. `preproc.slurm` [8-10 hours per subject]
2. `runGlm.slurm` [15-45 minutes per subject]
3. `runPat.slurm` [<5 minutes per subject]

The rest of the codes are ran fairly quickly (within 30 min)


#### Filesize
For a single subject data, the original raw files are ~1GB<br>
After preproc, file size is ~134GB<br>
After GLM stage, file size is ~136GB<br>
After creating patterns, file size is ~137GB<br>
Removing middle-step func files reduces the file size to ~13GB.


## Useful Links
- [Repository Wiki](https://github.com/vucml/cdcat_spm/wiki)
- [SPM struct overview](http://people.duke.edu/~njs28/spmdatastructure.htm)
