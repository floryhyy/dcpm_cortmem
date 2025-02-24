### This folder contains generated results/itermedium data 

# network overlap
 - this folder contains 2 types of file:
  - 1. .*overlap.txt
  - these files contain a 377*377 binary matrix representing the overlap edges between 2 networks.
  - E.g. 'arousal_cortisol_alc_neg&cortisol_tool_neg_overlap' means overlap edges between arousal cortisol emotional network and arousal cortisol neutral netorks.

  - 2. ".*overlap_percent.txt"
  - these files contain percentage of overlap for networks within a construct or across constructs.

# dynamic_connectivity
 - this folder contains dynamic connectivity matrices.
 - each .mat file contains 3 variables:
     X: (total_trials,377,377) conn matrix for all subjects, all trials.
     y: (total_trials) behavior measurements for this block
     id_ls: (total_trials) subject id for each trial.

 - within per subj folder contains the matrices for each subject individually. within each file, the matrix has shape (n_trials, 377, 377), aka there is one conn matrix per trial.


