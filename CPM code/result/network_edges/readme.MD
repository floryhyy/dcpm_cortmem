# edges in each predictive network. 8 predictive networks for memory and 8 predictive networks for arousal.
# Each text file contains a 377*377 binary matrix for each predictive network, with 1 means this edge is in the network, 0 means not. 

# file name
    - RecogAcc_coded == Memory
    - Arous_All == Arousal
    - tool == Neutral Run
    - alc == Emotional Run
    - pos == postive network (edges in the network [positively] correlated with behavior)
    - neg == negative network (edges in the network negatively correlated with behavior)
    - e.g. "n26_RecogAcc_coded_byTrial_cortisol_alc_htBPfiltered_p0.01_neg.txt" means negative network predicting memory under cortisol in emotional run
