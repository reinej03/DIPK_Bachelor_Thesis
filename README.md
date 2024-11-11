# DIPK_Bachelor_Thesis

This is a repository containing code for my bachelor thesis. 
Additionally to this repository, the DIPK model integrated into the DrEval pipeline can be found in the DrEval GitHub (https://github.com/daisybio/drevalpy). 

The Preprocessing_Data directory contains the code for the preprocessing of the different inputs for the DIPK drug response prediction model.  The code was taken from the DIPK GitHub (https://github.com/user15632/DIPK) and merely altered in some cases to use pandas DataFrames as well as produce CSV files in the end. However, three files (Preprocessing_Data/Data/Cell_line_RMA_proc_basalExp.txt, Preprocessing_Data/PreprocessingB/PreTrain.pkl, Preprocessing_Data/PreprocessingC/PreprocessingC_MolGNet.pt) were too big for GitHub. These files can either be found under Data in DIPK’s Google Drive (https://drive.google.com/drive/folders/16hP48-noHi3-c_LP9TcZxkwAzqxgR0VB) or are created through running the code. 

The Randomizing_Drug_Features contains code for randomizing the drug features.

The Results_Analysis contains code for the analysis of the different evaluations that were used for my bachelor thesis. 
