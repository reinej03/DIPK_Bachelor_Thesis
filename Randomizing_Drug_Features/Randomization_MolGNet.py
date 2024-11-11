import numpy as np 
import os 
import pandas as pd

for each in os.listdir("./Data/Drugs"):
    MolGNet_df_old = pd.read_csv(f"../Data/Drugs/{each}/MolGNet_{each}.csv", sep='\t', index_col=0) 
    MolGNet_df_new = np.random.randn(MolGNet_df_old.shape[0], MolGNet_df_old.shape[1])
    MolGNet_df_new = pd.DataFrame(MolGNet_df_new)
    MolGNet_df_new.to_csv(f"./Data/Drugs_MolGNet_Randomized/{each}/MolGNet_{each}.csv", sep='\t')
    