import os
import pickle
import numpy as np

######################################################
########Get Data for Param KDE Plots v2.0#############
######################################################

#obtain 3D-HST GOODS-S photometric data at https://3dhst.research.yale.edu/Data.php

df_params = pd.read_csv("goodss_3dhst.v4.1_clean.fout", header=0, sep='\s+')

#get df1_final from GenerateErrors.py and merge on id
df_params_final = df1_final.merge(df_params, on=['id'], how='inner')

grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation

x_z = np.array(df_params_final[['z']])[:,0]
x_dust = np.array(df_params_final[['Av']])[:,0]
x_lage = np.array(df_params_final[['lage']])[:,0]
x_ltau = np.array(df_params_final[['ltau']])[:,0]
x_lmass = np.array(df_params_final[['lmass']])[:,0]

np.nanmin(x_dust)
np.nanmax(x_dust)
np.nanmin(x_lage)
np.nanmax(x_lage)
np.nanmin(x_ltau)
np.nanmax(x_ltau)
np.nanmin(x_lmass)
np.nanmax(x_lmass)

param_matrix = np.vstack([x_z, x_dust, x_lage, x_ltau, x_lmass]).T
#np.any(np.isnan(param_matrix))
#np.all(np.isfinite(param_matrix))
param_matrix = np.nan_to_num(param_matrix)

grid.fit(param_matrix)
kde = grid.best_estimator_

# Save file
pkl_filename_df = "dens_a.pkl"  
with open(pkl_filename_df, 'wb') as file:  
    pickle.dump(kde, file)