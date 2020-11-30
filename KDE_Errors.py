import numpy as np
import pandas as pd
import sys
import os
import math
import random
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sb
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import pickle
import itertools

######################################
######Create KDEs of obs errors#######
######################################

############################################################################

grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation

from scipy.stats import beta
x = beta.rvs(a=0.75, b=7.5, size=4500)

#import matplotlib.pyplot as plt
#plt.hist(x, bins=100)

#x = np.array(df1_final[['e_F160W']])[:,0]
#x = x[~np.isnan(x)]
#x_995 = np.percentile(x, 99.9)
#x[x>x_995] = np.nan
#x = x[~np.isnan(x)]
grid.fit(x[:, None])
print grid.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde = grid.best_estimator_

############################################################################

grid2 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1001, 0.2001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x2 = np.array(df1_final[['e_U38']])[:,0]
x2 = x2[~np.isnan(x2)]
x2_995 = np.percentile(x2, 99.9)
x2[x2>x2_995] = np.nan
x2 = x2[~np.isnan(x2)]
grid2.fit(x2[:, None])
print grid2.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde2 = grid2.best_estimator_

############################################################################

grid3 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x3 = np.array(df1_final[['e_U']])[:,0]
x3 = x3[~np.isnan(x3)]
x3_995 = np.percentile(x3, 99.9)
x3[x3>x3_995] = np.nan
x3 = x3[~np.isnan(x3)]
grid3.fit(x3[:, None])
print grid3.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde3 = grid3.best_estimator_

############################################################################

grid4 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x4 = np.array(df1_final[['e_B']])[:,0]
x4 = x4[~np.isnan(x4)]
x4_995 = np.percentile(x4, 99.9)
x4[x4>x4_995] = np.nan
x4 = x4[~np.isnan(x4)]
grid4.fit(x4[:, None])
print grid4.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde4 = grid4.best_estimator_

############################################################################

grid5 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x5 = np.array(df1_final[['e_V']])[:,0]
x5 = x5[~np.isnan(x5)]
x5_995 = np.percentile(x5, 99.9)
x5[x5>x5_995] = np.nan
x5 = x5[~np.isnan(x5)]
grid5.fit(x5[:, None])
print grid5.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde5 = grid5.best_estimator_

############################################################################

grid6 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x6 = np.array(df1_final[['e_F606W']])[:,0]
x6 = x6[~np.isnan(x6)]
x6_995 = np.percentile(x6, 99.9)
x6[x6>x6_995] = np.nan
x6 = x6[~np.isnan(x6)]
grid6.fit(x6[:, None])
print grid6.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde6 = grid6.best_estimator_

############################################################################

grid7 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x7 = np.array(df1_final[['e_R']])[:,0]
x7 = x7[~np.isnan(x7)]
x7_995 = np.percentile(x7, 99.9)
x7[x7>x7_995] = np.nan
x7 = x7[~np.isnan(x7)]
grid7.fit(x7[:, None])
print grid7.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde7 = grid7.best_estimator_

############################################################################

grid8 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x8 = np.array(df1_final[['e_Rc']])[:,0]
x8 = x8[~np.isnan(x8)]
x8_995 = np.percentile(x8, 99.9)
x8[x8>x8_995] = np.nan
x8 = x8[~np.isnan(x8)]
grid8.fit(x8[:, None])
print grid8.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde8 = grid8.best_estimator_

############################################################################

grid9 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x9 = np.array(df1_final[['e_F775W']])[:,0]
x9 = x9[~np.isnan(x9)]
x9_995 = np.percentile(x9, 99.9)
x9[x9>x9_995] = np.nan
x9 = x9[~np.isnan(x9)]
grid9.fit(x9[:, None])
print grid9.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde9 = grid9.best_estimator_

############################################################################

grid10 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x10 = np.array(df1_final[['e_I']])[:,0]
x10 = x10[~np.isnan(x10)]
x10_995 = np.percentile(x10, 99.9)
x10[x10>x10_995] = np.nan
x10 = x10[~np.isnan(x10)]
grid10.fit(x10[:, None])
print grid10.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde10 = grid10.best_estimator_

############################################################################

grid11 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x11 = np.array(df1_final[['e_F814Wcand']])[:,0]
x11 = x11[~np.isnan(x11)]
x11_995 = np.percentile(x11, 99.9)
x11[x11>x11_995] = np.nan
x11 = x11[~np.isnan(x11)]
grid11.fit(x11[:, None])
print grid11.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde11 = grid11.best_estimator_

############################################################################

grid12 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x12 = np.array(df1_final[['e_F850LP']])[:,0]
x12 = x12[~np.isnan(x12)]
x12_995 = np.percentile(x12, 99.9)
x12[x12>x12_995] = np.nan
x12 = x12[~np.isnan(x12)]
grid12.fit(x12[:, None])
print grid12.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde12 = grid12.best_estimator_

############################################################################

grid13 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x13 = np.array(df1_final[['e_F125W']])[:,0]
x13 = x13[~np.isnan(x13)]
x13_995 = np.percentile(x13, 99.9)
x13[x13>x13_995] = np.nan
x13 = x13[~np.isnan(x13)]
grid13.fit(x13[:, None])
print grid13.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde13 = grid13.best_estimator_

############################################################################

grid14 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x14 = np.array(df1_final[['e_J']])[:,0]
x14 = x14[~np.isnan(x14)]
x14_995 = np.percentile(x14, 99.9)
x14[x14>x14_995] = np.nan
x14 = x14[~np.isnan(x14)]
grid14.fit(x14[:, None])
print grid14.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde14 = grid14.best_estimator_

############################################################################

grid15 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x15 = np.array(df1_final[['e_tenisJ']])[:,0]
x15 = x15[~np.isnan(x15)]
x15_995 = np.percentile(x15, 99.9)
x15[x15>x15_995] = np.nan
x15 = x15[~np.isnan(x15)]
grid15.fit(x15[:, None])
print grid15.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde15 = grid15.best_estimator_

############################################################################

grid16 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x16 = np.array(df1_final[['e_F140W']])[:,0]
x16 = x16[~np.isnan(x16)]
x16_995 = np.percentile(x16, 99.9)
x16[x16>x16_995] = np.nan
x16 = x16[~np.isnan(x16)]
grid16.fit(x16[:, None])
print grid16.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde16 = grid16.best_estimator_

############################################################################

grid17 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x17 = np.array(df1_final[['e_H']])[:,0]
x17 = x17[~np.isnan(x17)]
x17_995 = np.percentile(x17, 99.9)
x17[x17>x17_995] = np.nan
x17 = x17[~np.isnan(x17)]
grid17.fit(x17[:, None])
print grid17.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde17 = grid17.best_estimator_

############################################################################

grid18 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x18 = np.array(df1_final[['e_tenisK']])[:,0]
x18 = x18[~np.isnan(x18)]
x18_995 = np.percentile(x18, 99.9)
x18[x18>x18_995] = np.nan
x18 = x18[~np.isnan(x18)]
grid18.fit(x18[:, None])
print grid18.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde18 = grid18.best_estimator_

############################################################################

grid19 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x19 = np.array(df1_final[['e_Ks']])[:,0]
x19 = x19[~np.isnan(x19)]
x19_995 = np.percentile(x19, 99.9)
x19[x19>x19_995] = np.nan
x19 = x19[~np.isnan(x19)]
grid19.fit(x19[:, None])
print grid19.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde19 = grid19.best_estimator_

############################################################################

grid20 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x20 = np.array(df1_final[['e_IRAC1']])[:,0]
x20 = x20[~np.isnan(x20)]
x20_995 = np.percentile(x20, 99.0)
x20[x20>x20_995] = np.nan
x20 = x20[~np.isnan(x20)]
grid20.fit(x20[:, None])
print grid20.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde20 = grid20.best_estimator_

############################################################################

grid21 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x21 = np.array(df1_final[['e_IRAC2']])[:,0]
x21 = x21[~np.isnan(x21)]
x21_995 = np.percentile(x21, 99.0)
x21[x21>x21_995] = np.nan
x21 = x21[~np.isnan(x21)]
grid21.fit(x21[:, None])
print grid21.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde21 = grid21.best_estimator_

############################################################################

grid22 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x22 = np.array(df1_final[['e_IRAC3']])[:,0]
x22 = x22[~np.isnan(x22)]
x22_995 = np.percentile(x22, 95.0)
x22[x22>x22_995] = np.nan
x22 = x22[~np.isnan(x22)]
grid22.fit(x22[:, None])
print grid22.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde22 = grid22.best_estimator_

############################################################################

#grid23 = GridSearchCV(KernelDensity(),
#                    {'bandwidth': np.linspace(0.001, 1.0, 1000)},
#                    cv=10) # 10-fold cross-validation
#x23 = np.array(df1_final[['e_IRAC4']])[:,0]
#x23 = x23[~np.isnan(x23)]
#grid23.fit(x23[:, None])
#print grid23.best_params_

#kde23 = grid23.best_estimator_
#min(x23)
#max(x23)
#x23_grid = np.linspace(-0.1, 10, 4500) # unique to each distribution
#pdf23 = np.exp(kde23.score_samples(x23_grid[:, None]))

#fig23, ax23 = plt.subplots()
#ax23.plot(x23_grid, pdf23, linewidth=3, alpha=0.5, label='bw=%.2f' % kde23.bandwidth)
#ax23.hist(x23, 300, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
#ax23.legend(loc='upper left')
#ax23.set_xlim(-0.1, 10);

#random.choice(pdf23)

#max((random.gauss(0.02, 0.002)), kde23.sample(1)[0][0])

############################################################################

grid24 = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.1001, 101),
                     'algorithm': ['kd_tree']},
                    cv=5, n_jobs=-1) # 5-fold cross-validation
x24 = np.array(df1_final[['e_F435W']])[:,0]
x24 = x24[~np.isnan(x24)]
x24_995 = np.percentile(x24, 99.9)
x24[x24>x24_995] = np.nan
x24 = x24[~np.isnan(x24)]
grid24.fit(x24[:, None])
print grid24.best_params_

#if .0002 to .1000, rerun in between the range
#if .1001, rerun with np.linspace(0.1001, 0.2001, 101)

kde24 = grid24.best_estimator_

############################################################################

os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/bpv079/GOODS-S 3DHST KDE Errors/')

# Save to file in the current working directory
pkl_filename1 = "kde_e_F160W.pkl"  
with open(pkl_filename1, 'wb') as file:  
    pickle.dump(kde, file)
    
pkl_filename2 = "kde_e_U38.pkl"  
with open(pkl_filename2, 'wb') as file:  
    pickle.dump(kde2, file)
    
pkl_filename3 = "kde_e_U.pkl"  
with open(pkl_filename3, 'wb') as file:  
    pickle.dump(kde3, file)
    
pkl_filename4 = "kde_e_B.pkl"  
with open(pkl_filename4, 'wb') as file:  
    pickle.dump(kde4, file)

pkl_filename5 = "kde_e_V.pkl"  
with open(pkl_filename5, 'wb') as file:  
    pickle.dump(kde5, file)
    
pkl_filename6 = "kde_e_F606W.pkl"  
with open(pkl_filename6, 'wb') as file:  
    pickle.dump(kde6, file)
    
pkl_filename7 = "kde_e_R.pkl"  
with open(pkl_filename7, 'wb') as file:  
    pickle.dump(kde7, file)
    
pkl_filename8 = "kde_e_Rc.pkl"  
with open(pkl_filename8, 'wb') as file:  
    pickle.dump(kde8, file)
    
pkl_filename9 = "kde_e_F775W.pkl"  
with open(pkl_filename9, 'wb') as file:  
    pickle.dump(kde9, file)
    
pkl_filename10 = "kde_e_I.pkl"  
with open(pkl_filename10, 'wb') as file:  
    pickle.dump(kde10, file)
    
pkl_filename11 = "kde_e_F814Wcand.pkl"  
with open(pkl_filename11, 'wb') as file:  
    pickle.dump(kde11, file)
    
pkl_filename12 = "kde_e_F850LP.pkl"  
with open(pkl_filename12, 'wb') as file:  
    pickle.dump(kde12, file)
    
pkl_filename13 = "kde_e_F125W.pkl"  
with open(pkl_filename13, 'wb') as file:  
    pickle.dump(kde13, file)

pkl_filename14 = "kde_e_J.pkl"  
with open(pkl_filename14, 'wb') as file:  
    pickle.dump(kde14, file)
    
pkl_filename15 = "kde_e_tenisJ.pkl"  
with open(pkl_filename15, 'wb') as file:  
    pickle.dump(kde15, file)

pkl_filename16 = "kde_e_F140W.pkl"  
with open(pkl_filename16, 'wb') as file:  
    pickle.dump(kde16, file)
    
pkl_filename17 = "kde_e_H.pkl"  
with open(pkl_filename17, 'wb') as file:  
    pickle.dump(kde17, file)
    
pkl_filename18 = "kde_e_tenisK.pkl"  
with open(pkl_filename18, 'wb') as file:  
    pickle.dump(kde18, file)
    
pkl_filename19 = "kde_e_Ks.pkl"  
with open(pkl_filename19, 'wb') as file:  
    pickle.dump(kde19, file)
    
pkl_filename20 = "kde_e_IRAC1.pkl"  
with open(pkl_filename20, 'wb') as file:  
    pickle.dump(kde20, file)
    
pkl_filename21 = "kde_e_IRAC2.pkl"  
with open(pkl_filename21, 'wb') as file:  
    pickle.dump(kde21, file)
    
pkl_filename22 = "kde_e_IRAC3.pkl"  
with open(pkl_filename22, 'wb') as file:  
    pickle.dump(kde22, file)
    
#pkl_filename23 = "kde_e_IRAC4.pkl"  
#with open(pkl_filename23, 'wb') as file:  
#    pickle.dump(kde23, file)
    
pkl_filename24 = "kde_e_F435W.pkl"  
with open(pkl_filename24, 'wb') as file:  
    pickle.dump(kde24, file)    