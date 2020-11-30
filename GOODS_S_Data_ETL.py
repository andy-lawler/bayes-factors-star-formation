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

###############################################
#########Data ETL and filtering################
###############################################

#obtain 3D-HST GOODS-S photometric data at https://3dhst.research.yale.edu/Data.php

df = pd.read_csv("goodss_3dhst.v4.1_clean.cat", header=0, sep='\s+')

df1 = df[['id','f_F160W', 'e_F160W', 'f_U38', 'e_U38', 'f_U', 'e_U',
          'f_F435W', 'e_F435W', 'f_B', 'e_B', 'f_V', 'e_V',
          'f_F606W', 'e_F606W', 'f_R', 'e_R', 'f_Rc', 'e_Rc',
          'f_F775W', 'e_F775W', 'f_I', 'e_I', 'f_F814Wcand', 'e_F814Wcand', 
          'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_J', 'e_J', 
          'f_tenisJ', 'e_tenisJ', 'f_F140W', 'e_F140W', 'f_H', 'e_H', 
          'f_tenisK', 'e_tenisK', 'f_Ks', 'e_Ks', 'f_IRAC1', 'e_IRAC1', 
          'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', 
          'star_flag', 'use_phot', 'z_spec']]

df_0 = pd.read_csv("goodss_3dhst.v4.1_clean.zout", header=0, sep='\s+')

df_1 = df_0[['z_peak']]

df1 = df1.join(df_1)

df1_star_flag_0 = df1[(df1['star_flag'] == 0)] # For F160W<25, star=1 and galaxy=0, 12835 out of 50507
df1_star_flag_1 = df1[(df1['star_flag'] == 1)] # For F160W<25, star=1 and galaxy=0, 403 out of 50507
df1_star_flag_2 = df1[(df1['star_flag'] == 2)] # 37269 objects are fainter than 25 mag (dotted red line),
# which cannot be cleanly separated and are assigned a star_flag of 2. For the small fraction of objects 
# with no coverage in F 160W , we assign a star_flag of 2.
df1_use_phot = df1[(df1['use_phot'] == 1)] # 39998 out of 50507
df1_z_peak = df1[(df1['z_peak'] >= 0.9) & (df1['z_peak'] <= 1.1)] # 5345 out of 50507
df1_z_peak_low_z = df1[(df1['z_peak'] <= 1)] # 15414 are z = 1 or less
df1_final = df1[(df1['star_flag'] != 1) & (df1['use_phot'] == 1) & ((df1['z_peak'] >= 0.9) & (df1['z_peak'] <= 1.1))]
df1_final[df1_final < 0] = np.nan