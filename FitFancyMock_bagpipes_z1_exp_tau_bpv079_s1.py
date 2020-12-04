import numpy as np
import pandas as pd
import bagpipes as pipes
import sys
import os
import pickle
import mpi4py

os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/bpv079/')

GOODSS_3DHST_filt= ['3DHST GOODS-S Filters For BAGPIPES/vimos_u.res',
                    '3DHST GOODS-S Filters For BAGPIPES/R.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/wfi_BB_U38_ESO841.res', 
                    '3DHST GOODS-S Filters For BAGPIPES/B_cdfs_tot.dat', 
                    '3DHST GOODS-S Filters For BAGPIPES/V_cdfs_tot.dat', 
                    '3DHST GOODS-S Filters For BAGPIPES/R_cdfs_tot.dat', 
                    '3DHST GOODS-S Filters For BAGPIPES/I_cdfs_tot.dat',  
                    '3DHST GOODS-S Filters For BAGPIPES/wfc_f435w_t77.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/wfc_f775w_t77.dat',            
                    '3DHST GOODS-S Filters For BAGPIPES/wfc_f606w_t81.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/wfc_f814w_t81.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/wfc_f850lp_t81.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/f140w.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/f125w.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/f160w.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/isaac_j.res',
                    '3DHST GOODS-S Filters For BAGPIPES/isaac_h.res',
                    '3DHST GOODS-S Filters For BAGPIPES/isaac_ks.res',
                    '3DHST GOODS-S Filters For BAGPIPES/cfh8101_J.txt',
                    '3DHST GOODS-S Filters For BAGPIPES/cfh8302_Ks.txt',
                    '3DHST GOODS-S Filters For BAGPIPES/irac_tr1_2004-08-09.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/irac_tr2_2004-08-09.dat',
                    '3DHST GOODS-S Filters For BAGPIPES/irac_tr3_2004-08-09.dat'#,
                    #'irac_tr4_2004-08-09.dat'
                    ]

with open('goodss_3dhst_mock_df_08-15-19_z1_s1.pkl', 'rb') as file10:  
    goodss_3dhst_mock_df = pickle.load(file10)

goodss_3dhst_mock_df = np.round(goodss_3dhst_mock_df, decimals=7)
#goodss_3dhst_mock_df = goodss_3dhst_mock_df.fillna(0.001)

#goodss_3dhst_mock_df.to_csv("goodss_3dhst_mock_df_08-15-19_z1_s1.dat", sep='\t')

goodss_3dhst_mock_df_fit = np.loadtxt("goodss_3dhst_mock_df_08-15-19_z1_s1.dat", skiprows=1, usecols=(np.arange(2, 48)))
#goodss_3dhst_mock_df_fit = goodss_3dhst_mock_df_fit[:-3567]

def load_CANDELS_GDSS(ID):

    # load up the relevant columns from the catalogue.
    cat = np.loadtxt("goodss_3dhst_mock_df_08-15-19_z1_s1.dat", skiprows=1,
                     usecols=(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,
                              3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47))
    
    # Find the correct row for the object we want.
    row = int(ID) - 1

    # Extract the object we want from the catalogue.
    fluxes = cat[row, :23]
    fluxerrs = cat[row, 23:]
    
    # Turn these into a 2D array.
    photometry = np.c_[fluxes, fluxerrs]
    
    # blow up the errors associated with any missing fluxes.
    for i in range(len(photometry)):
        if (photometry[i, 0] <= 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]
            
    #Enforce a maximum SNR of 20, or 10 in the IRAC channels.
    #for i in range(len(photometry)):
    #    if i < 20:
    #       max_snr = 100.
    #        
    #    else:
    #        max_snr = 50.
    #    
    #    if photometry[i, 0]/photometry[i, 1] > max_snr:
    #        photometry[i, 1] = photometry[i, 0]/max_snr
     
    return photometry

####################################################
##############Delayed Tau Model#####################
####################################################

dust = {}
dust["type"] = "Calzetti"  # Type of dust, other options "Cardelli", "CF00" (required)
dust["Av"] = (0.0001, 3.0)        # Absolute attenuation in the V band in magnitudes (required)
dust["Av_prior"] = "log_10"
#dust["eta"] = 2.            # Multiplicative factor on Av for stars in birth clouds (optional)

nebular = {}
nebular["logU"] = -3.0     # Log_10 of the ionization parameter (required).

delayed = {}               # Delayed Tau model e.g. Thomas et al. (2017)
delayed["massformed"] = (6.5, 12.5)   # Log_10 total stellar mass formed in Solar masses (required)
delayed["metallicity"] = 1.0  # Metallicity in old Solar units, i.e. Z_sol = 0.02 (required)
delayed["age"] = (0.05, 10.)       # Time since star formation began in Gyr (required)
delayed["age_prior"] = "log_10"
delayed["tau"] = (0.01, 10.)       # Timescale of exponential decrease in Gyr (required)
delayed["tau_prior"] = "log_10"

burst = {}                           # Delta function burst
burst["age"] = (0.05, 0.90)     # Time since burst: Gyr
#burst["age_prior"] = "log_10"
burst["massformed"] = 7.7  # Log_10 total stellar mass formed in Solar masses (required)
burst["metallicity"] = 1.0  # Metallicity in old Solar units, i.e. Z_sol = 0.02 (required)

fit_info = {}
fit_info["dust"] = dust
fit_info["delayed"] = delayed
fit_info["burst"] = burst
fit_info["nebular"] = nebular
fit_info["t_bc"] = 0.01
#fit_info["veldisp"] = 0.

####New Version Catalog Call
IDs = np.arange(1, 501)
redshifts = np.loadtxt("goodss_3dhst_mock_df_08-15-19_z1_s1.dat", skiprows=1, usecols=1)
cat_fit = pipes.fit_catalogue(IDs, fit_info, load_CANDELS_GDSS, spectrum_exists=False,
                              cat_filt_list=GOODSS_3DHST_filt, run="cat_fit_delayed_pmn_11-11-19_z1_s1",
                              redshifts=redshifts, make_plots=False)
cat_fit.fit(verbose=False, n_live=1500)