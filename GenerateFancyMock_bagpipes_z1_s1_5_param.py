import numpy as np
import os
import sys
import math
import random
from scipy import integrate
import bagpipes as pipes
import pickle
import pandas as pd
import gzip
import itertools
import seaborn as sb
import dill

os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/3DHST GOODS-S Filters For BAGPIPES/')

################################
#######Clean Up Filters#########
################################

#U38 = pd.read_csv("wfi_BB_U38_ESO841.res", header=None, sep='\s+')
#U38 = U38.drop(U38.columns[0], axis=1)
#U38.to_csv("wfi_BB_U38_ESO841.res", sep='\t', header=False, index=False)

# from table 3 GOODS-S of Skelton 2014
#file_names=['vimos_u.res',
#            'R.dat',
#            'wfi_BB_U38_ESO841.res', 
#            'B_cdfs_tot.dat', 
#            'V_cdfs_tot.dat', 
#            'R_cdfs_tot.dat', 
#            'I_cdfs_tot.dat',  
#            'wfc_f435w_t77.dat',
#            'wfc_f775w_t77.dat',            
#            'wfc_f606w_t81.dat',
#            'wfc_f814w_t81.dat',
#            'wfc_f850lp_t81.dat',
#            'f140w.dat',
#            'f125w.dat',
#            'f160w.dat',
#            'isaac_j.res',
#            'isaac_h.res',
#            'isaac_ks.res',
#            'cfh8101_J.txt',
#            'cfh8302_Ks.txt',
#            'irac_tr1_2004-08-09.dat',
#            'irac_tr2_2004-08-09.dat',
#            'irac_tr3_2004-08-09.dat',
#            'irac_tr4_2004-08-09.dat']

#for i in range(0, len(file_names)-1):
#    a = pd.read_csv(file_names[i], header=None, sep='\s+')
#    b = a.drop(a.columns[0], axis=1)
#    b.to_csv(file_names[i], sep='\t', header=False, index=False)
 
GOODSS_3DHST_filt= ['vimos_u.res',
                    'R.dat',
                    'wfi_BB_U38_ESO841.res', 
                    'B_cdfs_tot.dat', 
                    'V_cdfs_tot.dat', 
                    'R_cdfs_tot.dat', 
                    'I_cdfs_tot.dat',  
                    'wfc_f435w_t77.dat',
                    'wfc_f775w_t77.dat',            
                    'wfc_f606w_t81.dat',
                    'wfc_f814w_t81.dat',
                    'wfc_f850lp_t81.dat',
                    'f140w.dat',
                    'f125w.dat',
                    'f160w.dat',
                    'isaac_j.res',
                    'isaac_h.res',
                    'isaac_ks.res',
                    'cfh8101_J.txt',
                    'cfh8302_Ks.txt',
                    'irac_tr1_2004-08-09.dat',
                    'irac_tr2_2004-08-09.dat',
                    'irac_tr3_2004-08-09.dat'#,
                    #'irac_tr4_2004-08-09.dat'
                    ]

os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/bpv079/GOODS-S 3DHST KDE Errors/')

kde_errors=['kde_e_U.pkl',
            'kde_e_R.pkl',
            'kde_e_U38.pkl', 
            'kde_e_B.pkl', 
            'kde_e_V.pkl', 
            'kde_e_Rc.pkl', 
            'kde_e_I.pkl',  
            'kde_e_F435W.pkl',
            'kde_e_F775W.pkl',            
            'kde_e_F606W.pkl',
            'kde_e_F814Wcand.pkl',
            'kde_e_F850LP.pkl',
            'kde_e_F140W.pkl',
            'kde_e_F125W.pkl',
            'kde_e_F160W.pkl',
            'kde_e_J.pkl',
            'kde_e_H.pkl',
            'kde_e_Ks.pkl',
            'kde_e_tenisJ.pkl',
            'kde_e_tenisK.pkl',
            'kde_e_IRAC1.pkl',
            'kde_e_IRAC2.pkl',
            'kde_e_IRAC3.pkl'#,
            #'kde_e_IRAC4.pkl'
            ]

kde_e_mins = [3.8e-05,
              9.300000000000001e-05,
              0.0065780000000000005,
              0.00011100000000000001,
              0.000144,
              0.000146,
              0.011942,
              0.015753,
              0.02087,
              0.012582,
              0.011251,
              0.028312999999999998,
              0.069388,
              0.039409,
              0.021195,
              0.001482,
              0.0028710000000000003,
              0.0033759999999999997,
              0.0019649999999999997,
              0.003094,
              0.000146,
              0.00015900000000000002,
              0.00054]

kde_e_50 = [0.0133, 0.0201, 0.1083, 0.0345, 0.0442, 0.0446, 0.2675, 0.0265,
            0.0377, 0.0225, 0.0261, 0.0508, 0.176 , 0.0775, 0.0616, 0.1876,
            0.3088, 0.3418, 0.2026, 0.3108, 0.2285, 0.2201, 1.1398]

kde_e_68 = [0.0152, 0.0237, 0.1236, 0.0384, 0.0497, 0.0504, 0.2971, 0.0309,
            0.0263, 0.0443, 0.0595, 0.0321, 0.2208, 0.0973, 0.0819, 0.219 ,
            0.3648, 0.4073, 0.2285, 0.3477, 0.291 , 0.2744, 1.4449]

kde_e_95 = [0.0237, 0.0400, 0.1873, 0.0596, 0.0768, 0.0783, 0.4453, 0.0521,
            0.0464, 0.0787, 0.1044, 0.0697, 0.3695, 0.1936, 0.1884, 0.3764,
            0.6073, 0.7285, 0.3521, 0.5345, 0.6241, 0.5230, 3.8152]

kde_e_99 = [0.0363, 0.0604, 0.2889, 0.0939, 0.1205, 0.124 , 0.7307, 0.0802,
            0.0765, 0.127 , 0.1672, 0.1139, 0.5902, 0.3267, 0.3121, 0.5506,
            0.9378, 1.1679, 0.5345, 0.8203, 0.9499, 0.8209, 5.2828]

mock_galax_count = 1000000

obs_samples = []
kde_samples = []
redshifts = []
params = []

obs_samples_lowSNR = []
kde_samples_lowSNR = []
redshifts_lowSNR = []
params_lowSNR = []

scale_factor = 1/10**0.44

# Load file
with open('dens_a_z1.pkl', 'rb') as f:
    dens_a = dill.load(f)

for a in range(mock_galax_count):
    
    if len(obs_samples) > 999:
      break
    
    while True:
        sample_test = dens_a.sample(1)[0]
        if (7.7000 <= sample_test[2] <= 9.700 and #lage
            0.9000 <= sample_test[0] <= 1.1000 and #z
            -1.000 <= sample_test[1] <= 3.0000 and #Av
            6.5000 <= sample_test[4] <= 12.500 and #lmass
            7.0000 <= sample_test[3] <= 10.000): #ltau            
            sample_mvkde = sample_test
            break
    
    if sample_mvkde[1] < 0.0:
        dust_sample = 0.001
    else:
        dust_sample = sample_mvkde[1]

    os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/3DHST GOODS-S Filters For BAGPIPES/')
    
    possibleDust = ['Cardelli', 'Calzetti', 'CF00']
    Dust = 'Calzetti' # using Calzetti for sampling errors test instead of random.choice(possibleDust)
    dust_av = dust_sample # Absolute attenuation in the V band in magnitudes (required)
    if(Dust == 'CF00'):
        dust = {}
        dust["n"] = random.gauss(0.3,0.1) # Power-law slope of attenuation law (required, "CF00" only)
    else:
        dust = {}

    dust["type"] = Dust  # Type of dust, e.g. 'Cardelli', 'Calzetti', 'CF00' (required)
    dust["Av"] = dust_av        # Absolute attenuation in the V band in magnitudes (required)
    #dust["eta"] = 1.           # Multiplicative factor on Av for stars in birth clouds (optional)
    #dust["qpah"] = 2.          # PAH mass fraction
    #dust["umin"] = 1.          # Lower limit of starlight intensity distribution
    #dust["gamma"] = 0.01       # Fraction of stars at umin

    nebular = {}
    nebular["logU"] = -3.0     # Log_10 of the ionization parameter (required).

    possibleSFH = ['exponential','delayed'] # Tau model e^-(t/tau) or Delayed Tau model t*e^-(t/tau)
    Star = 'delayed' # using delayed for sampling errors instead of random.choice(possibleSFH)
    z = sample_mvkde[0]
    age = (10**sample_mvkde[2])/10**9 # Time since star formation began in Gyr (required)
    masstot = (10**sample_mvkde[4])/10**9 # Log_10 total stellar mass formed in Solar masses (required)
    random_number = random.uniform(0.10, 0.90)
    mass_burst_orig = masstot*random_number
    mass_orig = masstot
    frac_mass = mass_burst_orig/mass_orig
    mass = masstot/(frac_mass + 1)
    mass_burst = masstot/(1 + (1/frac_mass))
    mass = np.log10(mass*10**9)
    mass_burst = np.log10(mass_burst*10**9)
    mass_burst_age = 0.1
    tauGyr = (10**sample_mvkde[3])/10**9 # Timescale of exponential decrease in Gyr (required)
    if(Star == 'exponential'):
        exponential = {}               # Delayed Tau model e.g. Thomas et al. (2017)
        exponential["massformed"] = mass   # Log_10 total stellar mass formed in Solar masses (required)
        exponential["metallicity"] = 1.0  # Metallicity in old Solar units, i.e. Z_sol = 0.02 (required)
        exponential["age"] = age       # Time since star formation began in Gyr (required)
        exponential["tau"] = tauGyr      # Timescale of exponential decrease in Gyr (required)
        burst = {}                           # Delta function burst
        burst["age"] = mass_burst_age        # Time since burst: Gyr
        burst["massformed"] = mass_burst  # Log_10 total stellar mass formed in Solar masses (required)
        burst["metallicity"] = 1.0  # Metallicity in old Solar units, i.e. Z_sol = 0.02 (required)
    else:
        delayed = {}               # Delayed Tau model e.g. Thomas et al. (2017)
        delayed["massformed"] = mass   # Log_10 total stellar mass formed in Solar masses (required)
        delayed["metallicity"] = 1.0  # Metallicity in old Solar units, i.e. Z_sol = 0.02 (required)
        delayed["age"] = age      # Time since star formation began in Gyr (required)
        delayed["tau"] = tauGyr     # Timescale of exponential decrease in Gyr (required)
        burst = {}                           # Delta function burst
        burst["age"] = mass_burst_age          # Time since burst: Gyr
        burst["massformed"] = mass_burst  # Log_10 total stellar mass formed in Solar masses (required)
        burst["metallicity"] = 1.0  # Metallicity in old Solar units, i.e. Z_sol = 0.02 (required)

    model_comp = {}
    model_comp["redshift"] = z
    model_comp["dust"] = dust
    model_comp["nebular"] = nebular
    model_comp["t_bc"] = 0.01 #recently born stars are enshrouded in dust, so this would be for a two component dust model
    #if the calzetti law can handle a two component dust model, then include this for younger stars
    #model_comp["veldisp"] = 0.

    if(Star == 'exponential'):
        model_comp['burst'] = burst
        model_comp['exponential'] = exponential
    else:
        model_comp['burst'] = burst
        model_comp['delayed'] = delayed

    model = pipes.model_galaxy(model_comp, filt_list=GOODSS_3DHST_filt, phot_units="mujy")
    
    dm = np.log10(np.sum(model.sfh.ceh.grid)/10**9) - np.log10(np.sum(model.sfh.live_frac_grid*model.sfh.ceh.grid)/10**9)
    
    masstot = masstot + dm
    masstot = (10**sample_mvkde[4])/10**9
    mass_burst_orig = masstot*random_number
    mass_orig = masstot
    frac_mass = mass_burst_orig/mass_orig
    mass = masstot/(frac_mass + 1)
    mass_burst = masstot/(1 + (1/frac_mass))
    mass = np.log10(mass*10**9)
    mass_burst = np.log10(mass_burst*10**9)
    delayed["massformed"] = mass
    burst["massformed"] = mass_burst  # Log_10 total stellar mass formed in Solar masses (required)    
    
    model.update(model_comp)
    
    mock_phot = model.photometry
        
    os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/bpv079/GOODS-S 3DHST KDE Errors/')
    
    # Load from file
    kde_samples_init = []
    for i in range(len(kde_errors)):
        with open(kde_errors[i], 'rb') as file:
            kde = pickle.load(file)
            while True:
                sample_test = kde.sample(1)[0][0]

                if kde_e_mins[i] < sample_test < kde_e_95[i]:
                    sample_kde = sample_test
                    break
            kde_samples_init.append(sample_kde*scale_factor)
        
    params_list = [dust_av, z, age, mass, tauGyr, mass_burst, mass_burst_age, random_number]

    if mock_phot[14]/np.array(kde_samples_init[14]) > 0.:
        obs_samples.append(mock_phot.tolist())
        kde_samples.append(kde_samples_init)
        params.append(params_list)
        redshifts.append(np.array(z).tolist())
    else:
        obs_samples_lowSNR.append(mock_phot.tolist())
        kde_samples_lowSNR.append(kde_samples_init)
        params_lowSNR.append(params_list)
        redshifts_lowSNR.append(np.array(z).tolist())
    
goodss_3dhst_mock = []
for i, j in itertools.product(range(0, len(obs_samples)), range(0, len(kde_samples[0]))):
    goodss_3dhst_mock.append(obs_samples[i][j])
    goodss_3dhst_mock.append(kde_samples[i][j])
    
galax_each_row = [goodss_3dhst_mock[x:x+2*len(obs_samples[0])] for x in range(0, 
                                    len(goodss_3dhst_mock), 2*len(obs_samples[0]))]
for i in range(0, len(redshifts)):
    galax_each_row[i].insert(0, redshifts[i])

for i in range(0, len(params)):
    galax_each_row[i].extend((params[i]))
   
goodss_3dhst_mock_header = []
for i in range(len(kde_errors)):
    goodss_3dhst_mock_header.append(GOODSS_3DHST_filt[i])
    goodss_3dhst_mock_header.append(kde_errors[i])
goodss_3dhst_mock_header.insert(0, 'redshift')
goodss_3dhst_mock_header.extend(('dust_av', 'z', 'age', 'mass', 'tauGyr', 'mass_burst', 'mass_burst_age', 'random_number'))

goodss_3dhst_mock_df = pd.DataFrame(galax_each_row, columns=goodss_3dhst_mock_header)

goodss_3dhst_mock_df.replace(0, np.nan, inplace=True)
    
kde_samples2 = []

os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/bpv079/GOODS-S 3DHST KDE Errors/')

for a in range(len(obs_samples)):
    kde_samples_init2 = []
    for i in range(len(kde_errors)):
        with open(kde_errors[i], 'rb') as file:
            kde = pickle.load(file)
            while True:
                sample_test = kde.sample(1)[0][0]

                if kde_e_mins[i] < sample_test < kde_e_95[i]:
                    sample_kde = sample_test
                    break
            kde_samples_init2.append(sample_kde*scale_factor)
        
    kde_samples2.append(kde_samples_init2)

goodss_3dhst_mock2 = []
for i, j in itertools.product(range(0, len(obs_samples)), range(0, len(kde_samples2[0]))):
    goodss_3dhst_mock2.append(obs_samples[i][j])
    goodss_3dhst_mock2.append(kde_samples2[i][j])
    
galax_each_row2 = [goodss_3dhst_mock2[x:x+2*len(obs_samples[0])] for x in range(0, 
                                    len(goodss_3dhst_mock2), 2*len(obs_samples[0]))]
    
for i in range(0, len(redshifts)):
    galax_each_row2[i].insert(0, redshifts[i])
    
for i in range(0, len(params)):
    galax_each_row2[i].extend((params[i]))

goodss_3dhst_mock_df2 = pd.DataFrame(galax_each_row2, columns=goodss_3dhst_mock_header)

flux = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45]
errors = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]
goodss_3dhst_mock_df_flux = goodss_3dhst_mock_df2.iloc[:,flux]
goodss_3dhst_mock_df_errors = goodss_3dhst_mock_df2.iloc[:,errors]

random_numbers = np.random.randint(2, size=len(flux)*len(obs_samples))
random_numbers[random_numbers == 0] = -1
random_numbers = random_numbers.reshape((len(obs_samples), len(flux)))
random_numbers_df = pd.DataFrame(random_numbers, columns=kde_errors)

goodss_3dhst_mock_df_flux_adj = []
for i, j in itertools.product(range(0, len(obs_samples)), range(0, len(flux))):
    goodss_3dhst_mock_df_flux_adj.append((goodss_3dhst_mock_df_flux.iloc[i][j] + goodss_3dhst_mock_df_errors.iloc[i][j]*random_numbers_df.iloc[i][j]))
goodss_3dhst_mock_df_flux_adj = pd.DataFrame([goodss_3dhst_mock_df_flux_adj[i:i + len(flux)] for i in xrange(0, len(goodss_3dhst_mock_df_flux_adj), len(flux))])

goodss_3dhst_mock_df_flux_adj.columns = GOODSS_3DHST_filt

goodss_3dhst_mock_df.update(goodss_3dhst_mock_df_flux_adj)

os.chdir('/home/andy/Downloads/BAGPIPES/Mock Data Generation/bpv079/')

pkl_filename1 = "goodss_3dhst_mock_df_09-07-19_z1_s1.pkl"  
with open(pkl_filename1, 'wb') as file1:  
    pickle.dump(goodss_3dhst_mock_df, file1)