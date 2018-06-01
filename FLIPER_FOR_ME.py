#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:17:00 2018

@author: Lisa Bugnet
@contact: lisa.bugnet@cea.fr

This python code is made for the estimation of surface gravity of Kepler Solar-type
oscillating targets with 0.3 < logg < 4.6 dex.
The user should first use the FLIPER class to calculate FliPer values
from 0.2,0.7,7,20 and 50 muHz (see Bugnet et al.,2018) (see CALLING SEQUENCE at the end of this code).
These values are the parameters needed for the machine learning along with the effective temperature.
The Random Forest regressor is already trained and stored in the "ML_training"
file that should be download before running this code.
The estimation of surface gravity should be made by the use of the ML class
(see CALLING SEQUENCE at the end of this code).

What you need:
- The power density spectrum of the star filtered with a 20 days high pass filter.
- The power density spectrum of the star filtered with a 80 days high pass filter.
- The Kepler magnitude of the star
- The effective temperature of the star (from Mathur et al., 2017)
- The ??? file containing the training of the Random Forest, to be dowload on https://????????

A calling example is reported at the end of the code
"""

from astropy.io import fits
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import os, os.path


class CONVERT:
    def __init__(self):
        self.psd_path=[]
        self.psd=[[],[]]
        self.kic=[]

    def PSD_PATH_TO_PSD(self, star_path_psd):

        hdulist=fits.open(star_path_psd)
        hdu=hdulist[0]
        freq_tab=hdu.data[:,0]
        power_tab=hdu.data[:,1]
        star_tab_psd=np.column_stack((freq_tab,power_tab))

        self.psd += list(map(list, zip(*star_tab_psd)))
        return star_tab_psd


class DATA_PREPARATION:

    def __init__(self):
        if __name__ == "__main__":
            #"constructor"
            self.nom = "FliPer"
            self.kp=""
            self.freq=""
            self.power=""

    def GET_KEPMAG(self, star_path_psd):
        kepmag = fits.getval(star_path_psd, 'KEPMAG',0)
        if __name__ == "__main__":
            self.kp += str(kepmag)
        return kepmag

    def GET_ARRAY(self, star_tab_psd):
        freq_tab=star_tab_psd[:,0]
        power_tab=star_tab_psd[:,1]
        if __name__ == "__main__":
            self.freq += str(freq_tab)
            self.power += str(power_tab)

        return freq_tab, power_tab

    def MAG_COR_KEP(star_tab_psd,  kepmag):
        #-----------------------------------------------------------------------
        #function that computes photon noise from kepler magnitude
        data_arr_freq=DATA_PREPARATION().GET_ARRAY(star_tab_psd)[0]
        #Jenkins et al., 2012
        c = 3.46 * 10**(0.4*(12.-kepmag)+8)
        siglower = np.sqrt( c + 7e6 * max([1,kepmag/14.])**4. ) / c
        siglower = siglower * 1e6  # convert in ppm
        dt = 1./(2*data_arr_freq[int(len(data_arr_freq)-1.)])# en sec. #IL Y A UNE ERREUR DANS LE CODE IDL!!!!!!
        siglower = siglower**2.*2*dt*1.e-6 #unit ppm^2/muHz
        return siglower

    def REGION(star_tab_psd,inic,fin):
        #-----------------------------------------------------------------------
        #function that calculates the average power in a given frequency range
        x=np.float64(DATA_PREPARATION().GET_ARRAY(star_tab_psd)[0]*1e6) # convert frequencies in muHz
        y=np.float64(DATA_PREPARATION().GET_ARRAY(star_tab_psd)[1])
        ys=y[np.where((x >= inic) & (x <= fin))]
        average=np.mean(ys)
        return average

    def APODIZATION(star_tab_psd):
        #-----------------------------------------------------------------------
        #function that corrects the spectra from apodization
        power=star_tab_psd[:,1]
        freq=star_tab_psd[:,0]
        nq=max(freq)
        nu=np.sin(np.pi/2.*freq/nq) / (np.pi/2.*freq/nq)
        power=power/(nu**2)
        star_tab_psd[:,1]=power
        star_tab_psd[:,0]=freq
        return star_tab_psd

class FLIPER:
#"""Class defining the FliPer"""

    def __init__(self):
        self.nom = "FliPer"
        self.id=[]
        self.fp07=[]
        self.fp7=[]
        self.fp20=[]
        self.fp02=[]
        self.fp50=[]

    def Fp_20_days(self, star_tab_psd_20, kepmag):
        #-----------------------------------------------------------------------
        #Compute FliPer value from 0.2 muHz to Nyquist with 80 days filtered data
        star_tab_psd_20=DATA_PREPARATION.APODIZATION(star_tab_psd_20)
        fin_20=(np.amax(DATA_PREPARATION().GET_ARRAY(star_tab_psd_20)[0]*1e6))
        noise=DATA_PREPARATION.MAG_COR_KEP(star_tab_psd_20, kepmag)
        Fp07_val=DATA_PREPARATION.REGION(star_tab_psd_20, 0.7, fin_20) - noise
        Fp7_val=DATA_PREPARATION.REGION(star_tab_psd_20, 7, fin_20) - noise
        Fp20_val=DATA_PREPARATION.REGION(star_tab_psd_20, 20, fin_20) - noise
        Fp50_val=DATA_PREPARATION.REGION(star_tab_psd_20, 50, fin_20) - noise
        self.fp07.append(Fp07_val)
        self.fp7.append(Fp7_val)
        self.fp20.append(Fp20_val)
        self.fp50.append(Fp50_val)
        print(self.fp07[0])
        print(self.fp7[0])
        print(self.fp20[0])
        print(self.fp50[0])

        return self


    def Fp_80_days(self, star_tab_psd_80, kepmag):
        #-----------------------------------------------------------------------
        #Compute FliPer value from 0.7, 7, 20, and 50 muHz to Nyquist with 20 days filtered data
        star_tab_psd_80=DATA_PREPARATION.APODIZATION(star_tab_psd_80)
        fin_80=(np.amax(DATA_PREPARATION().GET_ARRAY(star_tab_psd_80)[0]*1e6))
        noise=DATA_PREPARATION.MAG_COR_KEP(star_tab_psd_80, kepmag)
        Fp02_val=DATA_PREPARATION.REGION(star_tab_psd_80, 0.2, fin_80) - noise
        self.fp02.append(Fp02_val)
        print(self.fp02[0])

        return self


class ML:
    def __init__(self):
        self.nom = "ML estimate"
        self.logg=[]

    def testing(self, teff, lnF02, lnF07, lnF7, lnF20, lnF50, path_to_training_file):
        #-----------------------------------------------------------------------
        #Estimation of logg with machine learning (training given by the path_to_training_file)
        inputs=[[teff,lnF02,lnF07,lnF7,lnF20,lnF50]]
        with open(path_to_training_file, 'rb') as f:
            rf = cPickle.load(f)
        logg_output = rf.predict(inputs)
        print(logg_output)
        self.logg.append(logg_output)
        return logg_output



# #-------------------------------------------------------------------------------
# # CALLING SEQUENCE
# #-------------------------------------------------------------------------------

# #1) Star parameters
# star_tab_psd_20=??? #format: power=star_tab_psd_20[:,1], freq=star_tab_psd_20[:,0]
# star_tab_psd_80=??? #format: power=star_tab_psd_80[:,1], freq=star_tab_psd_80[:,0]
# kepmag=???
# teff=???
# path_to_training_file=???

# #-------------------------------------------------------------------------------

# #2) FliPer calculation
# Fliper_20_d=FLIPER().Fp_20_days(star_tab_psd_20, kepmag)
# Fliper_80_d=FLIPER().Fp_80_days(star_tab_psd_80, kepmag)
# lnF07=np.log10(Fliper_20_d.fp07[0])
# lnF7=np.log10(Fliper_20_d.fp7[0])
# lnF20=np.log10(Fliper_20_d.fp20[0])
# lnF50=np.log10(Fliper_20_d.fp50[0])
# lnF02=np.log10(Fliper_80_d.fp02[0])

# #-------------------------------------------------------------------------------

# #3) logg estimation
# logg=ML().testing(teff, lnF02, lnF07, lnF7, lnF20, lnF50, path_to_training_file)

# #-------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------















#EXAMPLE FOR ME
#with open('/Users/lbugnet/WORK/pythongoogletest/FLIPER_TRY1/list_path_psd.txt', 'r') as myfile:#20 days paths
#    paths_20=myfile.read().splitlines()
#with open('/Users/lbugnet/WORK/pythongoogletest/FLIPER_TRY1/list_path_psd.txt', 'r') as myfile:#80 days paths
#    paths_80=myfile.read().splitlines()

star_path_psd_80='/Volumes/TEMP/RG_DR25/K001/RESULTS_KADACS_COARSE_CheckSTATUS_filt_polfitseg0_80.0000d_ppm0_inpaint20/LC_CORR_FILT_INP/kplr001296068_43_COR_PSD_filt_inp.fits'
star_path_psd_20='/Volumes/TEMP/RG_DR25/K001/RESULTS_KADACS_COARSE_CheckSTATUS_filt_polfitseg960.000_20.0000d_ppm0_inpaint20/LC_CORR_FILT_INP/kplr001296068_43_COR_PSD_filt_inp.fits'
#star_path_psd_20='/Volumes/TEMP/RG_DR25/K001/RESULTS_KADACS_COARSE_CheckSTATUS_filt_polfitseg960.000_20.0000d_ppm0_inpaint20/LC_CORR_FILT_INP/kplr001161447_16_COR_PSD_filt_inp.fits'
#star_path_psd_80='/Volumes/TEMP/RG_DR25/K001/RESULTS_KADACS_COARSE_CheckSTATUS_filt_polfitseg0_80.0000d_ppm0_inpaint20/LC_CORR_FILT_INP/kplr001161447_16_COR_PSD_filt_inp.fits'
#star_path_psd_20=paths_20[1]
#star_path_psd_80=paths_80[1]
kepmag=DATA_PREPARATION().GET_KEPMAG(star_path_psd_20)
star_tab_psd_20=CONVERT().PSD_PATH_TO_PSD(star_path_psd_20)
# plt.plot(star_tab_psd_20[:,0], star_tab_psd_20[:,1])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

star_tab_psd_80=CONVERT().PSD_PATH_TO_PSD(star_path_psd_80)

#teff=4784.09726#test
teff=4615.300681
Fliper_20_d=FLIPER().Fp_20_days(star_tab_psd_20, kepmag)
lnF07=np.log10(Fliper_20_d.fp07[0])
lnF7=np.log10(Fliper_20_d.fp7[0])
lnF20=np.log10(Fliper_20_d.fp20[0])
lnF50=np.log10(Fliper_20_d.fp50[0])

Fliper_80_d=FLIPER().Fp_80_days(star_tab_psd_80, kepmag)
lnF02=np.log10(Fliper_80_d.fp02[0])
#calculate fliper(s):
logg=ML().testing(teff,lnF02,lnF07,lnF7,lnF20,lnF50, '/Users/lbugnet/WORK/pythongoogletest/FLIPER_TRY1/uploads/ML_logg_training')
print(logg)
