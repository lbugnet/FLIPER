#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:17:00 2018

@author: Lisa Bugnet
@contact: lisa.bugnet@cea.fr
This code is the property of L. Bugnet (please see and cite Bugnet et al.,2018).

This python code is made for the estimation of surface gravity of Kepler
Solar-type oscillating targets with 0.1 < logg < 4.5 dex.

The user should first use the FLIPER class to calculate FliPer values
from 0.2,0.7,7,20 and 50 muHz (see Bugnet et al.,2018)
(see CALLING SEQUENCE at the end of this code).
These values are the parameters needed by the machine learning Random Forest
(along with the effective temperature and the Kepler magnitude of the star).

The Random Forest regressors are already trained and stored in the
"ML_logg_training_paper" and "ML_logg_training_paper_numax" files to estimate
logg or numax. They should be download on GitHub (LINK?????) before running this code.
The estimation of surface gravity should be made by the use of the "ML" class
(see CALLING SEQUENCE at the end of this code).

What you need:
- The power density spectrum of the star filtered with a 20 days high pass filter.
- The power density spectrum of the star filtered with a 80 days high pass filter.
- The Kepler magnitude of the star
- The effective temperature of the star (from Mathur et al., 2017 for instance)
- The "ML_logg_training_paper" file containing the training of the Random Forest,
    to be dowload on https://????????

A calling example is reported at the end of the code.
"""

from astropy.io import fits
import numpy as np
import _pickle as cPickle
import os, os.path
from math import *
import pandas as pd


class FLIPER:
    def __init__(self):
        self.nom        =   "FliPer"
        self.fp07       =   []
        self.fp7        =   []
        self.fp20       =   []
        self.fp02       =   []
        self.fp50       =   []
        self.sig_fp07   =   []
        self.sig_fp7    =   []
        self.sig_fp20   =   []
        self.sig_fp02   =   []
        self.sig_fp50   =   []

    def Fp_20_days(self, star_tab_psd_20, kepmag):
        """
        Compute FliPer value from 0.7, 7, 20, and 50 muHz to Nyquist with 20 days filtered data.
        """
        star_tab_psd_20 =   DATA_PREPARATION().APODIZATION(star_tab_psd_20)
        end_20          =   (np.amax(DATA_PREPARATION().GET_ARRAY(star_tab_psd_20)[0]*1e6))
        noise           =   DATA_PREPARATION().MAG_COR_KEP(star_tab_psd_20, kepmag)
        Fp07_val        =   DATA_PREPARATION().REGION(star_tab_psd_20, 0.7, end_20) - noise
        Fp7_val         =   DATA_PREPARATION().REGION(star_tab_psd_20, 7, end_20) - noise
        Fp20_val        =   DATA_PREPARATION().REGION(star_tab_psd_20, 20, end_20) - noise
        Fp50_val        =   DATA_PREPARATION().REGION(star_tab_psd_20, 50, end_20) - noise
        sig_Fp07        =   self.Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 0.7, end_20))
        sig_Fp7         =   self.Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 7, end_20))
        sig_Fp20        =   self.Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 20, end_20))
        sig_Fp50        =   self.Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 50, end_20))

        self.fp07.append(Fp07_val)
        self.fp7.append(Fp7_val)
        self.fp20.append(Fp20_val)
        self.fp50.append(Fp50_val)
        self.sig_fp07.append(sig_Fp07)
        self.sig_fp7.append(sig_Fp7)
        self.sig_fp20.append(sig_Fp20)
        self.sig_fp50.append(sig_Fp50)
        return self


    def Fp_80_days(self, star_tab_psd_80, kepmag):
        """
        Compute FliPer value from 0.2 muHz to Nyquist with 80 days filtered data.
        """
        star_tab_psd_80 =   DATA_PREPARATION().APODIZATION(star_tab_psd_80)
        end_80          =   (np.amax(DATA_PREPARATION().GET_ARRAY(star_tab_psd_80)[0]*1e6))
        noise           =   DATA_PREPARATION().MAG_COR_KEP(star_tab_psd_80, kepmag)
        Fp02_val        =   DATA_PREPARATION().REGION(star_tab_psd_80, 0.2, end_80) - noise
        sig_Fp02        =   self.Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_80, 0.2, end_80))

        self.fp02.append(Fp02_val)
        self.sig_fp02.append(sig_Fp02)
        return self

    def Fp_error(self, power): #GUY
        """
        Compute errors on FliPer values du to noise.
        """
        n           =   50                                                                      #   rebin of the spectra to have normal distribution on the uncertainties
        Ptmp        =   np.array([np.sum(power[i*n:(i+1)*n]) for i in range(int(len(power)/n))])#   power on the rebin
        Ptot        =   np.sum(Ptmp)                                                            #   total power
        sig_Ptot    =   (np.sum((2 * Ptmp / 2 / n * n**0.5)**2))**0.5                           #   uncertainties on total power
        error_Fp    =   ((sig_Ptot / len(power))**2)**0.5
        return error_Fp

    def RANDOM_PARAMS(self, param, error_param):
        param_tab   =  np.full((100),param) +   error_param * np.random.normal(0,1, 100)
        return param_tab

class DATA_PREPARATION:

    def __init__(self):
        self.nom        =   "DATA PREPARATION"

    def PSD_PATH_TO_PSD(self, star_path_psd):
        """
        Function that takes PSD from PSD_path.
        """
        hdulist     =   fits.open(star_path_psd)
        hdu         =   hdulist[0]
        freq_tab    =   hdu.data[:,0]
        power_tab   =   hdu.data[:,1]
        star_tab_psd=   np.column_stack((freq_tab,power_tab))
        return star_tab_psd

    def MAG_COR_KEP(self, star_tab_psd,  kepmag):
        """
        Function that computes photon noise from kepler magnitude following Jenkins et al., 2012.
        """
        data_arr_freq   =   self.GET_ARRAY(star_tab_psd)[0]
        c               =   3.46 * 10**(0.4*(12.-kepmag)+8)
        siglower        =   np.sqrt( c + 7e6 * max([1,kepmag/14.])**4. ) / c
        siglower        =   siglower * 1e6                                   #  [ppm]
        dt              =   1./(2*0.000278)                                  #  [sec]
        siglower        =   siglower**2.*2*dt*1.e-6                          #  [ppm^2/muHz]
        return siglower

    def GET_ARRAY(self, star_tab_psd):
        freq_tab        =   star_tab_psd[:,0]
        power_tab       =   star_tab_psd[:,1]
        return freq_tab, power_tab

    def REGION(self,star_tab_psd,inic,end):
        """
        Function that calculates the average power in a given frequency range.
        """
        x       =   np.float64(self.GET_ARRAY(star_tab_psd)[0]*1e6) # convert frequencies in muHz
        y       =   np.float64(self.GET_ARRAY(star_tab_psd)[1])
        ys      =   y[np.where((x >= inic) & (x <= end))]
        average =   np.mean(ys)
        return average

    def CUT_SPECTRA(self,star_tab_psd,inic,end):
        """
        Function that returns the power contained in a given frequency range.
        """
        x       =   np.float64(self.GET_ARRAY(star_tab_psd)[0]*1e6) # convert frequencies in muHz
        y       =   np.float64(self.GET_ARRAY(star_tab_psd)[1])
        ys      =   y[np.where((x >= inic) & (x <= end))]
        return ys

    def APODIZATION(self,star_tab_psd):
        """
        Function that corrects the spectra from apodization.
        """
        power   =   star_tab_psd[:,1]
        freq    =   star_tab_psd[:,0]
        nq      =   max(freq)
        nu      =   np.sin(np.pi/2.*freq/nq) / (np.pi/2.*freq/nq)
        power   =   power/(nu**2)
        star_tab_psd[:,1]   =   power
        star_tab_psd[:,0]   =   freq
        return star_tab_psd



class ML:
    def __init__(self):
        self.nom = "ML estimate"
        self.logg=[]

    def PREDICTION(self, Teff, KP,F02, F07, F7, F20, F50,  path_to_training_file):
        """
        Estimation of logg with machine learning (training given by 'ML_logg_training_paper' to be dowload in GitHub).
        """
        listing         =   {'Teff': Teff, 'KP': KP,'lnF02': self.CONVERT_TO_LOG(F02), 'lnF07': self.CONVERT_TO_LOG(F07), 'lnF7': self.CONVERT_TO_LOG(F7), 'lnF20': self.CONVERT_TO_LOG(F20), 'lnF50': self.CONVERT_TO_LOG(F50)}
        df              =   pd.DataFrame(data=listing)
        columnsTitles   =   ['Teff', 'KP', 'lnF02', 'lnF07', 'lnF7', 'lnF20', 'lnF50']
        df              =   df.reindex(columns=columnsTitles)
        X               =   df.values

        with open(path_to_training_file, 'rb') as f:
            rf  =   cPickle.load(f)

        logg_estimated  =   rf.predict(X)
        self.logg.append(logg_estimated)
        return logg_estimated

    def CONVERT_TO_LOG(self, param):
        return np.log10(param)


"""
-------------------------------------------------------------------------------
 CALLING SEQUENCE
-------------------------------------------------------------------------------

ALL NEEDED INFORMATIONS:
"""
#Paths to PSD fits files computed from light curves filtered with 20 and 80 days
psd_path_20             =   '/???/???'
psd_path_80             =   '/???/???'

#Path to trained random forest (to be dowloaded on GitHub)
PATH_TO_TRAINING_FILE_LOGG   =   '/???/ML_logg_training_paper'
PATH_TO_TRAINING_FILE_NUMAX  =   '/???/ML_numax_training_paper'
#Give star parameters
kepmag          =   12.349
teff            =   4750.0698938311934
error_teff      =   55.844606659634337


"""
Open data from PSD_paths from 20 and 80 days light curves.
"""
star_tab_psd_20 =   DATA_PREPARATION().PSD_PATH_TO_PSD(psd_path_20)
star_tab_psd_80 =   DATA_PREPARATION().PSD_PATH_TO_PSD(psd_path_80)


"""
Calculate FliPer values.
"""
Fliper_20_d =   FLIPER().Fp_20_days(star_tab_psd_20, kepmag)
Fliper_80_d =   FLIPER().Fp_80_days(star_tab_psd_80, kepmag)
Fp02        =   Fliper_80_d.fp02[0]
Fp07        =   Fliper_20_d.fp07[0]
Fp7         =   Fliper_20_d.fp7[0]
Fp20        =   Fliper_20_d.fp20[0]
Fp50        =   Fliper_20_d.fp50[0]
Teff        =   teff
KP          =   kepmag

"""
Compute 100 stars per star by taking into account uncertainties on parameters.   (OPTIONNAL, ONLY TO REPORDUCE PAPER)
"""
Fp02    =   FLIPER().RANDOM_PARAMS(Fliper_80_d.fp02[0], Fliper_80_d.sig_fp02[0])
Fp07    =   FLIPER().RANDOM_PARAMS(Fliper_20_d.fp07[0], Fliper_20_d.sig_fp07[0])
Fp7     =   FLIPER().RANDOM_PARAMS(Fliper_20_d.fp7[0] , Fliper_20_d.sig_fp7[0] )
Fp20    =   FLIPER().RANDOM_PARAMS(Fliper_20_d.fp20[0], Fliper_20_d.sig_fp20[0])
Fp50    =   FLIPER().RANDOM_PARAMS(Fliper_20_d.fp50[0], Fliper_20_d.sig_fp50[0])
Teff    =   FLIPER().RANDOM_PARAMS(teff, error_teff)
KP      =   np.full((100),kepmag)                                                      #no uncertainties on KP

"""
Estimation of surface gravity and/or numax from the "ML_logg_training_paper" or "ML_logg_training_paper_numax" file.
"""
logg=ML().PREDICTION(Teff, KP, Fp02, Fp07, Fp7, Fp20, Fp50, PATH_TO_TRAINING_FILE_LOGG)
numax=10**(ML().PREDICTION(Teff, KP, Fp02, Fp07, Fp7, Fp20, Fp50, PATH_TO_TRAINING_FILE_NUMAX))
