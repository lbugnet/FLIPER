# FLIPER SURFACE GRAVITY and NUMAX ESTIMATION

This python code is made for the estimation of surface gravity of Kepler Solar-type
oscillating targets with 0.1 < logg < 4.5 dex.
The user should first use the FLIPER class to calculate FliPer values from 0.2,0.7,7,20 and 50 muHz (see Bugnet et al.,2018) (see CALLING SEQUENCE at the end of this code).
These values are the parameters needed for the machine learning along with the effective temperature and the kepler Magnitude of each star.
The Random Forest regressor is already trained and stored in the "ML_logg_training" file that should be download before running the code.
The estimation of surface gravity should be made by the use of the ML class
(see CALLING SEQUENCE at the end of this code).

What you need:
- The power density spectrum of the star (light curve filtered with a 20 days high pass filter).
- The power density spectrum of the star (light curve filtered with a 80 days high pass filter).
- The Kepler magnitude of the star
- The effective temperature of the star (from Mathur et al., 2017 for instance)
- The ??? file containing the training of the Random Forest, to be dowloaded.

A calling example is reported at the end of the code
