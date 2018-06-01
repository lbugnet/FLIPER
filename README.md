# FLIPER SURFACE GRAVITY ESTIMATION

This python code is made for the estimation of surface gravity of Kepler Solar-type
oscillating targets with 0.3 < logg < 4.6 dex.
The user should first use the FLIPER class to calculate FliPer values
from 0.2,0.7,7,20 and 50 muHz (see Bugnet et al.,2018) (see CALLING SEQUENCE at the end of this code).
These values are the parameters needed for the machine learning along with the effective temperature.
The Random Forest regressor is already trained and stored in the "ML_logg_training"
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
