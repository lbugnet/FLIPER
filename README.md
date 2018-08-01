# FLIPER SURFACE GRAVITY and NUMAX estimation

author: Lisa Bugnet
contact: lisa.bugnet@cea.fr
This repository is the property of L. Bugnet (please see and cite Bugnet et al.,2018).

The FLIPER python code is made for the estimation of surface gravity of Kepler
Solar-type oscillating targets with 0.1 < logg < 4.5 dex.

The user should first use the FLIPER class to calculate FliPer values
from 0.2,0.7,7,20 and 50 muHz (see Bugnet et al.,2018)
(see CALLING SEQUENCE at the end of this code).
These values are the parameters needed by the machine learning Random Forest
(along with the effective temperature and the Kepler magnitude of the star).

The Random Forest regressors are already trained and stored in the
"ML_logg_training_paper" and "ML_logg_training_paper_numax" files to estimate
logg or numax. They should be download on this GitHub repository before running the FLIPER code.
The estimation of surface gravity should be made by the use of the "ML" class
(see CALLING SEQUENCE at the end of the code).

What you need:
- The power density spectrum of the star filtered with a 20 days high pass filter.
- The power density spectrum of the star filtered with a 80 days high pass filter.
- The Kepler magnitude of the star
- The effective temperature of the star (from Mathur et al., 2017 for instance)
- The "ML_logg_training_paper" and "ML_logg_training_paper_numax" files containing the training of the Random Forest algorithms, to be dowload on GitHub.

A calling example is reported at the end of the FLIPER code.
