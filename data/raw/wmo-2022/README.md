Data from [the 2022 WMO Ozone Assessment](https://csl.noaa.gov/assessments/ozone/2022/).

Provided by the authors only for this purpose as the data are not available elsewhere.
If you want to use the data for something else,
please contact the 2022 WMO Ozone Assessment authors directly
before using the data so you can make sure it is appropriate for your purposes.

The file `wmo-2022-cmip7-mixing-ratios-inverse-emissions.csv`
is created in https://github.com/climate-resource/cmip7-scenariomip-ghg-concentrations/blob/main/notebooks/1020_calculate-inverse-emissions.py
before being copied here.
The backstory is that the WMO 2022 team provided recommended mixing ratios for CMIP7
which are similar, but not identical to the WMO 2022 ones.
These are then inverted to create the emissions.
For CMIP7, the concentrations are prescribed
(there are no emissions-driven runs for gases covered under the Montreal Protocol)
so these emissions are of secondary importance
(particularly the fact that a perfect inversion is not possible).
They are useful as a first-order estimate of the emissions consistent with the concentrations,
which is all we need and may be useful in other exercises.
