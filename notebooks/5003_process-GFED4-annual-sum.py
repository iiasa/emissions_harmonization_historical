# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Process GFED4 annual totals
#
# Process data from [GFED4](https://www.globalfiredata.org/related.html)
# into annual totals.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import tqdm.auto
import xarray as xr

from emissions_harmonization_historical.constants_5000 import GFED4_ANNUAL_SUM_OUTPUT_DIR, GFED4_RAW_PATH
from emissions_harmonization_historical.gfed import (
    read_cell_area,
    read_year,
)

# %% [markdown]
# ## Process

# %% [markdown]
# Do this year by year
# because dask explodes with the complexity of reading the hdf5 files.

# %%
GFED4_ANNUAL_SUM_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
cell_area = read_cell_area(next(iter(GFED4_RAW_PATH.glob("*.hdf5")))).assign_attrs(dict(unit="m2"))
cell_area.to_netcdf(GFED4_ANNUAL_SUM_OUTPUT_DIR / "cell_area.nc")
# cell_area

# %%
res_l = []
for filename in tqdm.auto.tqdm(sorted(GFED4_RAW_PATH.glob("*.hdf5"))):
    emissions_yr = read_year(filename)

    emissions_yr_dry_matter = emissions_yr["DM"].compute()
    emissions_yr_dry_matter.attrs.update(dict(unit="kg DM m-2 / month"))

    res_l.append(emissions_yr_dry_matter)

emissions_dry_matter = xr.concat(res_l, dim="time")
emissions_dry_matter_per_yr = emissions_dry_matter.groupby("time.year").sum().assign_attrs(dict(unit="kg DM / yr"))
# emissions_dry_matter_per_yr

# %%
time_bounds_start = emissions_yr_dry_matter["time"].values
time_bounds_end = np.hstack([time_bounds_start[1:], type(time_bounds_start[0])(time_bounds_start[-1].year + 1, 1, 1)])
time_step_size = [v.total_seconds() for v in time_bounds_end - time_bounds_start]
time_step_size

# %%
emissions_yr_dry_matter

# %%
emissions_dry_matter_per_yr.to_netcdf(GFED4_ANNUAL_SUM_OUTPUT_DIR / "emissions_dry_matter_per_yr.nc")
