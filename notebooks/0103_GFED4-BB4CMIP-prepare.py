# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from collections import defaultdict

import xarray as xr

from emissions_harmonization_historical.constants import DATA_ROOT

# %%
raw_data_path = DATA_ROOT / "national/gfed-bb4cmip/data_raw/data"
raw_data_path

# %% [markdown]
# Group raw data into variable groups which can be used for processing.

# %%
bb4cmip_files = list(raw_data_path.rglob("*.nc"))
bb4cmip_file_groups = defaultdict(list)

for file in bb4cmip_files:
    variable = file.name.split("_")[0]
    bb4cmip_file_groups[variable].append(file)

bb4cmip_file_groups.keys()

# %% [markdown]
# An example of loading data.

# %%
cell_area = xr.open_mfdataset(bb4cmip_file_groups["gridcellarea"])
cell_area

# %%
gas = "BC"
gas_ds = xr.open_mfdataset(bb4cmip_file_groups["BC"])
gas_ds

# %%
global_sum_emissions = (gas_ds["BC"] * cell_area["gridcellarea"]).sum(["latitude", "longitude"]).compute()
global_sum_emissions

# %%
global_sum_emissions.plot()

# %%
global_sum_emissions.groupby("time.year").mean().plot()
