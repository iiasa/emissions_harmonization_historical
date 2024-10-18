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

import cf_xarray.units
import matplotlib.pyplot as plt
import numpy as np
import pint_xarray
import scmdata
import xarray as xr

from emissions_harmonization_historical.constants import DATA_ROOT

# %%
pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

# %%
raw_data_path = DATA_ROOT / "national/gfed-bb4cmip/data_raw/data"
raw_data_path

# %%
rcmip_data_path = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"

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
unit_target = "Mt / yr"
unit_label = "Mt BC / yr"  # Have to be careful here, depends on the species
rcmip_gas = "BC"

# gas = "CO"
# unit_target = "Mt / yr"
# unit_label = "Mt CO / yr"
# rcmip_gas = "CO"

# # TODO: check. I don't think we used this last time so we may want to ignore this again.
# gas = "CO2"
# unit_target = "Mt / yr"
# unit_label = "Mt CO2 / yr"
# rcmip_gas = "CO2"

# gas = "CH4"
# unit_target = "Mt / yr"
# unit_label = "Mt CH4 / yr"
# rcmip_gas = "CH4"

# # TODO: check. I don't think we used this last time.
# gas = "N2O"
# unit_target = "Mt / yr"
# unit_label = "Mt N2O / yr"
# rcmip_gas = "N2O"

# gas = "OC"
# unit_target = "Mt / yr"
# unit_label = "Mt OC / yr"
# rcmip_gas = "OC"

# gas = "NH3"
# unit_target = "Mt / yr"
# unit_label = "Mt NH3 / yr"
# rcmip_gas = "NH3"

# This reveals the bug in NOx processing for RCMIP v0.5.1 data
# first identified by Chris.
# Long story short: NOx emissions were too low by ~50%.
# In this round, let's report NOx emissions with units of Mt N / yr
# to avoid the ambiguous Mt NOx / yr.
gas = "NOx"
unit_target = "Mt / yr"
unit_label = "Mt NO / yr"  # File metadata has molecular weight of 30 so assume NO
rcmip_gas = "NOx"

# gas = "SO2"
# unit_target = "Mt / yr"
# unit_label = "Mt SO2 / yr"
# rcmip_gas = "Sulfur"

# %%
gas_ds = xr.open_mfdataset(bb4cmip_file_groups[gas], combine_attrs="drop_conflicts")
gas_ds

# %% [markdown]
# Do a unit-aware sum

# %%
emissions_da = gas_ds[gas].pint.quantify()
cellarea_da = cell_area["gridcellarea"].pint.quantify(gridcellarea="m^2")
global_sum_emissions = (emissions_da * cellarea_da).sum(["latitude", "longitude"]).compute()
global_sum_emissions

# %%
global_sum_emissions.plot()

# %%
# Should really weight the mean by the length of the month.
# One for the future.
global_sum_emissions_annual_mean = global_sum_emissions.groupby("time.year").mean()

# Check that the sum matches the data provider's check values
for original_file in bb4cmip_file_groups[gas]:
    orig_ds = xr.open_dataset(original_file)
    timerange = original_file.stem.split("_")[-1].split("-")

    for i, timestamp in enumerate(timerange):
        year = int(timestamp[:4])
        if i < 1:
            compare_val = float(orig_ds.attrs["annual_total_first_year_Tg_yr"])
        elif i == 1:
            compare_val = float(orig_ds.attrs["annual_total_last_year_Tg_yr"])
        else:
            raise NotImplementedError()

        np.testing.assert_allclose(
            global_sum_emissions_annual_mean.sel(year=year).pint.to("Tg / yr").data.m,
            compare_val,
            rtol=5e-2,
        )

global_sum_emissions_annual_mean.plot()


# %%
def gfed_to_scmrun(in_da: xr.DataArray, *, unit_target: str, unit_label: str, region: str) -> scmdata.ScmRun:
    """
    Convert an `xr.DataArray` to `scmdata.ScmRun`

    Custom to this notebook

    Parameters
    ----------
    in_da
        Input `xr.DataArray` to convert

    unit_target
        The unit to which to convert the data to.

        This is done based on the original data unit,
        so almost always has to exclude the gas.
        I.e. this should be e.g. "Mt / yr", not "Mt BC / yr".

    unit_label
        The label to apply to the unit as it will be handled in the output.

        This typically includes the gas, e.g. "Mt BC / yr"

    region
        The region to which the data applies.

        Typically "World".

    Returns
    -------
    :
        `scmdata.ScmRun`
    """
    df = in_da.pint.to(unit_target).pint.dequantify().to_pandas().to_frame().T
    df["model"] = "GFED-BB4CMIP"
    df["scenario"] = "historical-CMIP6Plus"
    df["region"] = region
    df["unit"] = unit_label
    df["variable"] = f"Emissions|{gas}|BB4CMIP"
    out = scmdata.ScmRun(df)

    return out


# %%
gfed_to_plot = gfed_to_scmrun(
    global_sum_emissions_annual_mean, unit_target=unit_target, unit_label=unit_label, region="World"
)
gfed_to_plot.timeseries()

# %%
rcmip_data = scmdata.ScmRun(str(rcmip_data_path), lowercase_cols=True)
rcmip_data

# %%
tmp = rcmip_data.filter(
    variable=[
        f"Emissions|{rcmip_gas}",
        f"Emissions|{rcmip_gas}|MAGICC AFOLU",
        f"Emissions|{rcmip_gas}|MAGICC AFOLU|*Burning*",
    ],
    region="World",
    scenario=["historical", "ssp*"],
)

tmp_total = tmp.filter(variable=[f"Emissions|{rcmip_gas}", f"Emissions|{rcmip_gas}|MAGICC AFOLU"])
tmp_total_burning = tmp.filter(variable=f"Emissions|{rcmip_gas}|MAGICC AFOLU|*Burning*", log_if_empty=False)
if tmp_total_burning.empty:
    # No RCMIP burning emissions
    tmp = scmdata.run_append([tmp_total])

else:
    tmp_total_burning = tmp_total_burning.process_over(
        "variable", "sum", op_cols={"variable": f"Emissions|{rcmip_gas}|MAGICC AFOLU|Burning"}, as_run=True, min_count=1
    )
    tmp = scmdata.run_append([tmp_total, tmp_total_burning])

rcmip_to_plot = scmdata.run_append(
    [
        tmp.filter(scenario="historical"),
        tmp.filter(scenario="historical", keep=False).filter(year=range(2014, 3000 + 1)),
    ]
)

# %%
to_plot = scmdata.run_append([gfed_to_plot, rcmip_to_plot]).convert_unit(unit_label)

ax = to_plot.filter(scenario="hist*").lineplot(style="variable", alpha=0.5, linewidth=2)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.set_ylim(ymin=0)
plt.show()

ax = to_plot.filter(variable=f"Emissions|{rcmip_gas}", keep=False).lineplot(style="variable", alpha=0.5, linewidth=2)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.set_ylim(ymin=0)
