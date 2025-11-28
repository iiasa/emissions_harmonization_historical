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
# # Process CEDS CMIP7 data
#
# Process the anthropogenic emissions
# from the CEDS data on ESGF.
# In theory, these are the same as the CEDS data
# we process in `500*`.
# We do this as an extra double check
# (prompted by [this issue](https://github.com/PCMDI/input4MIPs_CVs/issues/393)).
# We download the data, process it, then delete it
# to avoid filling up our disks.
# We also run this species by species,
# again to avoid overloading our disks.

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import pint_xarray  # noqa: F401
import pooch
import tqdm.auto
import xarray as xr

# %% [markdown]
# ## Setup

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
species: str = "CO2"
species_esgf: str = "CO2"
compute_early: bool = True


# %% [markdown]
# ## Load data

# %%

# %% [markdown]
# ## Download data


# %%
def get_download_urls(
    species: str,
    time_periods: tuple[str, ...] = ("200001-202312",),
) -> tuple[str, ...]:
    """
    Get URLs to download
    """
    total_files = (
        f"https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18/atmos/mon/{species}{suffix}/gn/v20250421/{species}{suffix.replace('_', '-')}_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_{tp}.nc"  # noqa: E501
        for suffix in ["_em_anthro", "_em_AIR_anthro"]
        for tp in time_periods
    )

    res = tuple(
        (
            *total_files,
            "https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18/atmos/fx/areacella/gn/v20250421/areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc",
        )
    )

    return res


# %%
downloaded_files_l = []
for download_url in get_download_urls(
    species_esgf,
    # Full timeseries rather than just the last bit
    time_periods=(
        "175001-179912",
        "180001-184912",
        "185001-189912",
        "190001-194912",
        "195001-199912",
        "200001-202312",
    ),
):
    downloaded_files_l.append(
        pooch.retrieve(
            download_url,
            known_hash=None,  # from ESGF, assume safe
            fname=download_url.split("/")[-1],
            progressbar=True,
        )
    )

# downloaded_files_l

# %%
cell_area = xr.open_mfdataset([esgf_file for esgf_file in downloaded_files_l if "areacella" in esgf_file])["areacella"]
if compute_early:
    cell_area = cell_area.compute()

# cell_area

# %%
other_ems = xr.open_mfdataset([esgf_file for esgf_file in downloaded_files_l if "em-anthro" in esgf_file])
if compute_early:
    other_ems = other_ems.compute()

# other_ems


# %%
def to_annual_sum(da: xr.DataArray, time_bounds: xr.DataArray, bnd_dim: str) -> xr.DataArray:
    """
    Convert to an annual sum
    """
    # Much easier with cf-python
    # Have to use compute otherwise calculation is wrong
    seconds_per_step = (time_bounds.sel(**{bnd_dim: 1}) - time_bounds.sel(**{bnd_dim: 0})).compute()

    # Yuck but ok
    if str(seconds_per_step.values.dtype) != "timedelta64[ns]":
        raise AssertionError

    seconds_per_step = seconds_per_step / 1e9

    if " s-1" not in da.attrs["units"]:
        raise AssertionError

    out_units = da.attrs["units"].replace(" s-1", "")

    res = (da * seconds_per_step.astype(int)).groupby("time.year").sum().assign_attrs(dict(units=out_units))

    return res


def to_annual_global_sum(
    ds: xr.Dataset, variable_of_interest: str, cell_area: xr.DataArray, bnd_dim: str = "bound"
) -> xr.DataArray:
    """
    Convert to an annual- global-sum
    """
    # Much easier with cf-python
    annual_sum = to_annual_sum(
        ds[variable_of_interest],
        time_bounds=ds["time_bnds"],
        bnd_dim=bnd_dim,
    )

    res = (annual_sum * cell_area).sum(["lat", "lon"])

    if cell_area.attrs["units"] != "m2":
        raise AssertionError(cell_area.attrs["units"])

    res.attrs["units"] = annual_sum.attrs["units"].replace(" m-2", "")

    return res


# %%
other_emms_annual_global = to_annual_global_sum(
    other_ems,
    variable_of_interest=f"{species_esgf}_em_anthro",
    cell_area=cell_area,
)
# other_emms_annual_global

# %%
other_emms_annual_global_sector_sum = other_emms_annual_global.sum("sector", keep_attrs=True)
# other_emms_annual_global_sector_sum

# %%
aviation_ems_annual_global_level_sum_l = []
for aviation_file in tqdm.auto.tqdm([esgf_file for esgf_file in downloaded_files_l if "em-AIR-anthro" in esgf_file]):
    chunk_sum = (
        to_annual_global_sum(
            xr.open_mfdataset([aviation_file]).compute(),
            variable_of_interest=f"{species_esgf}_em_AIR_anthro",
            cell_area=cell_area,
            bnd_dim="bnds",
        )
        .sum("level", keep_attrs=True)
        .compute()
    )
    # aviation_ems_annual_global

    aviation_ems_annual_global_level_sum_l.append(chunk_sum)

aviation_ems_annual_global_level_sum = xr.concat(aviation_ems_annual_global_level_sum_l, dim="year")
aviation_ems_annual_global_level_sum

# %%
# aviation_ems = xr.open_mfdataset([esgf_file for esgf_file in downloaded_files_l if "em-AIR-anthro" in esgf_file])
# if compute_early:
#     aviation_ems = aviation_ems.compute()

# # aviation_ems

# %%
# aviation_ems_annual_global = to_annual_global_sum(
#     aviation_ems.compute(),
#     variable_of_interest=f"{species_esgf}_em_AIR_anthro",
#     cell_area=cell_area,
#     bnd_dim="bnds",
# )
# # aviation_ems_annual_global

# %%
# aviation_ems_annual_global_level_sum = aviation_ems_annual_global.sum("level", keep_attrs=True)
# # aviation_ems_annual_global_level_sum

# %%
annual_totals = (
    other_emms_annual_global_sector_sum.pint.quantify() + aviation_ems_annual_global_level_sum.pint.quantify()
).pint.dequantify()
annual_totals

# %%
out_s = annual_totals.to_series()
out_ts = pd.DataFrame(
    out_s,
    columns=pd.MultiIndex.from_tuples(
        [
            (
                f"{annual_totals.attrs['units']} {species_esgf} / yr",
                "Emissions",
                species_esgf,
                "CEDS",
                "World",
                "historical",
            )
        ],
        names=["unit", "table", "species", "source", "region", "scenario"],
    ),
).T
out_ts.T.plot()
out_ts

# %%
# Save as CSV
try:
    from emissions_harmonization_historical.constants_5000 import (
        CEDS_CMIP_PROCESSED_DIR,
    )
except ImportError:
    print("Saving to current directory")
    CEDS_CMIP_PROCESSED_DIR = "."

out_ts.to_csv(CEDS_CMIP_PROCESSED_DIR / f"{species_esgf}_ceds-cmip-annual-total.csv")

# %%
# Save to DB
from emissions_harmonization_historical.constants_5000 import (  # noqa: E402
    CEDS_CMIP_PROCESSED_DB,
)

CEDS_CMIP_PROCESSED_DB.save(out_ts, allow_overwrite=True)

# %% [markdown]
# ## Delete raw files
#
# Save our disks

# %%
assert False, "Reactivate"
# for fp in downloaded_files_l:
#     os.unlink(fp)
