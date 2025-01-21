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

# %% [markdown]
# # Composite historical scenarios
#
# Create our historical timeseries from the underlying data.

# %%
import pandas_indexing as pix

from emissions_harmonization_historical.constants import (
    CEDS_PROCESSING_ID,
    DATA_ROOT,
    GCB_PROCESSING_ID,
    GFED_PROCESSING_ID,
    HISTORICAL_COMPOSITE_PROCESSING_ID,
    VELDERS_ET_AL_2022_PROCESSING_ID,
)
from emissions_harmonization_historical.io import load_csv

# %%
OUT_FILE = DATA_ROOT / "global-composite" / f"historical-global-composite_{HISTORICAL_COMPOSITE_PROCESSING_ID}.csv"


# %%
ceds_raw = load_csv(DATA_ROOT / "national" / "ceds" / "processed" / f"ceds_cmip7_global_{CEDS_PROCESSING_ID}.csv")
ceds_raw

# %%
ceds_sum = (
    ceds_raw.pix.extract(variable="Emissions|{species}|{sector}", dropna=False)
    .groupby([*(set(ceds_raw.index.names) - {"variable"}), "species"])
    .sum()
    .pix.format(variable="Emissions|{species}|CEDS", drop=True)
)

# %%
ceds_co2_search = "Emissions|CO2"
ceds_co2_out_name = f"{ceds_co2_search}|Energy and Industrial Processes"
ceds_co2 = (
    ceds_raw.loc[pix.ismatch(variable=f"{ceds_co2_search}**")]
    .groupby([*set(ceds_raw.index.names) - {"variable"}])
    .sum()
    .pix.assign(variable=ceds_co2_out_name)
)
ceds_co2 = ceds_co2.pix.assign(scenario="history", model=ceds_co2.pix.unique("scenario"))
ceds_co2

# %%
bb4cmip_raw = load_csv(
    DATA_ROOT / "national/gfed-bb4cmip/processed" / f"gfed-bb4cmip_cmip7_global_{GFED_PROCESSING_ID}.csv"
)
bb4cmip_raw

# %%
gcb_afolu_raw = load_csv(DATA_ROOT / "global" / "gcb" / "processed" / f"gcb-afolu_cmip7_global_{GCB_PROCESSING_ID}.csv")
gcb_afolu_raw

# %%
velders_et_al_2022_raw = load_csv(
    DATA_ROOT
    / "global"
    / "velders-et-al-2022"
    / "processed"
    / f"velders-et-al-2022_cmip7_global_{VELDERS_ET_AL_2022_PROCESSING_ID}.csv"
)
velders_et_al_2022_raw

# %%
history_non_co2 = (
    pix.concat([ceds_sum, bb4cmip_raw])
    # Need to add GCP data for CO2 AFOLU
    # and figure out what to do with CO2 burning data
    .loc[~pix.ismatch(variable="Emissions|CO2**")]
    .pix.extract(variable="Emissions|{species}|{source}")
    .pix.assign(model="CEDS-BB4CMIP", scenario="history")
    .groupby([*(set(ceds_raw.index.names) - {"variable"}), "species"])
    .sum()  # not unit aware, but could make it so in future
    .pix.format(variable="Emissions|{species}", drop=True)
    # TODO: rename NMVOC to VOC here, same for units
)
history = pix.concat(
    [
        history_non_co2,
        ceds_co2,
        gcb_afolu_raw.pix.assign(scenario="history"),
        velders_et_al_2022_raw.pix.assign(scenario="history"),
    ]
)
if len(history.pix.unique("scenario")) > 1:
    msg = f"{history.pix.unique(['model', 'scenario'])}"
    raise AssertionError(msg)

history

# %%
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)
history.to_csv(OUT_FILE)
OUT_FILE
