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
# # Infill scenarios - leftovers
#
# Infill whatever we haven't infilled already.

# %%
import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import seaborn as sns

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
)

# %%
# Disable all logging to avoid a million messages
logging.disable()

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_PATH


# %%
def transform_rcmip_to_iamc_variable(v):
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (
        ("F-Gases|", ""),
        ("PFC|", ""),
        ("HFC4310mee", "HFC43-10"),
        ("MAGICC AFOLU", "AFOLU"),
        ("MAGICC Fossil and Industrial", "Energy and Industrial Processes"),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
rcmip_clean


# %%
def get_sns_df(indf):
    """
    Get data frame to use with seaborn's plotting
    """
    out = indf.copy()
    out.columns.name = "year"
    out = out.stack().to_frame("value").reset_index()

    return out


# %%
make_var_comparison_plot = partial(
    sns.relplot,
    x="year",
    y="value",
    hue="scenario",
    style="variable",
    facet_kws=dict(sharey=False),
)

# %%
# Leftovers - make this dynamic in future
# by doing difference between RCMIP variables
# and what is in the scenarios as they've been infilled up to this point.
leftovers = [
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C7F16",
    "Emissions|C8F18",
    # # Velders probably
    # 'Emissions|HFC|HFC152a',
    # 'Emissions|HFC|HFC236fa',
    # 'Emissions|HFC|HFC365mfc',
    # WMO almost definitely
    # 'Emissions|Montreal Gases|HCFC141b',
    # 'Emissions|Montreal Gases|HCFC142b',
    #    # WMO probably
    # 'Emissions|Montreal Gases|CCl4',
    # 'Emissions|Montreal Gases|CFC|CFC11',
    # 'Emissions|Montreal Gases|CFC|CFC113',
    # 'Emissions|Montreal Gases|CFC|CFC114',
    # 'Emissions|Montreal Gases|CFC|CFC115',
    # 'Emissions|Montreal Gases|CFC|CFC12',
    "Emissions|Montreal Gases|CH2Cl2",
    # 'Emissions|Montreal Gases|CH3Br',
    # 'Emissions|Montreal Gases|CH3CCl3',
    # 'Emissions|Montreal Gases|CH3Cl',
    "Emissions|Montreal Gases|CHCl3",
    # 'Emissions|Montreal Gases|HCFC22',
    # 'Emissions|Montreal Gases|Halon1202',
    # 'Emissions|Montreal Gases|Halon1211',
    # 'Emissions|Montreal Gases|Halon1301',
    # 'Emissions|Montreal Gases|Halon2402',
    #
    "Emissions|NF3",
    "Emissions|SO2F2",
    "Emissions|cC4F8",
]

# %%
ssps = (
    rcmip_clean.loc[pix.ismatch(scenario="ssp*", region="World")]
    .reset_index(["mip_era", "activity_id"], drop=True)
    .dropna(axis="columns", how="all")
)
ssps = ssps.loc[:, 2015:2090]
ssps


# %%
def get_std(idf):
    return np.std(idf.values)


# %%
follow_leaders = {}
for follow in leftovers:
    potential_leads = ssps[~(ssps == 0.0).all(axis="columns") & ~pix.isin(variable=[follow, *leftovers])]

    lead = (
        potential_leads.divide(ssps.loc[pix.isin(variable=[follow])].reset_index(["variable", "unit"], drop=True))
        .groupby("variable")
        .apply(get_std)
        .idxmin()
    )

    follow_leaders[follow] = lead

    # display(ssps.loc[pix.isin(variable=[lead])].divide(
    #     ssps.loc[pix.isin(variable=[follow])].reset_index(["variable", "unit"], drop=True)
    # ).round(3))

    pdf = ssps.loc[pix.isin(variable=[lead, follow])]
    pdf = pdf.divide(pdf[2015].groupby("variable").mean(), axis="rows")
    fg = make_var_comparison_plot(
        data=get_sns_df(pdf),
        kind="line",
        alpha=0.5,
    )
    for ax in fg.axes.flatten():
        ax.set_ylim(ymin=0)

    plt.show()
    # break

# %%
follow_leaders

# %%
# Extracted from `magicc-archive/run/SSP5_34_OS_HFC_C2F6_CF4_SF6_MISSGAS_ExtPlus.SCEN7`
follow_leaders_mm = {
    "Emissions|cC4F8": "Emissions|CF4",
    "Emissions|SO2F2": "Emissions|CF4",
    "Emissions|NF3": "Emissions|SF6",
    "Emissions|HFC|HFC365mfc": "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC32": "Emissions|HFC|HFC23",
    "Emissions|HFC|HFC236fa": "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC152a": "Emissions|HFC|HFC43-10",
    "Emissions|Montreal Gases|CHCl3": "Emissions|C2F6",
    "Emissions|Montreal Gases|CH3Br": "Emissions|C2F6",
    "Emissions|Montreal Gases|CH3Cl": "Emissions|CF4",
    "Emissions|Montreal Gases|CH2Cl2": "Emissions|HFC|HFC134a",
    "Emissions|Montreal Gases|Halon1202": "Emissions|Montreal Gases|Halon1211",
    "Emissions|C3F8": "Emissions|C2F6",
    "Emissions|C4F10": "Emissions|C2F6",
    "Emissions|C5F12": "Emissions|C2F6",
    "Emissions|C6F14": "Emissions|C2F6",
    "Emissions|C7F16": "Emissions|C2F6",
    "Emissions|C8F18": "Emissions|C2F6",
}

# %%
for follow, lead in follow_leaders_mm.items():
    pdf = ssps.loc[pix.isin(variable=[lead, follow])]
    pdf = pdf.divide(pdf[2015].groupby("variable").mean(), axis="rows")
    fg = make_var_comparison_plot(data=get_sns_df(pdf), kind="line", alpha=0.5, dashes={lead: "", follow: (3, 3)})
    for ax in fg.axes.flatten():
        ax.set_ylim(ymin=0)

    plt.show()
    # break

# %% [markdown]
# Have to be a bit clever with scaling to consider background/natural emissions.
#
# Instead of
#
# $$
# f = a * l
# $$
# where $f$ is the follow variable, $a$ is the scaling factor and $l$ is the lead.
#
# We want
# $$
# f - f_0 = a * (l - l_0)
# $$
# where $f_0$ is pre-industrial emissions of the follow variable and $l_0$ is pre-industrial emissions of the lead.

# %%
for follow, lead in follow_leaders_mm.items():
    # if not follow.endswith("CHCl3"):
    #     continue

    lead_df = ssps.loc[pix.isin(variable=[lead])]
    follow_df = ssps.loc[pix.isin(variable=[follow])]

    follow_pi = rcmip_clean.loc[
        pix.isin(scenario=["historical"], mip_era=["CMIP6"], variable=[follow]), 1750
    ].values.squeeze()
    lead_pi = rcmip_clean.loc[
        pix.isin(scenario=["historical"], mip_era=["CMIP6"], variable=[lead]), 1750
    ].values.squeeze()
    quotient = (lead_df - lead_pi).divide(
        (follow_df - follow_pi).reset_index(["variable", "unit"], drop=True), axis="rows"
    )

    scaling_factor = quotient[2015].mean()

    pdf = pix.concat([lead_df, (follow_df - follow_pi) * scaling_factor])
    # pdf = ssps.loc[pix.isin(variable=[lead, follow])]
    # pdf = pdf.divide(pdf[2015].groupby("variable").mean(), axis="rows")
    fg = make_var_comparison_plot(data=get_sns_df(pdf), kind="line", alpha=0.5, dashes={lead: "", follow: (3, 3)})
    for ax in fg.axes.flatten():
        ax.set_ylim(ymin=0)

    plt.show()
    # break
