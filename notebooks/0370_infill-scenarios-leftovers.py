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
import pandas as pd
import pandas_indexing as pix
import seaborn as sns

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    INFILLING_LEFTOVERS_ID,
    INFILLING_SILICONE_ID,
    INFILLING_WMO_ID,
)
from emissions_harmonization_historical.io import load_csv

# %%
# Disable all logging to avoid a million messages
logging.disable()

# %%
SCENARIO_TIME_ID = "20250122-140031"

# %%
harmonised_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "harmonised"
    / f"harmonised-scenarios_{SCENARIO_TIME_ID}_{HARMONISATION_ID}.csv"
)
harmonised_file

# %%
infilled_silicone_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "interim"
    / f"infilled-silicone_{SCENARIO_TIME_ID}_{INFILLING_SILICONE_ID}.csv"
)
infilled_silicone_file

# %%
infilled_wmo_file = (
    DATA_ROOT / "climate-assessment-workflow" / "interim" / f"infilled-wmo_{SCENARIO_TIME_ID}_{INFILLING_WMO_ID}.csv"
)
infilled_wmo_file

# %%
out_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "interim"
    / f"infilled-leftovers_{SCENARIO_TIME_ID}_{INFILLING_LEFTOVERS_ID}.csv"
)
out_file

# %%
all_so_far = pix.concat(
    [
        load_csv(harmonised_file),
        load_csv(infilled_silicone_file),
        load_csv(infilled_wmo_file),
    ]
)
all_so_far.sort_index()

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
ar6_history = rcmip_clean.loc[pix.isin(mip_era=["CMIP6"], scenario=["ssp245"], region=["World"])]
ar6_history = (
    ar6_history.loc[
        ~pix.ismatch(
            variable=[
                f"Emissions|{stub}|**" for stub in ["BC", "CH4", "CO", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC"]
            ]
        )
        & ~pix.ismatch(variable=["Emissions|CO2|*|**"])
        & ~pix.isin(variable=["Emissions|CO2"])
    ]
    .T.interpolate("index")
    .T
)
full_var_set = ar6_history.pix.unique("variable")
n_variables_in_full_scenario = 52
if len(full_var_set) != n_variables_in_full_scenario:
    raise AssertionError

sorted(full_var_set)

# %%
# TODO: use Guus' data for the HFCs rather than this infilling (?)
leftover_vars = set(full_var_set) - set(all_so_far.pix.unique("variable"))
leftover_vars

# %%
for (model, scenario), msdf in all_so_far.groupby(["model", "scenario"]):
    if set(full_var_set) - set(msdf.pix.unique("variable")) != leftover_vars:
        print(f"{model=} {scenario=}")
    # break

# %%
# Extracted from `magicc-archive/run/SSP5_34_OS_HFC_C2F6_CF4_SF6_MISSGAS_ExtPlus.SCEN7`
follow_leaders_mm = {
    "Emissions|C3F8": "Emissions|C2F6",
    "Emissions|C4F10": "Emissions|C2F6",
    "Emissions|C5F12": "Emissions|C2F6",
    "Emissions|C7F16": "Emissions|C2F6",
    "Emissions|C8F18": "Emissions|C2F6",
    "Emissions|cC4F8": "Emissions|CF4",
    "Emissions|SO2F2": "Emissions|CF4",
    "Emissions|HFC|HFC236fa": "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC152a": "Emissions|HFC|HFC43-10",
    "Emissions|HFC|HFC365mfc": "Emissions|HFC|HFC134a",
    "Emissions|Montreal Gases|CH2Cl2": "Emissions|HFC|HFC134a",
    "Emissions|Montreal Gases|CHCl3": "Emissions|C2F6",
    # "Emissions|Montreal Gases|CH3Br": "Emissions|C2F6",
    # "Emissions|Montreal Gases|CH3Cl": "Emissions|CF4",
    "Emissions|NF3": "Emissions|SF6",
    "Emissions|Montreal Gases|Halon1202": "Emissions|Montreal Gases|Halon1211",
}
if leftover_vars - set(follow_leaders_mm.keys()):
    raise AssertionError()

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
harmonisation_year = 2021

infilled_l = []
for v in leftover_vars:
    leader = follow_leaders_mm[v]

    lead_df = all_so_far.loc[pix.isin(variable=[leader])]

    # TODO: use better source for pre-industrial emissions
    f_0 = ar6_history.loc[pix.isin(variable=[v])][1750].values.squeeze()
    print(f"{v=} {f_0=}")
    l_0 = ar6_history.loc[pix.isin(variable=[leader])][1750].values.squeeze()

    # TODO use actual history for this
    follow_df_history = ar6_history.loc[pix.isin(variable=[v])]
    unit = follow_df_history.pix.unique("unit").unique().tolist()
    if len(unit) > 1:
        raise AssertionError(unit)

    unit = unit[0]

    # TODO: use actual history for this and remove the mean stuff everywhere
    norm_factor = (lead_df - l_0)[harmonisation_year].values.mean() / (follow_df_history - f_0)[
        harmonisation_year
    ].values.mean()

    follow_df = ((lead_df - l_0) / norm_factor + f_0).pix.assign(variable=v, unit=unit)

    infilled_l.append(follow_df)

infilled = pix.concat(infilled_l)
infilled


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
sns.relplot(
    data=get_sns_df(
        pix.concat(
            [
                infilled,
                ar6_history.loc[pix.isin(variable=list(leftover_vars)), 2000:harmonisation_year].reset_index(
                    ["mip_era", "activity_id"], drop=True
                ),
            ]
        )
    ),
    kind="line",
    x="year",
    y="value",
    hue="model",
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
    units="scenario",
    estimator=None,
)

# %%
out_file.parent.mkdir(exist_ok=True, parents=True)
infilled.to_csv(out_file)
out_file

# %% [markdown]
# ## CMIP6
#
# We can also show that this was used in CMIP6.
# It's not perfect for CHCl3 (probably because of feedbacks),
# but it's near enough.

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
ssps = (
    rcmip_clean.loc[pix.ismatch(scenario="ssp*", region="World")]
    .reset_index(["mip_era", "activity_id"], drop=True)
    .dropna(axis="columns", how="all")
)
ssps = ssps.loc[:, 2015:2100]
ssps

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
