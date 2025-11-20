# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Infilling
#
# Here we infill each model's data.

# %% [markdown]
# ## Imports

# %%
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import silicone.database_crunchers
import tqdm.auto
from gcages.completeness import assert_all_groups_are_complete
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    CMIP7_GHG_PROCESSED_DB,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_OUT_DIR,
    INFILLED_SCENARIOS_DB_2100,
    INFILLING_DB,
    MARKERS_BY_SCENARIOMIP_NAME,
    WMO_2022_PROCESSED_DB,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
    assert_harmonised,
)
from emissions_harmonization_historical.infilling import (
    get_complete,
    get_direct_copy_infiller,
    get_direct_scaling_infiller,
    get_silicone_based_infiller,
    infill,
)
from emissions_harmonization_historical.scm_running import complete_index_reporting_names

# ## Set up

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %%
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "REMIND"

# %% editable=true slideshow={"slide_type": ""}
output_dir_model = INFILLED_OUT_DIR / model
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Harmonised data for SCMs
#
# We're not interested in infilling the other workflows.

# %%
harmonised = HARMONISED_SCENARIO_DB.load(pix.isin(workflow="for_scms") & pix.ismatch(model=f"*{model}*")).reset_index(
    "workflow", drop=True
)
# harmonised

# %% [markdown]
# ### Infilling database

# %%
infilling_db = INFILLING_DB.load()
# infilling_db

# %% [markdown]
# ### History

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

# history

# %% [markdown]
# ### CMIP7 GHG inversions

# %%
cmip7_ghg_inversions = CMIP7_GHG_PROCESSED_DB.load()
# cmip7_ghg_inversions

# %% [markdown]
# ## Check that the infilling database and scenario data are harmonised the same


# %%
species_tolerances = {
    "BC": dict(rtol=1e-3, atol=Q(1e-3, "Mt BC/yr")),
    "CH4": dict(rtol=1e-3, atol=Q(1e-2, "Mt CH4/yr")),
    "CO": dict(rtol=1e-3, atol=Q(1e-1, "Mt CO/yr")),
    "CO2": dict(rtol=1e-3, atol=Q(1e0, "Mt CO2/yr")),
    "NH3": dict(rtol=1e-3, atol=Q(1e-2, "Mt NH3/yr")),
    "NOx": dict(rtol=1e-3, atol=Q(1e-2, "Mt NO2/yr")),
    "OC": dict(rtol=1e-3, atol=Q(1e-3, "Mt OC/yr")),
    "Sulfur": dict(rtol=1e-3, atol=Q(1e-2, "Mt SO2/yr")),
    "VOC": dict(rtol=1e-3, atol=Q(1e-2, "Mt VOC/yr")),
    "N2O": dict(rtol=1e-3, atol=Q(1e-1, "kt N2O/yr")),
}

# %%
assert_harmonised(harmonised, history, species_tolerances=species_tolerances)

# %%
assert_harmonised(infilling_db, history, species_tolerances=species_tolerances)

# %%
wmo_locator = pix.ismatch(model="WMO**")
infilling_db_wmo = infilling_db.loc[wmo_locator]

velders_locator = pix.ismatch(model="Velders**")
infilling_db_velders = infilling_db.loc[velders_locator]

infilling_db_silicone = infilling_db.loc[~wmo_locator & ~velders_locator]


# %% [markdown]
# ## Infill

# %% [markdown]
# ### Very low marker should use F-gas emissions in line with Kigali
#
# We get these from [Velders et al., 2022](https://zenodo.org/records/6520707)
# (processed in a previous notebook)

# %%
vl_model = MARKERS_BY_SCENARIOMIP_NAME["vl"]["model"]
vl_scenario = MARKERS_BY_SCENARIOMIP_NAME["vl"]["scenario"]

# %% editable=true slideshow={"slide_type": ""}
vl_marker = harmonised.loc[pix.isin(model=vl_model) & pix.isin(scenario=vl_scenario)]

if not vl_marker.empty:
    print(f"Infilling {vl_model} {vl_scenario} with Velders lower Kigali scenario")
    infilled_vl_exception = infilling_db.loc[
        pix.isin(model="Velders et al., 2022", scenario="Kigali2022-lower")
    ].pix.assign(model=vl_model, scenario=vl_scenario)

else:
    print("vl marker is not in the input scenarios")
    infilled_vl_exception = None

complete_vl_exception = get_complete(harmonised, infilled_vl_exception)

# %% [markdown]
# ### Silicone


# %% editable=true slideshow={"slide_type": ""}
lead = "Emissions|CO2|Energy and Industrial Processes"
infillers_silicone = {}
for variable in tqdm.auto.tqdm([v for v in infilling_db_silicone.pix.unique("variable") if v != lead]):
    infillers_silicone[variable] = get_silicone_based_infiller(
        infilling_db=infilling_db_silicone,
        follower_variable=variable,
        lead_variables=[lead],
        silicone_db_cruncher=silicone.database_crunchers.RMSClosest,
    )

# %%
infilled_silicone = infill(
    complete_vl_exception,
    # harmonised.loc[~pix.isin(variable=infill_silicone_gcages[:3]) | pix.isin(variable=lead)],
    infillers_silicone,
)
complete_silicone = get_complete(complete_vl_exception, infilled_silicone)
# complete_silicone

# %%
if infilled_silicone is None:
    print("Nothing infilled with silicone")

else:
    col_order = [lead, *sorted(infilled_silicone.pix.unique("variable"))]
    infilling_db_silicone_plt = infilling_db_silicone.loc[pix.isin(variable=col_order)]
    # # This is why we get the crazy spike in HFC43-10.
    # # Have emailed the REMIND team to get them to check.
    # hfc4310 = infilling_db_silicone_plt.loc[pix.isin(variable="Emissions|HFC|HFC43-10")]
    # display(hfc4310.sort_values(by=2050).loc[:, [2023, 2050]])

    infilled_silicone_pdf = complete_silicone.loc[pix.isin(variable=col_order)].openscm.to_long_data()
    fg = sns.relplot(
        infilled_silicone_pdf,
        x="time",
        y="value",
        hue="scenario",
        col="variable",
        col_wrap=5,
        col_order=col_order,
        kind="line",
        facet_kws=dict(sharey=False),
        linewidth=3.0,
        zorder=3.0,
    )
    for ax in fg.axes.flatten():
        variable = ax.get_title().split("variable = ")[1]
        infiller_ts = infilling_db_silicone_plt.loc[pix.isin(variable=variable)].T
        ax.plot(
            infiller_ts.index.values,
            infiller_ts.values,
            color="gray",
            zorder=1.0,
            linewidth=0.25,
        )

# %% [markdown]
# ### WMO 2022
#
# Some of the infilling timeseries look super weird in isolation.
# However, this is mostly rounding errors,
# which is much clearer when you plot them in their historical context.


# %%
wmo_2022_smoothed_full = WMO_2022_PROCESSED_DB.load(pix.isin(model=infilling_db_wmo.pix.unique("model")))
# wmo_2022_smoothed_full

# %%
pdf = pix.concat(
    [
        infilling_db_wmo.pix.assign(use="infilling_db"),
        wmo_2022_smoothed_full.pix.assign(use="full_ts"),
    ]
).openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="use",
    style="use",
    dashes={
        "infilling_db": "",
        "full_ts": (1, 2),
    },
    col="variable",
    col_wrap=7,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# #### Infill

# %%
infillers_wmo = {}
for wmo_var in infilling_db_wmo.pix.unique("variable"):
    infillers_wmo[wmo_var] = get_direct_copy_infiller(
        variable=wmo_var,
        copy_from=infilling_db_wmo,
    )

# %%
infilled_wmo = infill(complete_silicone, infillers_wmo)
complete_wmo = get_complete(complete_silicone, infilled_wmo)
complete_wmo


# %%
# Just checking all scenarios get the same
pdf = infilled_wmo.openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="scenario",
    col="variable",
    col_wrap=7,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# ### Scale timeseries
#
# Surprisingly, this is the most mucking around of all.
# The hard part here is that the scaling needs to be aware
# of the fact that the pre-industrial value is different for each tiemseries.
# The naming mucking around also adds to the fun of course.


# %%
to_reporting_names = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.GCAGES,
    to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
)

scaling_leaders_gcages = {
    "Emissions|C3F8": "Emissions|C2F6",
    "Emissions|C4F10": "Emissions|C2F6",
    "Emissions|C5F12": "Emissions|C2F6",
    "Emissions|C7F16": "Emissions|C2F6",
    "Emissions|C8F18": "Emissions|C2F6",
    "Emissions|cC4F8": "Emissions|CF4",
    "Emissions|SO2F2": "Emissions|CF4",
    "Emissions|HFC236fa": "Emissions|HFC245fa",
    "Emissions|HFC152a": "Emissions|HFC4310mee",
    "Emissions|HFC365mfc": "Emissions|HFC134a",
    "Emissions|CH2Cl2": "Emissions|HFC134a",
    "Emissions|CHCl3": "Emissions|C2F6",
    "Emissions|CH3Br": "Emissions|C2F6",
    "Emissions|CH3Cl": "Emissions|CF4",
    "Emissions|NF3": "Emissions|SF6",
}
scaling_leaders = {to_reporting_names(k): to_reporting_names(v) for k, v in scaling_leaders_gcages.items()}

# %%
cmip7_ghg_inversions_reporting_names = update_index_levels_func(cmip7_ghg_inversions, {"variable": to_reporting_names})

# %% [markdown]
# We considered breaking the following out,
# but for now it's better to see the logic.
# We can move it later if we need
# into a function like `get_pre_industrial_aware_direct_scaling_infiller`.

# %%
PI_YEAR = 1750
infillers_scaling = {}
for follower, leader in tqdm.auto.tqdm(scaling_leaders.items()):
    # For each follower, leader pair we need:
    # - f_harmonisation_year: The value of the follower in the harmonisation year
    # - l_harmonisation_year: The value of the leader in the harmonisation year
    # - f_0: The value of the follower at pre-industrial
    # - l_0: The value of the leader at pre-industrial
    #
    # We can then do 'pre-industrial aware scaling' with
    # f_future =  (l_future - l_0) * (f_harmonisation_year - f_0) / (l_harmonisation_year - l_0) + f_0
    #
    # so that:
    #
    # - f_future(l_0) = f_0 i.e. if the lead goes to its pre-industrial value,
    #   the result is the follower's pre-industrial value
    #
    # - f_future(l_harmonisation_year) = f_harmonisation_year
    #   i.e. we preserve harmonisation of the follower
    #
    # - there is a linear transition between these two points
    #   as the lead variable's emissions change

    lead_df = history.loc[pix.isin(variable=[leader])]
    follow_df = history.loc[pix.isin(variable=[follower])]
    lead_cmip7_inverse_df = cmip7_ghg_inversions_reporting_names.loc[pix.isin(variable=[leader])]
    follow_cmip7_inverse_df = cmip7_ghg_inversions_reporting_names.loc[pix.isin(variable=[follower])]

    f_unit = follow_df.pix.unique("unit")
    if len(f_unit) != 1:
        raise AssertionError
    f_unit = f_unit[0].replace("-", "")

    l_unit = lead_df.pix.unique("unit")
    if len(l_unit) != 1:
        raise AssertionError
    l_unit = l_unit[0].replace("-", "")

    for harmonisation_yr_use in [HARMONISATION_YEAR, 2021]:
        l_harmonisation_year = float(lead_df[harmonisation_yr_use].values.squeeze())
        f_harmonisation_year = float(follow_df[harmonisation_yr_use].values.squeeze())
        if not (pd.isnull(l_harmonisation_year) or pd.isnull(f_harmonisation_year)):
            break
    else:
        raise AssertionError

    f_0 = float(follow_cmip7_inverse_df[PI_YEAR].values.squeeze())
    l_0 = float(lead_cmip7_inverse_df[PI_YEAR].values.squeeze())

    # if (f_harmonisation_year - f_0) == 0.0:
    #     scaling_factor = 0.0
    # else:
    scaling_factor = (f_harmonisation_year - f_0) / (l_harmonisation_year - l_0)

    if np.isnan(scaling_factor):
        msg = f"{f_harmonisation_year=} {l_harmonisation_year=} {f_0=} {l_0=}"
        raise AssertionError(msg)

    # break
    infillers_scaling[follower] = get_direct_scaling_infiller(
        leader=leader,
        follower=follower,
        scaling_factor=scaling_factor,
        l_0=l_0,
        f_0=f_0,
        f_unit=f_unit,
        calculation_year=harmonisation_yr_use,
        f_calculation_year=f_harmonisation_year,
    )

# %%
infilled_scaling = infill(complete_wmo, infillers_scaling)
complete = get_complete(complete_wmo, infilled_scaling)

# %%
for follower, leader in tqdm.auto.tqdm(scaling_leaders.items()):
    pdf = complete.loc[pix.isin(variable=[follower, leader])].openscm.to_long_data()
    fg = sns.relplot(
        data=pdf,
        x="time",
        y="value",
        hue="scenario",
        col="variable",
        col_wrap=2,
        col_order=[leader, follower],
        kind="line",
        facet_kws=dict(sharey=False),
        height=2.5,
        aspect=1.25,
    )
    for ax in fg.axes.flatten():
        ax.set_ylim(ymin=0.0)

    plt.show()

# %% [markdown]
# ## Check completeness

# %%
assert_all_groups_are_complete(complete, complete_index_reporting_names)

# %%
# complete

# %% [markdown]
# ## Save

# %%
for ids, df in (
    ("silicone", infilled_silicone),
    ("wmo", infilled_wmo),
    ("scaled", infilled_scaling),
    ("complete", complete),
):
    if df is not None:
        INFILLED_SCENARIOS_DB_2100.save(
            df.pix.assign(stage=ids).sort_index(axis="columns"), allow_overwrite=True, progress=True
        )
