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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Create history for global workflow
#
# Includes some extensions and other trickery.

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
from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.index_manipulation import set_new_single_value_levels
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func
from scipy import optimize

from emissions_harmonization_historical.constants_5000 import (
    ADAM_ET_AL_2024_PROCESSED_DB,
    CEDS_RAW_PATH,
    CMIP7_GHG_PROCESSED_DB,
    CREATE_HISTORY_FOR_GRIDDING_ID,
    GCB_PROCESSED_DB,
    HISTORY_FOR_HARMONISATION_ID,
    HISTORY_HARMONISATION_INTERIM_DIR,
    HISTORY_SCENARIO_NAME,
    RCMIP_PROCESSED_DB,
    VELDERS_ET_AL_2022_PROCESSED_DB,
    WMO_2022_PROCESSED_DB,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Setup

# %%
pandas_openscm.register_pandas_accessor()

# %%
pix.set_openscm_registry_as_default()

# %%
Q = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Load data

# %% [markdown]
# ### History data

# %%
history_for_gridding_harmonisation = pd.read_feather(
    HISTORY_HARMONISATION_INTERIM_DIR / f"gridding-history_{CREATE_HISTORY_FOR_GRIDDING_ID}.feather"
)
# history_for_gridding_harmonisation

# %% [markdown]
# ## Compile history for global workflow

# %% [markdown]
# ### Aggregate gridding history

# %%
aggregation_regions = [
    r for r in history_for_gridding_harmonisation.loc[pix.ismatch(region="iso3**")].pix.unique("region")
]
# aggregation_regions

# %%
to_gcages_names = partial(
    update_index_levels_func,
    updates={
        "variable": partial(
            convert_variable_name,
            to_convention=SupportedNamingConventions.GCAGES,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
history_for_gridding_harmonisation_aggregated = to_global_workflow_emissions(
    history_for_gridding_harmonisation.loc[pix.isin(region=aggregation_regions)].pix.assign(model="gridding-emissions"),
    global_workflow_co2_fossil_sector="Energy and Industrial Processes",
    global_workflow_co2_biosphere_sector="AFOLU",
)
history_for_gridding_harmonisation_aggregated = to_gcages_names(history_for_gridding_harmonisation_aggregated)
history_for_gridding_harmonisation_aggregated

# %%
# Weird that it isn't easier to figure this out. Anyway.
tmp = history_for_gridding_harmonisation.loc[pix.ismatch(variable="**CH4|Energy Sector", region="iso3ish|usa")]
ceds_ext_years = tmp[tmp == 0.0].dropna(axis="columns").columns.values
ceds_ext_years

# %%
ceds_extensions_l = []
for ext_f in (CEDS_RAW_PATH / "CEDS_v_2025_03_18_supplementary_extension").glob(
    "*Extension_CEDS_global_estimates_by_sector_v*.csv"
):
    tmp = pd.read_csv(ext_f)

    em_l = tmp["em"].unique()
    if len(em_l) > 1:
        raise AssertionError(em_l)

    em = em_l[0]
    species = em.split("_Extension")[0]

    tmp["unit"] = tmp["units"] + f" {species}/yr"
    tmp["variable"] = f"Emissions|{species}"
    tmp = tmp.drop(["em", "units"], axis="columns").set_index(["variable", "unit", "sector"])
    tmp.columns = [int(v.lstrip("X")) for v in tmp.columns]

    ext_ts = (
        tmp.loc[:, ceds_ext_years].openscm.groupby_except("sector").sum().pix.convert_unit({"kt CH4/yr": "Mt CH4/yr"})
    )
    ceds_extensions_l.append(ext_ts)

ceds_extensions = set_new_single_value_levels(
    pix.concat(ceds_extensions_l),
    {"model": "gridding-emissions-ceds-extension", "scenario": HISTORY_SCENARIO_NAME, "region": "World"},
)
ceds_extensions

# %%
tmp_t = ceds_extensions.reset_index("model", drop=True).align(
    history_for_gridding_harmonisation_aggregated.reset_index("model", drop=True)
)
history_for_gridding_harmonisation_aggregated_incl_ceds_extension = tmp_t[0].fillna(0.0) + tmp_t[1]
history_for_gridding_harmonisation_aggregated_incl_ceds_extension = (
    history_for_gridding_harmonisation_aggregated_incl_ceds_extension.pix.assign(
        model="gridding-emissions-incl-ceds-extension"
    )
)
# history_for_gridding_harmonisation_aggregated_incl_ceds_extension

# %%
for v in ["CH4", "N2O"]:
    pix.concat(
        [
            history_for_gridding_harmonisation_aggregated,
            history_for_gridding_harmonisation_aggregated_incl_ceds_extension,
        ]
    ).loc[pix.ismatch(variable=f"Emissions|{v}"), :].pix.project(["model", "variable"]).T.plot()
    plt.show()

# %% [markdown]
# ### GCB

# %%
gcb = GCB_PROCESSED_DB.load()
gcb.T.plot()


# %% [markdown]
# ### Extend back to 1750

# %% [markdown]
# We extend back by assuming exponential decay.
# We do this with a very basic geometric sequence.


# %%
def common_ratio_finder(x):
    """Calculate the common ratio to use for the extension"""
    return (gcb_pre_1850_cumulative_estimate / timestep - basic_extrap / (x ** (n - 1)) * (x**n - 1) / (x - 1)).m


# %%
gcb_pre_1850_cumulative_estimate = Q(30.0, "GtC")
gcb_pre_1850_cumulative_estimate.to("Mt CO2")

# %%
gcb_u_l = gcb.pix.unique("unit").tolist()
if len(gcb_u_l) != 1:
    raise AssertionError

gcb_u = gcb_u_l[0]

# %%
to_fill_years = np.setdiff1d(
    np.arange(1750, HARMONISATION_YEAR + 1),
    gcb.columns,
)
basic_extrap = Q((2 * gcb[1850] - gcb[1851]).iloc[0], gcb_u)
n = to_fill_years.size
timestep = Q(1, "yr")

# %%
sol = optimize.newton(common_ratio_finder, [1.05])

# %%
start = basic_extrap / (sol ** (n - 1))
# start

# %%
gcb_extension = pd.DataFrame(
    (start * sol ** np.arange(to_fill_years.size)).to(gcb_u).m[np.newaxis, :],
    columns=pd.Index(to_fill_years, name="year"),
    index=gcb.index,
)
# Check continuity
np.testing.assert_allclose(gcb_extension[to_fill_years[-1]], basic_extrap.m)
# Check sum
np.testing.assert_allclose(gcb_extension.sum(axis=1), (gcb_pre_1850_cumulative_estimate / timestep).to(gcb_u).m)

# %%
gcb_full = pix.concat([gcb_extension, gcb], axis="columns").sort_index(axis="columns").pix.assign(model="GCB-extended")
gcb_full.T.plot()
gcb_full

# %% [markdown]
# ## CMIP inversions

# %%
cmip_inversions = CMIP7_GHG_PROCESSED_DB.load()
# cmip_inversions


# %%
def get_extended_name(idf: pd.DataFrame) -> str:
    """Get extend name for output"""
    indf_model_l = idf.pix.unique("model")
    if len(indf_model_l) != 1:
        raise AssertionError(indf_model_l)

    indf_model = indf_model_l[0]

    return f"{indf_model}_cmip-inverse-extended"


def extend_with_inversions(indf: pd.DataFrame, cmip_inverse: pd.DataFrame) -> pd.DataFrame:
    """Extend with inversions of CMIP concentrations"""
    res_l = []
    for (variable, unit), vdf in indf.sort_index(axis="columns").groupby(["variable", "unit"]):
        cmip_inverse_vdf = cmip_inverse.loc[pix.isin(variable=variable)].pix.convert_unit(unit)
        vdf_no_nan = vdf.dropna(axis="columns")
        vdf_min_year = vdf_no_nan.columns.min()
        if vdf.shape[0] != 1:
            raise AssertionError

        if cmip_inverse_vdf.shape[0] != 1:
            raise AssertionError

        if vdf[vdf_min_year].iloc[0] > 0.0 and (cmip_inverse_vdf[vdf_min_year].iloc[0] > 0.0):
            ratio = vdf[vdf_min_year].iloc[0] / cmip_inverse_vdf[vdf_min_year].iloc[0]
        else:
            ratio = 0.0

        tmp = cmip_inverse_vdf.loc[:, : vdf_min_year - 1] * ratio
        tmp.index = vdf.index
        tmp = pix.concat([tmp, vdf_no_nan], axis="columns")

        res_l.append(tmp)

    res = pix.concat(res_l).pix.assign(model=get_extended_name(indf))

    return res


# %% [markdown]
# ### WMO 2022

# %%
wmo_2022 = WMO_2022_PROCESSED_DB.load()
# wmo_2022

# %%
# Not in CMIP inversions, probably should be added to CMIP dataset for CMIP8.
halon1202_locator = pix.ismatch(variable="**Halon1202")

wmo_2022_halon1202 = wmo_2022.loc[halon1202_locator]
wmo_2022_halon1202.loc[:, np.arange(1750, wmo_2022_halon1202.columns.min())] = 0.0
wmo_2022_halon1202 = wmo_2022_halon1202.sort_index(axis="columns").pix.assign(
    model=get_extended_name(wmo_2022_halon1202)
)
wmo_2022_halon1202

# %%
wmo_2022_ext_l = [wmo_2022_halon1202]
for model, mdf in wmo_2022.loc[~halon1202_locator].groupby("model"):
    wmo_2022_ext_l.append(extend_with_inversions(mdf, cmip_inversions))

wmo_2022_ext = pix.concat(wmo_2022_ext_l)
# wmo_2022_ext

# %%
pdf = pix.concat(
    [wmo_2022, wmo_2022_ext, cmip_inversions.loc[pix.isin(variable=wmo_2022.pix.unique("variable"))]]
).openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="model",
    style="model",
    col="variable",
    col_wrap=3,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# ### Velders et al., 2022

# %%
velders_et_al_2022_all = VELDERS_ET_AL_2022_PROCESSED_DB.load()
velders_et_al_2022 = velders_et_al_2022_all.loc[pix.ismatch(scenario="history")].dropna(axis="columns", how="all")
velders_et_al_2022

# %%
velders_et_al_2022_ext = extend_with_inversions(velders_et_al_2022, cmip_inversions)
# velders_et_al_2022_ext

# %%
pdf = pix.concat(
    [
        velders_et_al_2022,
        velders_et_al_2022_ext,
        cmip_inversions.loc[pix.isin(variable=velders_et_al_2022.pix.unique("variable"))],
    ]
).openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="model",
    style="model",
    col="variable",
    col_wrap=3,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# ### Adam et al., 2024

# %%
adam_et_al_2024 = ADAM_ET_AL_2024_PROCESSED_DB.load()
# adam_et_al_2024

# %%
adam_et_al_2024_ext = extend_with_inversions(adam_et_al_2024, cmip_inversions)
# adam_et_al_2024_ext

# %%
pdf = pix.concat(
    [
        adam_et_al_2024,
        adam_et_al_2024_ext,
        cmip_inversions.loc[pix.isin(variable=adam_et_al_2024.pix.unique("variable"))],
    ]
).openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="model",
    style="model",
    col="variable",
    col_wrap=3,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# ### Compile

# %%
all_sources = pix.concat(
    [
        history_for_gridding_harmonisation_aggregated_incl_ceds_extension,
        adam_et_al_2024_ext,
        cmip_inversions,
        gcb_full,
        velders_et_al_2022_ext,
        wmo_2022_ext,
    ]
)
all_sources.pix.unique("model")

# %%
global_variable_sources = {
    "Emissions|BC": "gridding-emissions-incl-ceds-extension",
    "Emissions|CF4": "WMO 2022 AGAGE inversions_cmip-inverse-extended",
    "Emissions|C2F6": "WMO 2022 AGAGE inversions_cmip-inverse-extended",
    "Emissions|C3F8": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|cC4F8": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C4F10": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C5F12": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C7F16": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C8F18": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C6F14": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|CH4": "gridding-emissions-incl-ceds-extension",
    "Emissions|CO": "gridding-emissions-incl-ceds-extension",
    "Emissions|CO2|Biosphere": "GCB-extended",
    "Emissions|CO2|Fossil": "gridding-emissions-incl-ceds-extension",
    "Emissions|HFC125": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC134a": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC143a": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC152a": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC227ea": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC23": "Adam et al., 2024_cmip-inverse-extended",
    "Emissions|HFC236fa": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC245fa": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC32": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC365mfc": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|HFC4310mee": "Velders et al., 2022_cmip-inverse-extended",
    "Emissions|CCl4": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CFC11": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CFC113": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CFC114": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CFC115": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CFC12": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CH2Cl2": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|CH3Br": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CH3CCl3": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CH3Cl": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|CHCl3": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|HCFC141b": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|HCFC142b": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|HCFC22": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|Halon1202": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|Halon1211": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|Halon1301": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|Halon2402": "WMO-2022-CMIP7-concentration-inversions_cmip-inverse-extended",
    "Emissions|N2O": "gridding-emissions-incl-ceds-extension",
    "Emissions|NF3": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|NH3": "gridding-emissions-incl-ceds-extension",
    "Emissions|NOx": "gridding-emissions-incl-ceds-extension",
    "Emissions|OC": "gridding-emissions-incl-ceds-extension",
    "Emissions|SF6": "WMO 2022 AGAGE inversions_cmip-inverse-extended",
    "Emissions|SO2F2": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|SOx": "gridding-emissions-incl-ceds-extension",
    "Emissions|NMVOC": "gridding-emissions-incl-ceds-extension",
}

# %%
to_reporting_names = partial(
    update_index_levels_func,
    updates={
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
global_workflow_harmonisation_emissions_l = []
for variable, source in global_variable_sources.items():
    source_loc = pix.isin(variable=[variable]) & pix.isin(model=[source])
    to_keep = all_sources.loc[source_loc]
    if to_keep.empty:
        msg = f"{variable} not available from {source}"
        raise AssertionError(msg)

    global_workflow_harmonisation_emissions_l.append(to_keep)

global_workflow_harmonisation_emissions = (
    (pix.concat(global_workflow_harmonisation_emissions_l).sort_index().sort_index(axis="columns"))
    .loc[:, :HARMONISATION_YEAR]
    .pix.assign(scenario=HISTORY_SCENARIO_NAME)
)

global_workflow_harmonisation_emissions_reporting_names = to_reporting_names(global_workflow_harmonisation_emissions)
global_workflow_harmonisation_emissions_reporting_names = update_index_levels_func(
    global_workflow_harmonisation_emissions_reporting_names, {"unit": lambda x: x.replace("HFC4310mee", "HFC4310")}
)

exp_n_timeseries = 52
if global_workflow_harmonisation_emissions.shape[0] != exp_n_timeseries:
    raise AssertionError

global_workflow_harmonisation_emissions_reporting_names = (
    global_workflow_harmonisation_emissions_reporting_names.rename_axis("year", axis="columns")
)
# global_workflow_harmonisation_emissions_reporting_names

# %% [markdown]
# ## Last checks

# %%
if global_workflow_harmonisation_emissions_reporting_names[HARMONISATION_YEAR].isnull().any():
    missing = global_workflow_harmonisation_emissions_reporting_names.loc[
        global_workflow_harmonisation_emissions_reporting_names[HARMONISATION_YEAR].isnull()
    ]

    display(missing)  # noqa: F821
    raise AssertionError

# %% [markdown]
# ## Compare with RCMIP

# %%
reporting_to_rcmip = partial(
    convert_variable_name,
    to_convention=SupportedNamingConventions.RCMIP,
    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
)
rcmip_to_reporting = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.RCMIP,
    to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
)

rcmip_hist_l = []
for scenario, endyear in [("ssp245", 2015), ("rcp45", 2005)]:
    rcmip_hist_l.append(
        RCMIP_PROCESSED_DB.load(
            pix.isin(
                region="World",
                scenario=scenario,
                variable=global_workflow_harmonisation_emissions_reporting_names.pix.unique("variable").map(
                    reporting_to_rcmip
                ),
            ),
            progress=True,
        ).loc[:, :endyear]
    )

rcmip_hist = pix.concat(rcmip_hist_l).sort_index(axis="columns")

rcmip_hist = rcmip_hist.openscm.update_index_levels({"variable": rcmip_to_reporting})
# rcmip_hist

# %%
pdf = (
    pix.concat([rcmip_hist, global_workflow_harmonisation_emissions_reporting_names])
    .loc[:, :HARMONISATION_YEAR]
    .openscm.to_long_data()
    .dropna()
)

fg = sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="scenario",
    col="variable",
    col_order=sorted(pdf["variable"].unique()),
    col_wrap=4,
    kind="line",
    facet_kws=dict(sharey=False),
)
for ax in fg.axes.flatten():
    ax.set_ylim(ymin=0.0)

fg.fig.savefig("global-workflow-history-over-cmip-phases.pdf", bbox_inches="tight")

# %% [markdown]
# ## Save

# %% editable=true slideshow={"slide_type": ""}
out_file = HISTORY_HARMONISATION_INTERIM_DIR / f"global-workflow-history_{HISTORY_FOR_HARMONISATION_ID}.feather"
out_file.parent.mkdir(exist_ok=True, parents=True)

global_workflow_harmonisation_emissions_reporting_names.reorder_levels(
    [
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
    ]
).to_feather(out_file)
