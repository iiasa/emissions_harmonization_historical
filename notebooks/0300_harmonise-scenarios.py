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
# # Harmonise scenarios

# %%
import logging
import multiprocessing
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import scipy.stats
import seaborn as sns
import tqdm.autonotebook as tqdman
from gcages.harmonisation import Harmoniser
from gcages.pre_processing import PreProcessor
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
)
from emissions_harmonization_historical.io import load_csv

# %%
# Disable all logging to avoid a million messages
logging.disable()

# %%
SCENARIO_TIME_ID = "20250122-140031"

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"


# %%
history = strip_pint_incompatible_characters_from_units(load_csv(HISTORICAL_GLOBAL_COMPOSITE_PATH))
history_cut = history.loc[:, 1990:2025]
history_cut

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
ar6_history = rcmip_clean.loc[
    pix.ismatch(mip_era="CMIP6") & pix.ismatch(scenario="historical") & pix.ismatch(region="World")
].reset_index(["mip_era", "activity_id"], drop=True)
ar6_history

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
scenario_files = tuple(SCENARIO_PATH.glob(f"{SCENARIO_TIME_ID}__scenarios-scenariomip__*.csv"))
if not scenario_files:
    msg = f"Check your scenario ID. {list(SCENARIO_PATH.glob('*.csv'))=}"
    raise AssertionError(msg)

scenario_files[:5]

# %%
scenarios_raw = pix.concat([load_csv(f) for f in tqdman.tqdm(scenario_files)]).sort_index(axis="columns")
scenarios_raw_global = scenarios_raw.loc[
    pix.ismatch(region="World")
    # & pix.isin(variable=history_cut.pix.unique("variable"))
]
scenarios_raw_global

# %%
# scenarios_raw.loc[pix.ismatch(variable="Emissions|VOC")].pix.unique(["variable", "unit"])

# %%
# What are we still missing.
# Notes:
# - CO2 hierarchy processing is below.
# - drop out CO2 AFOLU NGHGI from harmonisation (?)
# - can we drop out the totals e.g. HFC, PFC, Kyoto Gases?
#   does anyone provide these but not the components?
#   (If yes, this is fiddlier because you probably want to harmonise,
#   then infill directly using the totals as the lead gas,
#   which would be new.)
still_missing = sorted(
    scenarios_raw.loc[
        (
            pix.ismatch(variable="Emissions|*")
            | pix.ismatch(variable="Emissions|CO2|*")
            | pix.ismatch(variable="Emissions|F-Gases**")
            | pix.ismatch(variable="Emissions|HFC**")
            | pix.ismatch(variable="Emissions|PFC**")
        )
        & ~(pix.ismatch(variable="Emissions|Kyoto Gases") | pix.ismatch(variable="Emissions|CO2|AFOLU [NGHGI]"))
    ]
    .pix.unique("variable")
    .difference(history_cut.pix.unique("variable"))
)
still_missing

# %%
# Checking need for CO2 aggregation
tmp = scenarios_raw.loc[
    pix.ismatch(region="World")
    & pix.ismatch(variable="Emissions|CO2|*")
    & ~pix.ismatch(variable="Emissions|CO2|AFOLU*")
]
for (model, scenario), msdf in tmp.groupby(["model", "scenario"]):
    if "Emissions|CO2|Energy and Industrial Processes" not in msdf.pix.unique("variable"):
        msg = "Emissions|CO2|Energy and Industrial Processes missing"
        print(f"{model=} {scenario=}")
        raise NotImplementedError(msg)

# %%
# pandas-indexing is so well done
# scenarios_raw_global.pix.extract(
#     variable="Emissions|{species}|{sector}|{subsector}", dropna=False, keep=True
# ).index.to_frame(index=False)

# %%
scenarios_raw_global.pix.unique(["model", "scenario"]).to_frame(index=False)


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
make_all_var_plot = partial(
    sns.relplot,
    x="year",
    y="value",
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
)

# %%
make_all_var_plot(
    data=get_sns_df(scenarios_raw.loc[pix.isin(variable=["Emissions|CO2|AFOLU [NGHGI]"]) & pix.isin(region="World")]),
    kind="line",
    hue="scenario",
    style="model",
)

# %%
fg = make_all_var_plot(
    data=get_sns_df(
        pix.concat(
            [
                history_cut.pix.assign(mip_era="cmip7"),
                ar6_history.loc[:, history_cut.columns]
                .loc[pix.isin(variable=history_cut.pix.unique("variable"))]
                .pix.assign(mip_era="cmip6"),
            ]
        )
    ),
    kind="line",
    hue="mip_era",
    style="model",
)
for ax in fg.axes.flatten():
    ax.set_ylim(ymin=0)

# %%
to_infill = sorted(
    [
        v
        for v in ar6_history.pix.unique("variable").difference(scenarios_raw_global.pix.unique("variable"))
        if not any(s in v for s in ["|BC|", "|CH4|", "|CO2|", "|CO|", "|NH3|", "|NOx|", "|OC|", "|Sulfur|", "|VOC|"])
    ]
)
print(f"{len(to_infill)=}")
to_infill

# %%
rcmip_infill_vars = rcmip_clean.loc[pix.isin(variable=to_infill) & pix.ismatch(scenario="ssp*")]
rcmip_infill_vars

# %%
fg = make_all_var_plot(
    data=get_sns_df(rcmip_infill_vars),
    kind="line",
    hue="scenario",
    # style="model",
)
for ax in fg.axes.flatten():
    ax.set_ylim(ymin=0)

# %%
rcmip_helper = (
    rcmip_clean.loc[pix.ismatch(scenario="ssp*") & pix.ismatch(region="World")]
    .dropna(axis="columns", how="all")
    .reset_index(["mip_era", "activity_id"], drop=True)
)
rcmip_helper

# %%
for v_check in to_infill:
    v_loc = pix.isin(variable=v_check)
    v_values = rcmip_helper.loc[v_loc].reset_index(["variable", "unit"], drop=True)

    if (v_values.loc[:, 2015:] == 0.0).all().all():
        print(f"All future RCMIP values are zero for {v_check}")
        continue

    potential_leads = rcmip_helper.loc[
        ~v_loc & pix.isin(variable=scenarios_raw_global.pix.unique("variable")) & ~pix.ismatch(variable="**Burning**")
    ]
    potential_leads = potential_leads.loc[~(potential_leads == 0.0).all(axis="columns")]
    quotient = potential_leads.divide(v_values, axis="rows").loc[:, 2020:2090]
    quotient_stds = quotient.loc[:, np.isfinite(quotient).any()].groupby("variable").std()
    # lead = quotient_stds.loc[~(quotient==0.0).all(axis="columns"), :].idxmin(axis="rows", skipna=True)
    lead = quotient_stds.idxmin(axis="rows", skipna=True)
    if len(lead.unique()) > 1:
        print(f"Ambiguous lead for {v_check}, could be\n{lead}")
        make_all_var_plot(
            data=get_sns_df(rcmip_helper.loc[pix.isin(variable=[v_check, *lead.unique()])].loc[:, 2015:2100]),
            kind="line",
            hue="scenario",
            # style="model",
        )
        plt.show()
        continue
        # raise AssertionError(lead)

    lead = lead.unique()[0]
    print(f"{v_check} follows {lead}")
    make_all_var_plot(
        data=get_sns_df(rcmip_helper.loc[pix.isin(variable=[v_check, lead])].loc[:, 2015:2100]),
        kind="line",
        hue="scenario",
        # style="model",
    )
    plt.show()
    # break

# %%
montreal_vars = rcmip_helper.loc[pix.ismatch(variable="**Montreal**")].pix.unique("variable").tolist()
montreal_vars

# %%
for v_check in montreal_vars:
    v_loc = pix.isin(variable=v_check)
    v_values = rcmip_helper.loc[v_loc].reset_index(["variable", "unit"], drop=True)

    if (v_values.loc[:, 2015:] == 0.0).all().all():
        print(f"All future RCMIP values are zero for {v_check}")
        continue

    potential_leads = rcmip_helper.loc[~v_loc & pix.isin(variable=montreal_vars)]
    potential_leads = potential_leads.loc[~(potential_leads == 0.0).all(axis="columns")]
    quotient = potential_leads.divide(v_values, axis="rows").loc[:, 2020:2090]
    quotient_stds = quotient.loc[:, np.isfinite(quotient).any()].groupby("variable").std()

    # lead = quotient_stds.loc[~(quotient_stds==0.0).any(axis="columns"), :].idxmin(axis="rows", skipna=True)
    # lead = quotient_stds.idxmin(axis="rows", skipna=True)
    def get_std(g):
        # display(g)
        return np.std((g.values - g.values[0][0]) / g.values[0][0])

    lead = quotient.groupby("variable").apply(get_std).idxmin()

    #     if len(lead.unique()) > 1:
    #         print(f"Ambiguous lead for {v_check}, could be\n{lead}")
    #         make_all_var_plot(
    #     data=get_sns_df(rcmip_helper.loc[pix.isin(variable=[v_check, *lead.unique()])].loc[:, 2015:2100]),
    #     kind="line",
    #     hue="scenario",
    #     # style="model",
    # )
    #         plt.show()
    #         continue
    #         # raise AssertionError(lead)

    # lead = lead.unique()[0]
    print(f"{v_check} follows {lead}")
    make_all_var_plot(
        data=get_sns_df(rcmip_helper.loc[pix.isin(variable=[v_check, lead])].loc[:, 2015:2100]),
        kind="line",
        hue="scenario",
        # style="model",
    )
    plt.show()
    # break

# %%
quotient.loc[pix.isin(variable=[lead])]

# %%
rcmip_infill_vars = rcmip_clean.loc[pix.isin(variable=to_infill) & pix.ismatch(scenario="ssp*")]
rcmip_infill_vars

# %%
[
    # 'Emissions|C3F8',
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C7F16",
    "Emissions|C8F18",
    # 'Emissions|Montreal Gases|CCl4',
    # 'Emissions|Montreal Gases|CFC|CFC11',
    # 'Emissions|Montreal Gases|CFC|CFC113',
    # 'Emissions|Montreal Gases|CFC|CFC114',
    # 'Emissions|Montreal Gases|CFC|CFC115',
    # 'Emissions|Montreal Gases|CFC|CFC12',
    # 'Emissions|Montreal Gases|CH2Cl2',
    # 'Emissions|Montreal Gases|CH3Br',
    # 'Emissions|Montreal Gases|CH3CCl3',
    # 'Emissions|Montreal Gases|CH3Cl',
    # 'Emissions|Montreal Gases|CHCl3',
    # 'Emissions|Montreal Gases|HCFC141b',
    # 'Emissions|Montreal Gases|HCFC142b',
    # 'Emissions|Montreal Gases|HCFC22',
    "Emissions|Montreal Gases|Halon1202",
    # 'Emissions|Montreal Gases|Halon1211',
    # 'Emissions|Montreal Gases|Halon1301',
    # 'Emissions|Montreal Gases|Halon2402',
    # 'Emissions|NF3',
    # 'Emissions|SO2F2',
    "Emissions|cC4F8",
    # # Guus
    # 'Emissions|HFC|HFC152a',
    # 'Emissions|HFC|HFC236fa',
    # 'Emissions|HFC|HFC365mfc',
]

# %%
# make_all_var_plot(
#     data=get_sns_df(scenarios_raw_global),
#     kind="scatter",
#     hue="scenario",
#     style="model",
# )

# %%
# for model, mdf in scenarios_raw_global.groupby("model"):
#     pdf = pd.concat([get_sns_df(mdf), get_sns_df(history_cut)])
#     make_all_var_plot(
#         data=pdf,
#         kind="line",
#         hue="scenario",
#         style="model",
#     )
#     plt.show()

# %%
pre_processor = PreProcessor(emissions_out=tuple(history.pix.unique("variable")))

# %%
# As at 2024-01-13, just the list from AR6.
# We can tweak from here.
aneris_overrides = pd.DataFrame(
    [
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|BC'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C2F6",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C6F14",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|CF4",
        },  # high historical variance (cov=11.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|CO",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2080",
            "variable": "Emissions|CO2",
        },  # always ratio method by choice
        {
            # high historical variance,
            # but using offset method to prevent diff from increasing
            # when going negative rapidly (cov=23.2)
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'},
        {
            "method": "constant_ratio",
            "variable": "Emissions|F-Gases",
        },  # basket not used in infilling (sum of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC",
        },  # basket not used in infilling (sum of subset of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC125",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC134a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC143a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC227ea",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC23",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC32",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC43-10",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|OC",
        },  # high historical variance (cov=18.5)
        {
            "method": "constant_ratio",
            "variable": "Emissions|SF6",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|VOC",
        },  # high historical variance (cov=12.0)
    ]
)

# %%
# TODO: discuss and think through better
harmonisation_year = 2021
calc_scaling_year = 2015

# %%
history_values = history_cut.loc[:, calc_scaling_year:harmonisation_year].copy()

# TODO: decide which variables exactly to use averaging with
high_variability_variables = (
    "Emissions|BC",
    "Emissions|CO",
    # # Having looked at the data, I'm not sure I would do this for CO2 AFOLU
    # "Emissions|CO2|AFOLU",
    "Emissions|OC",
    "Emissions|VOC",
)
n_years_for_regress = 10
for high_variability_variable in high_variability_variables:
    regress_vals = history_cut.loc[
        pix.ismatch(variable=high_variability_variable),
        harmonisation_year - n_years_for_regress + 1 : harmonisation_year,
    ]
    regress_res = scipy.stats.linregress(x=regress_vals.columns, y=regress_vals.values)
    regressed_value = regress_res.slope * harmonisation_year + regress_res.intercept

    # Should somehow keep track that we've done this
    history_values.loc[pix.ismatch(variable=high_variability_variable), harmonisation_year] = regressed_value

history_values

# %%
harmoniser = Harmoniser(
    historical_emissions=history_values,
    harmonisation_year=harmonisation_year,
    calc_scaling_year=calc_scaling_year,
    aneris_overrides=aneris_overrides,
    # n_processes=1,
    n_processes=multiprocessing.cpu_count(),
    # TODO: implement and enable
    run_checks=False,
)

# %%
pre_processed = pre_processor(scenarios_raw_global)

# %%
# TODO: move this into pre-processing
pre_processed_l = []
for (model, scenario), msdf in pre_processed.groupby(["model", "scenario"]):
    if "Emissions|CO2|Energy and Industrial Processes" not in msdf.pix.unique("variable"):
        print(
            f"Excluding {model=} {scenario=} because there are no CO2 fossil emissions. "
            f"Available variables: {sorted(msdf.pix.unique('variable').tolist())}."
        )
        continue

    all_nan_or_zero = (msdf.isnull() | (msdf == 0.0)).all(axis=1)
    if all_nan_or_zero.any():
        print(f"Dropping out all nan or zero:\n{msdf[all_nan_or_zero].index.to_frame(index=False)}")

    msdf_use = msdf[~all_nan_or_zero]

    pre_processed_l.append(msdf_use)

pre_processed = pix.concat(pre_processed_l)
pre_processed

# %%
harmonised = harmoniser(pre_processed)

# %%
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colours

# %%
for model, mdf in pix.concat(
    [
        scenarios_raw_global.pix.assign(stage="raw"),
        harmonised.pix.assign(stage="harmonised"),
    ]
).groupby("model"):
    variables_to_show = mdf.pix.unique("variable").intersection(history_cut.pix.unique("variable"))
    pdf_wide = pix.concat([mdf, history_cut.pix.assign(stage="history")]).loc[pix.isin(variable=variables_to_show)]
    # pdf_wide = pdf_wide.loc[:, 2010: 2028]
    palette = {k: colours[i % len(colours)] for i, k in enumerate(pdf_wide.pix.unique("scenario"))}
    fg = make_all_var_plot(
        data=get_sns_df(pdf_wide),
        kind="line",
        hue="scenario",
        hue_order=sorted(pdf_wide.pix.unique("scenario")),
        palette={**palette, "history": "black"},
        style="stage",
    )
    fg.fig.suptitle(model, y=1.01)
    plt.show()
    # break

# %%
