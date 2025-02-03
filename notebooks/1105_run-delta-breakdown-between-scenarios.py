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
# # Calculate breakdown of deltas between scenarios

# %% [markdown]
# ## Imports

# %%
import json
import logging
import multiprocessing
import os
import re

import matplotlib.pyplot as plt
import openscm_runner.adapters
import pandas as pd
import pandas_indexing as pix
from gcages.database import GCDB
from gcages.io import load_timeseries_csv
from gcages.pandas_helpers import multi_index_lookup
from gcages.post_processing import PostProcessor
from gcages.scm_running import SCMRunner, convert_openscm_runner_output_names_to_magicc_output_names
from gcages.units_helpers import strip_pint_incompatible_characters_from_units
from loguru import logger

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
n_processes = multiprocessing.cpu_count()
run_checks = False  # TODO: turn on

# %%
INPUT_PATH = DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"

# %%
magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
magicc_expected_version = "v7.6.0a3"
magicc_prob_distribution_path = (
    DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

# %%
os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
if openscm_runner.adapters.MAGICC7.get_version() != magicc_expected_version:
    raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())

# %%
OUTPUT_PATH = (
    INPUT_PATH
    / f"magicc-{magicc_expected_version.replace('.', '-')}_{magicc_prob_distribution_path.stem}_scenario-delta-breakdown"  # noqa: E501
)
OUTPUT_PATH

# %% [markdown]
# ## Load complete scenario data

# %%
complete_scenarios = load_timeseries_csv(
    INPUT_PATH / "complete_scenarios.csv",
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
complete_scenarios

# %% [markdown]
# ## Calculate breakdown scenario runs

# %%
base = pd.MultiIndex.from_tuples(
    (("REMIND-MAgPIE 3.4-4.8", "SSP1 - Very Low Emissions"),),
    name=["model", "scenario"],
)
base

# %%
others = pd.MultiIndex.from_tuples(
    (
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Overshoot"),
        ("IMAGE 3.4", "SSP1 - Low Emissions"),
    ),
    name=["model", "scenario"],
)
others

# %%
base_scen = multi_index_lookup(complete_scenarios, base)
base_model = base.pix.unique("model")[0]
base_scenario = base.pix.unique("scenario")[0]

# %%
to_attribute = [
    # (label, variables to include)
    ("CO2 Fossil", ["Emissions|CO2|Energy and Industrial Processes"]),
    ("CO2 AFOLU", ["Emissions|CO2|AFOLU"]),
    ("CH4", ["Emissions|CH4"]),
    ("N2O", ["Emissions|N2O"]),
    ("BC OC Sulfur", ["Emissions|BC", "Emissions|OC", "Emissions|Sulfur"]),
    ("NH3 NOx CO VOCs", ["Emissions|NH3", "Emissions|NOx", "Emissions|CO", "Emissions|VOC"]),
    ("Montreal gases", [v for v in base_scen.pix.unique("variable") if "Montreal" in v]),
    (
        "HFCs PFCs SF6 NF3 SO2F2",
        [
            *[v for v in base_scen.pix.unique("variable") if "HFC" in v],
            "Emissions|CF4",
            *[v for v in base_scen.pix.unique("variable") if re.match(r"Emissions\|c?C\d*F\d*", v)],
            "Emissions|SF6",
            "Emissions|NF3",
            "Emissions|SO2F2",
        ],
    ),
]
to_attribute

# %%
not_attributed = set(base_scen.pix.unique("variable"))
for v in to_attribute:
    missing = set(v[1]).difference(not_attributed)
    if missing:
        raise AssertionError(missing)

    not_attributed = not_attributed - set(v[1])

if not_attributed:
    raise AssertionError(not_attributed)

# %%
to_run_l = []
for i in range(others.size):
    other_idx = others[[i]]
    for label, emms in to_attribute:
        model = other_idx[0][other_idx.names.index("model")]
        scenario = other_idx[0][other_idx.names.index("scenario")]

        other_idx = others[[i]]
        variable_loc = pix.isin(variable=emms)
        start = multi_index_lookup(complete_scenarios, other_idx).loc[~variable_loc]
        replace = base_scen.loc[variable_loc]
        to_run_tmp = pix.concat([start, replace]).pix.assign(
            model=f"{model} -- {scenario} --- {base_model} -- {base_scenario}",
            # model=f"{model} {scenario} - {base_model} {base_scenario}".replace(".", "_").replace(" ", "_"),
            scenario=label,
        )
        exp_n_ts = 52
        if to_run_tmp.shape[0] != exp_n_ts:
            raise AssertionError

        to_run_l.append(to_run_tmp)

to_run = pix.concat(to_run_l)
to_run

# %%
to_run.pix.unique(["model", "scenario"]).to_frame(index=False)

# %% [markdown]
# ## Run SCMs

# %%
batch_size_scenarios = 15
scm_output_variables = (
    # GSAT
    "Surface Air Temperature Change",
    # # GMST
    # "Surface Air Ocean Blended Temperature Change",
    # ERFs
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|Aviation|Cirrus",
    "Effective Radiative Forcing|Aviation|Contrail",
    "Effective Radiative Forcing|Aviation|H2O",
    "Effective Radiative Forcing|Black Carbon on Snow",
    # 'Effective Radiative Forcing|CH4 Oxidation Stratospheric',
    "CH4OXSTRATH2O_ERF",
    "Effective Radiative Forcing|Land-use Change",
    # "Effective Radiative Forcing|CFC11",
    # "Effective Radiative Forcing|CFC12",
    # "Effective Radiative Forcing|HCFC22",
    # "Effective Radiative Forcing|HFC125",
    # "Effective Radiative Forcing|HFC134a",
    # "Effective Radiative Forcing|HFC143a",
    # "Effective Radiative Forcing|HFC227ea",
    # "Effective Radiative Forcing|HFC23",
    # "Effective Radiative Forcing|HFC245fa",
    # "Effective Radiative Forcing|HFC32",
    # "Effective Radiative Forcing|HFC4310mee",
    # "Effective Radiative Forcing|CF4",
    # "Effective Radiative Forcing|C6F14",
    # "Effective Radiative Forcing|C2F6",
    # "Effective Radiative Forcing|SF6",
    # # Heat uptake
    # "Heat Uptake",
    # "Heat Uptake|Ocean",
    # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # # Carbon cycle
    # "Net Atmosphere to Land Flux|CO2",
    # "Net Atmosphere to Ocean Flux|CO2",
    # # permafrost
    # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)

# %% [markdown]
# ### MAGICC stuff

# %%
startyear = 1750
endyear = 2100
out_dynamic_vars = [
    f"DAT_{v}" for v in convert_openscm_runner_output_names_to_magicc_output_names(scm_output_variables)
]

# %%
scm_results_db = GCDB(OUTPUT_PATH / "db")
scm_results_db

# %%
# If you need to re-write.
# scm_results_db.delete()
# scm_results_db.load_metadata()

# %%
with open(magicc_prob_distribution_path) as fh:
    cfgs_raw = json.load(fh)

base_cfgs = [
    {
        "run_id": c["paraset_id"],
        **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
    }
    for c in cfgs_raw["configurations"]
]


common_cfg = {
    "startyear": startyear,
    "endyear": endyear,
    "out_dynamic_vars": out_dynamic_vars,
    "out_ascii_binary": "BINARY",
    "out_binary_format": 2,
}


run_config = [
    {
        **common_cfg,
        **base_cfg,
    }
    for base_cfg in base_cfgs
]

scm_runner = SCMRunner(
    climate_models_cfgs={"MAGICC7": run_config},
    output_variables=scm_output_variables,
    force_interpolate_to_yearly=True,
    db=scm_results_db,
    res_column_type=int,
    run_checks=False,  # TODO: turn on
    n_processes=n_processes,
)


# %% [markdown]
# Add in data from the end of MAGICC's internal historical emissions.

# %%
# TODO: split this out into a function

# %%
MAGICC_FORCE_START_YEAR = 2015

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = DATA_ROOT / "global-composite" / f"cmip7_history_world_{COMBINED_HISTORY_ID}.csv"
history = strip_pint_incompatible_characters_from_units(
    load_timeseries_csv(
        HISTORICAL_GLOBAL_COMPOSITE_PATH,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
)

# %%
history_cut = history.loc[:, MAGICC_FORCE_START_YEAR : to_run.columns.min() - 1]
if history_cut.isnull().any().any():
    raise AssertionError

exp_n_variables = 52
if history_cut.shape[0] != exp_n_variables:
    raise AssertionError

history_cut

# %%
a, b = history_cut.reset_index(["model", "scenario"], drop=True).align(to_run)
to_run = pix.concat(
    [a.dropna(how="all", axis="columns"), b.dropna(how="all", axis="columns")], axis="columns"
).sort_index(axis="columns")

# Interpolate so it's clearer what went into MAGICC
to_run = to_run.T.interpolate("index").T
if to_run.isnull().any().any():
    if not scm_runner.force_interpolate_to_yearly:
        raise AssertionError

to_run

# %% [markdown]
# ### Run

# %%
scm_results = scm_runner(to_run, batch_size_scenarios=batch_size_scenarios)
scm_results

# %% [markdown]
# ## Post-process

# %%
post_processor = PostProcessor(
    gsat_variable_name="Surface Air Temperature Change",
    gsat_in_line_with_assessment_variable_name="Assessed Surface Air Temperature Change",
    gsat_assessment_median=0.85,
    gsat_assessment_time_period=range(1995, 2014 + 1),
    gsat_assessment_pre_industrial_period=range(1850, 1900 + 1),
    percentiles_to_calculate=(0.05, 0.33, 0.5, 0.67, 0.95),
    exceedance_global_warming_levels=(1.5, 2.0, 2.5),
    run_checks=run_checks,
)

# %%
post_processed = post_processor(scm_results)
post_processed.metadata.sort_values(["category", "Peak warming 33.0"])

# %%
md_split = post_processed.metadata.copy()
md_split = md_split.pix.extract(
    model="{original_model} -- {original_scenario} --- {replacement_model} -- {replacement_scenario}"
)
md_split = md_split.pix.format(
    original="{original_model} - {original_scenario}", replacement="{replacement_model} - {replacement_scenario}"
)
md_split = md_split.pix.format(component="{scenario}", drop=True)
md_split

# %%
md_normal_workflow_all = pd.read_csv(
    INPUT_PATH / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0" / "metadata.csv"
).set_index(["model", "scenario"])
md_normal_workflow_all.index = md_normal_workflow_all.index.rename(
    {"model": "original_model", "scenario": "original_scenario"}
)
md_normal_workflow = multi_index_lookup(
    md_normal_workflow_all,
    md_split.index.droplevel(md_split.index.names.difference(md_normal_workflow_all.index.names)).drop_duplicates(),
)
md_normal_workflow

# %%
md_normal_workflow

# %%
columns = ["Peak warming 50.0"]
deltas = md_normal_workflow[columns].subtract(md_split[columns], axis="rows")
deltas

# %%
deltas_components_total = (
    deltas.groupby(deltas.index.names.difference(["component"])).sum().pix.assign(component="total")
)
deltas_components_total

# %%
md_base = multi_index_lookup(
    md_normal_workflow_all, base.rename({"model": "original_model", "scenario": "original_scenario"})
)
md_base

# %%
deltas_total = md_normal_workflow[columns] - md_base.iloc[0, :][columns]
deltas_total

# %%
deltas_residual = (deltas_total - deltas_components_total).pix.assign(component="residual")
deltas_residual

# %%
deltas_closed_sum = pix.concat([deltas, deltas_residual])
deltas_closed_sum.groupby(deltas.index.names.difference(["component"])).sum()

# %%
# sorted(deltas_closed_sum.pix.unique("component"))

# %%
fig, ax = plt.subplots(figsize=(16, 4))

legend_order = [
    "Total",
    "CO2 Fossil",
    "CO2 AFOLU",
    "CH4",
    "N2O",
    "BC OC Sulfur",
    "NH3 NOx CO VOCs",
    "Montreal gases",
    "HFCs PFCs SF6 NF3 SO2F2",
    "residual",
]
palette = {
    "CO2 Fossil": "tab:red",
    "CO2 AFOLU": "tab:green",
    "CH4": "tab:orange",
    "N2O": "darkgreen",
    "BC OC Sulfur": "tab:gray",
    "NH3 NOx CO VOCs": "tab:olive",
    "Montreal gases": "tab:blue",
    "HFCs PFCs SF6 NF3 SO2F2": "tab:purple",
    "residual": "k",
}
bar_width = 0.8
x_ticks = [
    f"{model}\n{scenario}" for (model, scenario), _ in deltas_total.groupby(["original_model", "original_scenario"])
]
locs = {label: i + 0.5 for i, label in enumerate(x_ticks)}

for i, (label, loc) in enumerate(locs.items()):
    model, scenario = label.split("\n")
    ax.plot(
        [i + (0.5 - bar_width / 2), i + 1 - (0.5 - bar_width / 2)],
        [deltas_total.loc[(model, scenario)], deltas_total.loc[(model, scenario)]],
        label="Total" if i < 1 else None,
        color="pink",
        # linestyle="--",
        zorder=3.0,
        linewidth=2.0,
    )

    value_to_plot = "Peak warming 50.0"

    deltas_bar = deltas_closed_sum.loc[pix.isin(original_model=model, original_scenario=scenario)]
    negative_components = deltas_bar[deltas_bar < 0.0][value_to_plot].dropna().sort_values(ascending=False)
    positive_components = deltas_bar[deltas_bar >= 0.0][value_to_plot].dropna().sort_values(ascending=False)
    negative_sum = negative_components.sum()

    bottom = negative_sum
    for r in range(negative_components.size):
        neg_row = negative_components.iloc[r]
        component = negative_components.index.get_level_values("component")[r]
        ax.bar(
            loc,
            -neg_row,
            width=bar_width,
            bottom=bottom,
            color=palette[component],
            label=component if i < 1 else None,
        )
        bottom -= neg_row

    for r in range(positive_components.size):
        pos_row = positive_components.iloc[r]
        component = positive_components.index.get_level_values("component")[r]
        ax.bar(
            loc,
            pos_row,
            bottom=bottom,
            color=palette[component],
            label=component if i < 1 else None,
        )
        bottom += pos_row

ax.set_xticks(list(locs.values()))
ax.set_xticklabels(x_ticks, rotation=0)
base_model = base[0][base.names.index("model")]
base_scenario = base[0][base.names.index("scenario")]
ax.set_title(f"Relative to\n{base_model}\n{base_scenario}")
ax.axhline(0.0, color="tab:gray")


handles, labels = plt.gca().get_legend_handles_labels()
new_order = [labels.index(v) for v in legend_order]
ax.legend(
    [handles[idx] for idx in new_order],
    [labels[idx] for idx in new_order],
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
)

plt.show()
