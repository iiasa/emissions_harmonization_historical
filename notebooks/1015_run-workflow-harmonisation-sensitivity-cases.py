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
# # Run workflow - harmonisation sensitivity cases
#
# Run the climate assessment workflow with our updated processing
# for harmonisation sensitivity cases of interest.

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing

import pandas as pd
import pandas_indexing as pix
from gcages.pandas_helpers import multi_index_lookup
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
    WORKFLOW_ID,
)
from emissions_harmonization_historical.harmonisation import AR7FTHarmoniser
from emissions_harmonization_historical.io import load_global_scenario_data
from emissions_harmonization_historical.workflow import run_magicc_and_post_processing, run_workflow_up_to_infilling

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## General set up

# %%
OUTPUT_PATHS = {
    "co2_afolu_2050": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow-harmonise-co2-afolu-2050",
    "all_2050": DATA_ROOT
    / "climate-assessment-workflow"
    / "output"
    / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow-harmonise-all-2050",
}
OUTPUT_PATHS

# %%
n_processes = multiprocessing.cpu_count()
run_checks = False  # TODO: turn on

# %% [markdown]
# ## Load scenario data

# %%
scenarios_raw_global = load_global_scenario_data(
    scenario_path=DATA_ROOT / "scenarios" / "data_raw",
    scenario_time_id=SCENARIO_TIME_ID,
    progress=True,
).loc[:, :2100]  # TODO: drop 2100 end once we have usable scenario data post-2100

# %% [markdown]
# ## Run workflow up to infilling

# %%
harmoniser_co2_afolu_2050 = AR7FTHarmoniser.from_default_config(data_root=DATA_ROOT)
harmoniser_co2_afolu_2050.aneris_overrides.loc[
    harmoniser_co2_afolu_2050.aneris_overrides["variable"] == "Emissions|CO2|AFOLU",
    "method",
] = "reduce_offset_2050"

# %%
harmoniser_all_2050 = AR7FTHarmoniser.from_default_config(data_root=DATA_ROOT)
harmoniser_all_2050.aneris_overrides = pd.DataFrame(
    [
        {
            "method": "reduce_ratio_2050"
            if not any(vv in v for vv in ["HFC", "PFC", "SF6", "CF4", "C2F6", "C6F14"])
            else "constant_ratio",
            "variable": v,
        }
        for v in sorted(
            scenarios_raw_global.loc[pix.ismatch(variable="Emissions|*") | pix.ismatch(variable="**HFC**")].pix.unique(
                "variable"
            )
        )
    ]
)
tweaks = {
    "Emissions|CO2|Energy and Industrial Processes": "reduce_ratio_2050",
    "Emissions|CO2|AFOLU": "reduce_offset_2050",
}
tmp = harmoniser_all_2050.aneris_overrides.set_index("variable")
for variable, method in tweaks.items():
    tmp.loc[
        variable,
        "method",
    ] = method


harmoniser_all_2050.aneris_overrides = tmp.reset_index()
harmoniser_all_2050.aneris_overrides

# %%
harmonisation_choices = {
    "co2_afolu_2050": harmoniser_co2_afolu_2050,
    "all_2050": harmoniser_all_2050,
}

# %%
diff = set(OUTPUT_PATHS.keys()).difference(set(harmonisation_choices.keys()))

# %%
# workflow_res_up_to_infilling_default = run_workflow_up_to_infilling(
#     # Use all scenarios here to ensure that we use an infilling database with everything
#     scenarios_raw_global,
#     n_processes=n_processes,
#     data_root=DATA_ROOT,
#     run_checks=False,  # TODO: implement
# )

# %%
workflow_up_to_infilling_res_d = {}
for key, harmoniser in harmonisation_choices.items():
    workflow_up_to_infilling_res_d[key] = run_workflow_up_to_infilling(
        # Use all scenarios here to ensure that we use an infilling database with everything
        scenarios_raw_global,
        harmoniser=harmoniser,
        n_processes=n_processes,
        data_root=DATA_ROOT,
        run_checks=False,  # TODO: implement
    )

workflow_up_to_infilling_res_d.keys()

# %% [markdown]
# ### Down-select scenarios
#
# Only pick one VLLO and one VLHO scenario per model.

# %%
vllo = "*Very Low*"
vlho = "*Low Overshoot"
overrides = {
    "GCAM 7.1 scenarioMIP": {
        vllo: "SSP1 - Very Low Emissions",
    }
}

# %%
mod_scens_l = []
for model, mdf in scenarios_raw_global.groupby("model"):
    overrides_model = overrides.get(model, None)
    for scenario_group in [vllo, vlho]:
        if overrides_model is not None and scenario_group in overrides_model:
            tmp = (model, overrides_model[scenario_group])

        else:
            mdf_sg = mdf.loc[pix.ismatch(scenario=scenario_group)]
            if mdf_sg.empty:
                print(f"No scenarios in {scenario_group=} for {model=}")
                continue
            else:
                tmp = (model, mdf_sg.pix.unique("scenario")[0])

        mod_scens_l.append(tmp)
        # break

mod_scens = pd.MultiIndex.from_tuples(mod_scens_l, names=("model", "scenario"))
# mod_scens = mod_scens[2:4]
mod_scens.to_frame(index=False)

# %% [markdown]
# ### Quick plots

# %%
tmp = pix.concat(
    [
        *[
            v
            for key, wr in workflow_up_to_infilling_res_d.items()
            for v in [
                wr.pre_processed_emissions.pix.assign(stage="pre-processed", workflow=key),
                wr.harmonised_emissions.pix.assign(stage="harmonised", workflow=key),
                wr.infilled_emissions.pix.assign(stage="infilled", workflow=key),
            ]
        ]
    ]
)
tmp = multi_index_lookup(tmp, mod_scens)
sns_df = tmp.melt(ignore_index=False, var_name="year").reset_index()
sns_df["ms"] = sns_df["model"] + sns_df["scenario"]
# Can plot with this if needed
sns_df

# %%
scm_results_d = {}
for key, res in workflow_up_to_infilling_res_d.items():
    scenarios_run = multi_index_lookup(res.complete_scenarios, mod_scens)
    run_res = run_magicc_and_post_processing(
        scenarios_run,
        n_processes=n_processes,
        magicc_exe_path=DATA_ROOT.parents[0] / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64",
        magicc_prob_distribution_path=(
            DATA_ROOT.parents[0]
            / "magicc"
            / "magicc-v7.6.0a3"
            / "configs"
            / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
        ),
        output_path=OUTPUT_PATHS[key],
        data_root=DATA_ROOT,
    )

    scm_results_d[key] = run_res

scm_results_d.keys()

# %% [markdown]
# ## Save

# %%
for key, out_path in OUTPUT_PATHS.items():
    for full_path, df in (
        (out_path / "pre-processed.csv", workflow_up_to_infilling_res_d[key].pre_processed_emissions),
        (out_path / "harmonised.csv", workflow_up_to_infilling_res_d[key].harmonised_emissions),
        (out_path / "infilled.csv", workflow_up_to_infilling_res_d[key].infilled_emissions),
        # (out_path / "complete_scenarios.csv", complete_scenarios),
        (out_path / "metadata.csv", scm_results_d[key].post_processed.metadata),
    ):
        print(f"Writing {full_path}")
        full_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(full_path)
        print(f"Wrote {full_path}")
        print()
    print()

# %%
