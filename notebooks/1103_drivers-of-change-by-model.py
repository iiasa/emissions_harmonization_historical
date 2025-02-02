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
# # Examine drivers of change by model

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import pyam
from gcages.io import load_timeseries_csv

from emissions_harmonization_historical.constants import DATA_ROOT, SCENARIO_TIME_ID, WORKFLOW_ID

# %% [markdown]
# ## Define some constants

# %%
ar6_workflow_output_dir = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_ar6-workflow"
)
updated_workflow_output_dir = (
    DATA_ROOT / "climate-assessment-workflow" / "output" / f"{WORKFLOW_ID}_{SCENARIO_TIME_ID}_updated-workflow"
)

# %%
ar6_workflow_magicc_v753_output_dir = ar6_workflow_output_dir / "magicc-ar6"
ar6_workflow_magicc_v753_output_dir

# %%
ar6_workflow_magicc_v76_output_dir = ar6_workflow_output_dir / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0"
ar6_workflow_magicc_v76_output_dir

# %%
updated_workflow_magicc_v753_output_dir = updated_workflow_output_dir / "magicc-v7-5-3_600-member"
updated_workflow_magicc_v76_output_dir = (
    updated_workflow_output_dir / "magicc-v7-6-0a3_magicc-ar7-fast-track-drawnset-v0-3-0"
)

# %%
updated_workflow_magicc_v76_output_dir

# %%
magicc_v76_version_label = "magicc-v7.6.0a3"

# %% [markdown]
# ## Helpers

# %%
scenario_group_order = [
    "VLLO",
    "VLHO",
    "L",
    "ML",
    "M",
    "H",
]


def get_scenario_group(scenario: str) -> str:
    """Get the scenario group"""
    scenario_map_mapping = {
        "Very Low Emissions": "VLLO",
        "Low Overshoot": "VLHO",
        "Low Emissions": "L",
        "Medium-Low Emissions": "ML",
        "Medium Emissions": "M",
        "High Emissions": "H",
    }
    key = scenario.split(" - ")[-1].split("_")[0].strip()

    return scenario_map_mapping[key]


# %%
def transform_ar6_workflow_output_to_iamc_variable(v):
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (("PFC|", ""),)
    for old, new in replacements:
        res = res.replace(old, new)

    direct_replacements = (
        ("HFC245ca", "HFC245fa"),
        ("CCl4", "Montreal Gases|CCl4"),
        ("CFC11", "Montreal Gases|CFC|CFC11"),
        ("CFC12", "Montreal Gases|CFC|CFC12"),
        ("CFC113", "Montreal Gases|CFC|CFC113"),
        ("CFC114", "Montreal Gases|CFC|CFC114"),
        ("CFC115", "Montreal Gases|CFC|CFC115"),
        ("CH2Cl2", "Montreal Gases|CH2Cl2"),
        ("CH3Br", "Montreal Gases|CH3Br"),
        ("CH3CCl3", "Montreal Gases|CH3CCl3"),
        ("CH3Cl", "Montreal Gases|CH3Cl"),
        ("CHCl3", "Montreal Gases|CHCl3"),
        ("HCFC141b", "Montreal Gases|HCFC141b"),
        ("HCFC142b", "Montreal Gases|HCFC142b"),
        ("HCFC22", "Montreal Gases|HCFC22"),
        ("Halon1202", "Montreal Gases|Halon1202"),
        ("Halon1211", "Montreal Gases|Halon1211"),
        ("Halon1301", "Montreal Gases|Halon1301"),
        ("Halon2402", "Montreal Gases|Halon2402"),
    )
    for old, new in direct_replacements:
        if res.endswith(old):
            res = res.replace(old, new)

    return res


# %%
def load_labelled_metadata(
    ar6_workflow_magicc_v753_output_dir: Path,
    ar6_workflow_magicc_v76_output_dir: Path,
    updated_workflow_magicc_v753_output_dir: Path,
    updated_workflow_magicc_v76_output_dir: Path,
    magicc_v76_version_label: str,
) -> pd.DataFrame:
    """
    Load metadata with labels
    """
    metadata_l = []
    for label, out_dir in (
        ("ar6-workflow_magiccv7.5.3", ar6_workflow_magicc_v753_output_dir),
        (f"ar6-workflow_{magicc_v76_version_label}", ar6_workflow_magicc_v76_output_dir),
        ("updated-workflow_magiccv7.5.3", updated_workflow_magicc_v753_output_dir),
        (f"updated-workflow_{magicc_v76_version_label}", updated_workflow_magicc_v76_output_dir),
    ):
        fto_load = out_dir / "metadata.csv"
        if not fto_load.exists():
            print(f"Does not exist: {fto_load=}")
            continue

        tmp = pd.read_csv(fto_load).set_index(["model", "scenario"]).pix.assign(workflow=label)
        metadata_l.append(tmp)

    metadata = pix.concat(metadata_l)
    return metadata


# %%
def load_stage(  # noqa: PLR0913
    out_file_name: str,
    ar6_prefix: str,
    ar6_workflow_output_dir: Path | None = None,
    updated_workflow_output_dir: Path | None = None,
    ar6_workflow_magicc_v753_output_dir: Path | None = None,
    ar6_workflow_magicc_v76_output_dir: Path | None = None,
    updated_workflow_magicc_v753_output_dir: Path | None = None,
    updated_workflow_magicc_v76_output_dir: Path | None = None,
    magicc_v76_version_label: str | None = None,
    index_columns: tuple[str, ...] = ("model", "scenario", "region", "variable", "unit"),
    magicc_output: bool = False,
) -> pd.DataFrame:
    """
    Load data for a given stage
    """
    stage_l = []
    if magicc_output:
        to_load = (
            ("ar6-workflow_magiccv7.5.3", ar6_workflow_magicc_v753_output_dir, True, True),
            (f"ar6-workflow_{magicc_v76_version_label}", ar6_workflow_magicc_v76_output_dir, True, True),
            ("updated-workflow_magiccv7.5.3", updated_workflow_magicc_v753_output_dir, False, False),
            (f"updated-workflow_{magicc_v76_version_label}", updated_workflow_magicc_v76_output_dir, False, False),
        )

    else:
        to_load = (
            ("ar6-workflow", ar6_workflow_output_dir, True, True),
            ("updated-workflow", updated_workflow_output_dir, False, False),
        )

    for label, out_dir, strip_out_ar6_prefix, transform_variables in to_load:
        fto_load = out_dir / out_file_name
        if not fto_load.exists():
            print(f"Does not exist: {fto_load=}")
            continue

        tmp = load_timeseries_csv(
            fto_load,
            index_columns=list(index_columns),
            out_column_type=int,
        ).pix.assign(workflow=label)

        if strip_out_ar6_prefix:
            tmp = pix.assignlevel(
                tmp,
                variable=tmp.index.get_level_values("variable").map(lambda x: x.replace(ar6_prefix, "")),
            )

        # Hmmm, need to work out what to do upstream to remove the need for this
        if transform_variables:
            tmp = pix.assignlevel(
                tmp,
                variable=tmp.index.get_level_values("variable").map(transform_ar6_workflow_output_to_iamc_variable),
            )

        stage_l.append(tmp)

    stage = pix.concat(stage_l)

    # Interpolate to make later life easier
    stage = stage.T.interpolate("index").T
    stage = stage.sort_index()
    stage = stage.pix.assign(scenario_group=stage.index.get_level_values("scenario").map(get_scenario_group))

    return stage


# %%
def get_sns_df(indf):
    """
    Get data frame to use with seaborn's plotting
    """
    out = indf.copy()
    out.columns.name = "year"
    out = out.stack().to_frame("value").reset_index()

    return out


# %% [markdown]
# ## Load

# %%
metadata = load_labelled_metadata(
    ar6_workflow_magicc_v753_output_dir=ar6_workflow_magicc_v753_output_dir,
    ar6_workflow_magicc_v76_output_dir=ar6_workflow_magicc_v76_output_dir,
    updated_workflow_magicc_v753_output_dir=updated_workflow_magicc_v753_output_dir,
    updated_workflow_magicc_v76_output_dir=updated_workflow_magicc_v76_output_dir,
    magicc_v76_version_label=magicc_v76_version_label,
)
metadata

# %%
scm_effective = load_stage(
    out_file_name="scm-effective-emissions.csv",
    ar6_prefix="AR6 climate diagnostics|Infilled|",
    magicc_output=True,
    ar6_workflow_output_dir=ar6_workflow_output_dir,
    updated_workflow_output_dir=updated_workflow_output_dir,
    ar6_workflow_magicc_v753_output_dir=ar6_workflow_magicc_v753_output_dir,
    ar6_workflow_magicc_v76_output_dir=ar6_workflow_magicc_v76_output_dir,
    updated_workflow_magicc_v753_output_dir=updated_workflow_magicc_v753_output_dir,
    updated_workflow_magicc_v76_output_dir=updated_workflow_magicc_v76_output_dir,
    magicc_v76_version_label=magicc_v76_version_label,
)
exp_n_variables = 52
if len(sorted(scm_effective.pix.unique("variable"))) != exp_n_variables:
    raise AssertionError

scm_effective

# %%
climate_percentiles = load_stage(
    out_file_name="timeseries-percentiles.csv",
    ar6_prefix="AR6 climate diagnostics|",
    index_columns=("climate_model", "model", "scenario", "region", "variable", "percentile", "unit"),
    magicc_output=True,
    ar6_workflow_output_dir=ar6_workflow_output_dir,
    updated_workflow_output_dir=updated_workflow_output_dir,
    ar6_workflow_magicc_v753_output_dir=ar6_workflow_magicc_v753_output_dir,
    ar6_workflow_magicc_v76_output_dir=ar6_workflow_magicc_v76_output_dir,
    updated_workflow_magicc_v753_output_dir=updated_workflow_magicc_v753_output_dir,
    updated_workflow_magicc_v76_output_dir=updated_workflow_magicc_v76_output_dir,
    magicc_v76_version_label=magicc_v76_version_label,
)
climate_percentiles.pix.unique("workflow")

# %% [markdown]
# ## Plots of drivers of change for each model

# %%
# Total changes
workflow_base = "ar6-workflow_magiccv7.5.3"
workflow_updated = f"updated-workflow_{magicc_v76_version_label}"
# workflow_updated = f"ar6-workflow_{magicc_v76_version_label}"
# workflow_updated = "updated-workflow_magiccv7.5.3"

# %%
percentile = 50.0
plot_years = range(2010, 2100 + 1)
forcing_breakdown_to_plot = [
    # "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Black Carbon on Snow",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|Ozone",
]

# %%
ctdrop = ["climate_model"]
unstacked = (
    climate_percentiles.loc[pix.isin(variable=forcing_breakdown_to_plot, percentile=percentile), plot_years]
    .unstack("workflow")
    .swaplevel(0, 1, axis=1)
    .reset_index("climate_model", drop=True)
)
# I have no idea why it's necessary to drop the nulls within the subtraction
diff = (
    (unstacked[workflow_updated].dropna(how="all") - unstacked[workflow_base].dropna(how="all"))
    .dropna(how="all")
    .pix.assign(workflow=f"{workflow_updated} - {workflow_base}")
    .round(6)
)
# diff

# %%
col_wrap = 3

for model, mdf in diff.groupby("model"):
    scenario_groups = [v for v in scenario_group_order if v in mdf.pix.unique("scenario_group")]

    scenario_l = []
    for sg in scenario_groups:
        scenario_l.extend(sorted(mdf.loc[pix.isin(scenario_group=sg)].pix.unique("scenario")))

    if len(scenario_l) < col_wrap:
        mosaic = [scenario_l]
    else:
        mosaic = []
        for i in range(0, len(scenario_l), col_wrap):
            if i + col_wrap <= len(scenario_l):
                mosaic.append(scenario_l[i : i + col_wrap])
            else:
                filler = col_wrap - len(scenario_l) % col_wrap
                tmp = [*scenario_l[i:], *(["."] * filler)]
                mosaic.append(tmp)

    fig, axes = plt.subplot_mosaic(
        mosaic=mosaic,
        figsize=(16, 4 * len(mosaic)),
    )
    scenarios_end_of_col = [v[-1] for v in mosaic]

    for i, (scenario, sdf) in enumerate(mdf.groupby("scenario")):
        metadata_ms = metadata.loc[pix.isin(model=model, scenario=scenario)]

        ax = axes[scenario]
        pdf = pyam.IamDataFrame(sdf)
        legend = scenario in scenarios_end_of_col
        pdf.plot.stack(
            stack="variable",
            title=None,
            total=True,
            ax=ax,
            legend=legend,
            cmap="tab20",
        )
        ax.grid()

        lines = [f"**{scenario}**"]
        for prefix, col in (
            ("Median peak warming", "Peak warming 50.0"),
            ("Median 2100 warming", "EOC warming 50.0"),
            ("33rd peak warming", "Peak warming 33.0"),
        ):
            lines.append(f"*{prefix}*")
            for workflow in [workflow_base, workflow_updated]:
                val = float(metadata_ms.loc[pix.isin(workflow=workflow), col].values.squeeze())
                lines.append(f"- {workflow}: {val:3.2f}")

        title = "\n".join(lines)
        ax.set_title(title, horizontalalignment="left", loc="left", fontsize="small")

    fig.suptitle(f"{model=}", y=1.02)
    fig.tight_layout()
    plt.show()

# %%
