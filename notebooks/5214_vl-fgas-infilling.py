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

# %%
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

from emissions_harmonization_historical.constants_5000 import (
    DATA_ROOT,
    DOWNLOAD_SCENARIOS_ID,
    HARMONISATION_ID,
    HISTORY_FOR_HARMONISATION_ID,
    MARKERS_BY_SCENARIOMIP_NAME,
    POST_PROCESSING_ID,
    PRE_PROCESSING_ID,
    # INFILLING_ID,
    SCM_RUNNING_ID,
)

# %%
pandas_openscm.register_pandas_accessor()

# %%
vl_loc = pix.isin(model=MARKERS_BY_SCENARIOMIP_NAME["vl"]["model"]) & pix.isin(
    scenario=MARKERS_BY_SCENARIOMIP_NAME["vl"]["scenario"]
)

ln_loc = pix.isin(model=MARKERS_BY_SCENARIOMIP_NAME["ln"]["model"]) & pix.isin(
    scenario=MARKERS_BY_SCENARIOMIP_NAME["ln"]["scenario"]
)

# %%
emissions_l = []
scm_output_l = []
metadata_l = []
for infilling_id, label, locator in (
    ("202511040855", "velders-kigali-low", vl_loc),
    ("202511040855-vl-standard-infilling", "use-closest-co2-fossil-infilling", vl_loc),
    ("202511040855-vl-5th-infilling", "5th-percentile-infilling", vl_loc),
    ("202511040855-vl-50th-infilling", "50th-percentile-infilling", vl_loc),
    ("202511040855", "ln", ln_loc),
):
    infilled_out_dir = (
        DATA_ROOT
        / "processed"
        / "infilled"
        / "_".join(
            [
                DOWNLOAD_SCENARIOS_ID,
                PRE_PROCESSING_ID,
                HISTORY_FOR_HARMONISATION_ID,
                HARMONISATION_ID,
                infilling_id,
            ]
        )
    )

    infilled_scenarios_db = OpenSCMDB(
        db_dir=infilled_out_dir / "db",
        backend_data=FeatherDataBackend(),
        backend_index=FeatherIndexBackend(),
    )

    scm_out_dir = (
        DATA_ROOT
        / "processed"
        / "scm-output"
        / "_".join(
            [
                DOWNLOAD_SCENARIOS_ID,
                PRE_PROCESSING_ID,
                HISTORY_FOR_HARMONISATION_ID,
                HARMONISATION_ID,
                infilling_id,
                SCM_RUNNING_ID,
            ]
        )
    )

    scm_output_db = OpenSCMDB(
        db_dir=scm_out_dir / "db",
        backend_data=FeatherDataBackend(),
        backend_index=FeatherIndexBackend(),
    )

    post_processing_dir = (
        DATA_ROOT
        / "processed"
        / "post-processed"
        / "_".join(
            [
                DOWNLOAD_SCENARIOS_ID,
                PRE_PROCESSING_ID,
                HISTORY_FOR_HARMONISATION_ID,
                HARMONISATION_ID,
                infilling_id,
                SCM_RUNNING_ID,
                POST_PROCESSING_ID,
            ]
        )
    )

    post_processed_metadata_quantile_db = OpenSCMDB(
        db_dir=post_processing_dir / "db-metadata-quantile",
        backend_data=FeatherDataBackend(),
        backend_index=FeatherIndexBackend(),
    )

    emissions_tmp = infilled_scenarios_db.load(
        locator & pix.isin(stage="complete") & pix.ismatch(variable="**HFC**") & ~pix.ismatch(variable="**HFC23")
    ).pix.assign(label=label)
    emissions_l.append(emissions_tmp)

    try:
        scm_output_tmp = scm_output_db.load(
            locator & pix.ismatch(variable=["Surface Air Temperature Change", "Effective Radiative Forcing|F-Gases"])
        ).pix.assign(label=label)
        scm_output_l.append(scm_output_tmp)

        metadata_tmp = post_processed_metadata_quantile_db.load(locator).pix.assign(label=label)
        metadata_l.append(metadata_tmp)
    except ValueError:
        print(f"Missing SCM runs for {infilling_id}")

emissions = pix.concat(emissions_l)
scm_output = pix.concat(scm_output_l)
metadata = pix.concat(metadata_l)

# %%
metadata.loc[pix.isin(metric=["max", "2100"]) & pix.isin(quantile=0.5)].unstack("metric").sort_values(
    ("value", "max"), ascending=True
)

# %%
emissions_pdf = emissions.openscm.to_long_data()
emissions_pdf["ln"] = emissions_pdf["label"] == "ln"

fg = sns.relplot(
    data=emissions_pdf,
    x="time",
    y="value",
    col="variable",
    col_wrap=4,
    hue="label",
    size="ln",
    facet_kws=dict(sharey=False),
    alpha=0.7,
    height=2.0,
    aspect=2.0,
)

for ax in fg.axes.flatten():
    ax.set_ylim(ymin=0)

# %%
scm_output_pdf = scm_output.loc[:, 2015:].openscm.groupby_except("run_id").median().openscm.to_long_data()
scm_output_pdf["ln"] = scm_output_pdf["label"] == "ln"

fg = sns.relplot(
    data=scm_output_pdf,
    x="time",
    y="value",
    col="variable",
    # col_wrap=3,
    hue="label",
    size="ln",
    facet_kws=dict(sharey=False),
    alpha=0.7,
)

# for ax in fg.axes.flatten():
#     ax.set_ylim(ymin=0)

# %%
