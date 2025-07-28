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
# # Harmonisation
#
# Here we harmonise each model's data.

# %% [markdown]
# ## Imports

# %%
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import tqdm.auto
from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.index_manipulation import split_sectors
from gcages.testing import compare_close
from matplotlib.backends.backend_pdf import PdfPages
from pandas_openscm.indexing import multi_index_lookup

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_OUT_DIR,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    PRE_PROCESSED_SCENARIO_DB,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
    harmonise,
    HarmonisationResult
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "WITCH"
make_region_sector_plots: bool = True
output_to_pdf: bool = False

# %% editable=true slideshow={"slide_type": ""}
output_dir_model = HARMONISED_OUT_DIR / model
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
PRE_PROCESSED_SCENARIO_DB.load_metadata().get_level_values("stage").unique()

# %%
model_pre_processed_for_gridding = PRE_PROCESSED_SCENARIO_DB.load(
    pix.ismatch(model=f"*{model}*", stage="gridding_emissions"), progress=True
)
if model_pre_processed_for_gridding.empty:
    raise AssertionError

# model_pre_processed_for_gridding

# %%
model_pre_processed_for_global_workflow = PRE_PROCESSED_SCENARIO_DB.load(
    pix.ismatch(model=f"*{model}*", stage="global_workflow_emissions_raw_names"), progress=True
)
if model_pre_processed_for_global_workflow.empty:
    raise AssertionError

# model_pre_processed_for_global_workflow

# %% [markdown]
# #### Temporary hack: interpolate scenario data to annual to allow harmonisation

# %%
if HARMONISATION_YEAR not in model_pre_processed_for_gridding:
    model_pre_processed_for_gridding[HARMONISATION_YEAR] = np.nan
    model_pre_processed_for_gridding = model_pre_processed_for_gridding.sort_index(axis="columns")
    model_pre_processed_for_gridding = model_pre_processed_for_gridding.T.interpolate(method="index").T

# model_pre_processed_for_gridding.sort_values(by=HARMONISATION_YEAR)

# %%
if HARMONISATION_YEAR not in model_pre_processed_for_global_workflow:
    model_pre_processed_for_global_workflow[HARMONISATION_YEAR] = np.nan
    model_pre_processed_for_global_workflow = model_pre_processed_for_global_workflow.sort_index(axis="columns")
    model_pre_processed_for_global_workflow = model_pre_processed_for_global_workflow.T.interpolate(method="index").T

# model_pre_processed_for_global_workflow.sort_values(by=HARMONISATION_YEAR)

# %% [markdown]
# ### History to use for harmonisation

# %%
history_for_gridding_harmonisation = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="gridding_emissions"))
# history_for_gridding_harmonisation

# %%
history_for_global_workflow_harmonisation = HISTORY_HARMONISATION_DB.load(
    pix.ismatch(purpose="global_workflow_emissions")
)
# history_for_global_workflow_harmonisation

# %%
history_for_harmonisation = pix.concat(
    [history_for_gridding_harmonisation, history_for_global_workflow_harmonisation]
).reset_index("purpose", drop=True)

# aneris explodes if any history year is Nan,
# even ones we don't use
history_for_harmonisation = history_for_harmonisation.dropna(axis="columns")
# make sure the harmonisation year is all there
if HARMONISATION_YEAR not in history_for_harmonisation:
    raise AssertionError

history_for_harmonisation

# %% [markdown]
# ## Harmonise

# %% [markdown]
# ### Overrides

# %%
# Could load in user overrides from elsewhere here.
# They need to be a series with name "method".
user_overrides_gridding = None
user_overrides_global = None

if model.startswith("WITCH"):
    user_overrides_gridding = pd.Series(
        np.nan,
        index=model_pre_processed_for_gridding.index.droplevel(
            model_pre_processed_for_gridding.index.names.difference(["model", "scenario", "region", "variable"])
        ),
        name="method",
    ).astype(str)
    user_overrides_gridding.loc[
        pix.isin(
            variable=[
                "Emissions|BC|Agricultural Waste Burning",
                "Emissions|BC|Forest Burning",
                "Emissions|CO|Agricultural Waste Burning",
                "Emissions|CO|Forest Burning",
                "Emissions|N2O|Agricultural Waste Burning",
                "Emissions|NH3|Agricultural Waste Burning",
                "Emissions|NH3|Forest Burning",
                "Emissions|NOx|Agricultural Waste Burning",
                "Emissions|NOx|Forest Burning",
                "Emissions|OC|Agricultural Waste Burning",
                "Emissions|OC|Forest Burning",
                "Emissions|Sulfur|Agricultural Waste Burning",
                "Emissions|Sulfur|Forest Burning",
                "Emissions|Sulfur|Grassland Burning",
                "Emissions|VOC|Agricultural Waste Burning",
            ]
        )
    ] = "constant_ratio"
    user_overrides_gridding = user_overrides_gridding[user_overrides_gridding != "nan"]
if model.startswith("REMIND"):

    # advised on 5 June 2025 by Elmar
    
    # template
    user_overrides_gridding = pd.Series(
        np.nan,
        index=model_pre_processed_for_gridding.index.droplevel(
            model_pre_processed_for_gridding.index.names.difference(["model", "scenario", "region", "variable"])
        ),
        name="method",
    ).astype(str)


    # index selector: combinations_model_zero_in_harmyear
    model_zero_in_harmyear = model_pre_processed_for_gridding[model_pre_processed_for_gridding[2023] == 0]
    combinations_model_zero_in_harmyear = model_zero_in_harmyear.index.unique()
    # combinations_model_zero_in_harmyear
    combinations_model_zero_in_harmyear_filter = combinations_model_zero_in_harmyear.droplevel(
        [level for level in combinations_model_zero_in_harmyear.names if level not in user_overrides_gridding.index.names]
    ) # only keep indices that are in the template 
    
    
    # set reduce_ratio_2050 for all that do NOT have zero in the harmonization year for model data
    user_overrides_gridding.loc[~user_overrides_gridding.index.isin(combinations_model_zero_in_harmyear_filter)] = "reduce_ratio_2050"
    user_overrides_gridding = user_overrides_gridding[user_overrides_gridding != "nan"]

    ## global (not implemented yet)
    # template
    user_overrides_global = pd.Series(
        np.nan,
        index=model_pre_processed_for_global_workflow.index.droplevel(
            model_pre_processed_for_global_workflow.index.names.difference(["model", "scenario", "region", "variable"])
        ),
        name="method",
    ).astype(str)


    # index selector: combinations_model_zero_in_harmyear
    model_zero_in_harmyear_global = model_pre_processed_for_global_workflow[model_pre_processed_for_global_workflow[2023] == 0]
    combinations_model_zero_in_harmyear_global = model_zero_in_harmyear_global.index.unique()
    combinations_model_zero_in_harmyear_global_filter = combinations_model_zero_in_harmyear_global.droplevel(
        [level for level in combinations_model_zero_in_harmyear_global.names if level not in user_overrides_global.index.names]
    ) # only keep indices that are in the template 
    
    
    # set reduce_ratio_2050 for all that do NOT have zero in the harmonization year for model data
    user_overrides_global.loc[~user_overrides_global.index.isin(combinations_model_zero_in_harmyear_global_filter)] = "reduce_ratio_2050"
    user_overrides_global = user_overrides_global[user_overrides_global != "nan"]

    # additional method tweaks advised by Leon on 25 June 2025 
    user_overrides_gridding.loc[
        pix.isin(
            variable=[
                "Emissions|BC|Energy Sector",
                "Emissions|BC|Industrial Sector",

                "Emissions|CO|Energy Sector",
                "Emissions|CO|Industrial Sector",
                "Emissions|CO|Transportation Sector",

                "Emissions|CO2|Waste",
                "Emissions|N2O|Waste",
                
                "Emissions|NH3|Energy Sector",
                "Emissions|NH3|Industrial Sector",
                
                "Emissions|Sulfur|Energy Sector",
                "Emissions|Sulfur|Industrial Sector",
                "Emissions|Sulfur|Transportation Sector"
            ]
        )
    ] = "constant_ratio"

user_overrides_gridding


# %%
# user_overrides_gridding.reset_index().variable.unique()


# %% [markdown]
# ### Harmonization

# %%
res = {}
for key, idf, user_overrides in (
    ("gridding", model_pre_processed_for_gridding, user_overrides_gridding),
    ("global", model_pre_processed_for_global_workflow, user_overrides_global),
):
    harmonised_key = harmonise(
        scenarios=idf.reset_index("stage", drop=True),
        history=history_for_harmonisation,
        harmonisation_year=HARMONISATION_YEAR,
        user_overrides=user_overrides,
    )
    res[key] = harmonised_key
    if user_overrides is not None:
        # Check overrides were passsed through correctly
        pd.testing.assert_series_equal(user_overrides, multi_index_lookup(res[key].overrides, user_overrides.index))

# %% [markdown]
# ### Post-harmonization fixes

# %%
# NOTE: do not do this for CDR variables!

# set hist_zero to zero
data = res["gridding"].timeseries
methods = res["gridding"].overrides
methods = methods.reindex(data.index) # add 'unit' to the methods index to enable matching with data

# find all places that have 'hist_zero' harmonization method
hist_zero_mask = methods[methods == 'hist_zero'].index
hist_zero_mask

# replace in all data
all_years = [col for col in data.columns if isinstance(col, int)]
data.loc[hist_zero_mask, all_years] = 0.0

res["gridding"] = HarmonisationResult(
    timeseries=data, 
    overrides=methods
)


# %% [markdown]
# ### Exploring bugs

# %%
# res["gridding"].overrides[res["gridding"].overrides == 'hist_zero']
# res["global"].overrides[res["global"].overrides == 'hist_zero']

# %%
# tmp = res["gridding"].overrides.loc[pix.ismatch(variable="**Peat**")]
# # These cause issues as the history is zero but the model is not
# # so the result isn't actually harmonised
# tmp[tmp == "hist_zero"].loc[pix.ismatch(variable="**CO2**") & pix.isin(scenario=tmp.pix.unique("scenario")[0])]

# %% [markdown]
# ## Ensure that harmonisation worked as expected

# %%
def keep_df_where_harmonisation_gridding_failed(df):
     
    # 1. Define tolerance-based uniqueness
    def _tolerant_nunique(values, tol=1e-6):
        seen = []
        for val in sorted(values):
            if not any(np.isclose(val, x, atol=tol) for x in seen):
                seen.append(val)
        return len(seen)
    
    # 2. Compute approx. unique value count per group
    approx_unique = (
        df.groupby(['model', 'region', 'variable'])[HARMONISATION_YEAR]
        .agg(lambda x: _tolerant_nunique(x, tol=1e-6))
        .reset_index(name='approx_unique_harmonisationyear')
    )

    # 3. Filter groups with more than 1 approx. unique value
    nonunique_groups = approx_unique.query("approx_unique_harmonisationyear > 1")[['model', 'region', 'variable']] # should be only one value in harmonization year

    # 4. Join back to full dataframe to get all relevant rows
    filtered_df = df.merge(nonunique_groups, on=['model', 'region', 'variable'])

    return filtered_df

def assert_harmonisation_gridding_success(df, harmonisation_year=HARMONISATION_YEAR):
   
    failed_df = keep_df_where_harmonisation_gridding_failed(df)

    if not failed_df.empty:
        print("❌ Gridding/harmonization failed for the following rows:")
        print(failed_df)
    else:
        print("✅ All harmonization values passed consistency check.")
        assert failed_df.empty


assert_harmonisation_gridding_success(res["gridding"].timeseries.reset_index())

# %% [markdown]
# ## Examine results

# %%
combo_gridding = pix.concat(
    [
        model_pre_processed_for_gridding.pix.assign(stage="pre-processed"),
        res["gridding"].timeseries.pix.assign(stage="harmonised"),
        history_for_harmonisation.openscm.mi_loc(
            res["gridding"].timeseries.index.droplevel(["model", "scenario"])
        ).pix.assign(stage="history"),
    ]
).sort_index(axis="columns")
combo_gridding.columns = combo_gridding.columns.astype(int)

# %% [markdown]
# ### Single variable

# %%
pdf = (
    combo_gridding.loc[
        pix.isin(
            variable="Emissions|OC|Agricultural Waste Burning",
            region=model_pre_processed_for_gridding.pix.unique("region")[-1],
        ),
        1990:2100,
    ]
    .openscm.to_long_data()
    .dropna()
)
# pdf

# %%
fg = sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="scenario",
    hue_order=sorted(pdf["scenario"].unique()),
    style="stage",
    dashes={
        "history": "",
        "harmonised": "",
        "pre-processed": (3, 3),
    },
    col="variable",
    col_wrap=1,
    facet_kws=dict(sharey=False),
    kind="line",
)
for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="tab:gray")

    else:
        ax.set_ylim(ymin=0.0)

# %% [markdown]
# ### Global harmonisation

# %%
combo_global = pix.concat(
    [
        model_pre_processed_for_global_workflow.pix.assign(stage="pre-processed"),
        res["global"].timeseries.pix.assign(stage="harmonised"),
        history_for_harmonisation.openscm.mi_loc(
            res["global"].timeseries.index.droplevel(["model", "scenario"])
        ).pix.assign(stage="history"),
    ]
).sort_index(axis="columns")
combo_global.columns = combo_global.columns.astype(int)

# %%
pdf_global_total = (
    combo_global.loc[
        :,
        1990:2100,
    ]
    .openscm.to_long_data()
    .dropna()
)
pdf_global_total

# %%
if output_to_pdf:
    ctx_manager = PdfPages(output_dir_model / f"harmonisation-results-global_{model}.pdf")

else:
    ctx_manager = nullcontext()

with ctx_manager as output_pdf_file:
    fg = sns.relplot(
        data=pdf_global_total,
        x="time",
        y="value",
        hue="scenario",
        hue_order=sorted(pdf_global_total["scenario"].unique()),
        style="stage",
        dashes={
            "history": "",
            "harmonised": "",
            "pre-processed": (3, 3),
        },
        col="variable",
        col_order=sorted(pdf_global_total["variable"].unique()),
        col_wrap=3,
        facet_kws=dict(sharey=False),
        kind="line",
    )
    for ax in fg.axes.flatten():
        if "CO2" in ax.get_title():
            ax.axhline(0.0, linestyle="--", color="tab:gray")

        else:
            ax.set_ylim(ymin=0.0)

    if output_to_pdf:
        output_pdf_file.savefig(bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Global vs. gridding harmonisation

# %%
history_gridding_aggregate = to_global_workflow_emissions(
    history_for_gridding_harmonisation.loc[pix.isin(region=res["gridding"].timeseries.pix.unique("region"))]
    .reset_index("purpose", drop=True)
    .rename_axis("year", axis="columns")
    .pix.assign(model="CEDS-BB4CMIP"),
    global_workflow_co2_fossil_sector="Energy and Industrial Processes",
    global_workflow_co2_biosphere_sector="AFOLU",
).pix.assign(workflow="gridding", stage="history")
# history_gridding_aggregate

# %%
harmonised_gridding_aggregate = to_global_workflow_emissions(
    res["gridding"].timeseries,
    global_workflow_co2_fossil_sector="Energy and Industrial Processes",
    global_workflow_co2_biosphere_sector="AFOLU",
).pix.assign(
    workflow="gridding",
    stage="harmonised",
)
# harmonised_gridding_aggregate

# %%
gridding_aggregates = pix.concat(
    [
        history_gridding_aggregate,
        harmonised_gridding_aggregate,
    ]
)
combo_global_v_gridding = pix.concat(
    [
        combo_global.loc[~pix.isin(stage="pre-processed")]
        .pix.assign(workflow="global")
        .loc[pix.isin(variable=gridding_aggregates.pix.unique("variable"))],
        gridding_aggregates,
    ]
).sort_index(axis="columns")
# combo_global_v_gridding

# %% [markdown]
# Key difference in historical emissions
# between gridding and global workflows is CO2 AFOLU,
# CH$_4$ pre-1970 and N$_2$O pre-1970,
# which makes sense as they use different data sources.

# %%
tmp = combo_global_v_gridding.loc[pix.isin(stage="history")].dropna(axis="columns")
diffs = (
    compare_close(
        tmp.loc[pix.isin(workflow="global")].reset_index(["workflow", "model"], drop=True),
        tmp.loc[pix.isin(workflow="gridding")].reset_index(["workflow", "model"], drop=True),
        left_name="global",
        right_name="gridding",
        rtol=1e-4,
    )
    .rename_axis("source", axis="columns")
    .unstack()
    .stack("source", future_stack=True)
    .pix.project(["variable", "source"])
)

for variable, vdf in diffs.groupby("variable"):
    ax = vdf.T.plot()
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.show()

# %%
pdf_global_v_gridding = (
    combo_global_v_gridding.loc[
        :,
        1990:2100,
    ]
    .openscm.to_long_data()
    .dropna()
)
# pdf_global_v_gridding

# %% editable=true slideshow={"slide_type": ""}
fg = sns.relplot(
    data=pdf_global_v_gridding,
    x="time",
    y="value",
    hue="scenario",
    hue_order=sorted(pdf_global_v_gridding["scenario"].unique()),
    style="workflow",
    dashes={
        "gridding": "",
        "global": (3, 3),
    },
    col="variable",
    col_order=sorted(pdf_global_v_gridding["variable"].unique()),
    col_wrap=3,
    units="stage",
    estimator=None,
    facet_kws=dict(sharey=False),
    kind="line",
)
for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="tab:gray")

    else:
        ax.set_ylim(ymin=0.0)

# %% [markdown]
# ### Gridding emissions

# %%
pdf_gridding = pix.concat(
    [
        combo_gridding,
        combo_gridding.openscm.groupby_except("region").sum(min_count=1).pix.assign(region="World"),
    ]
).sort_index(axis="columns")
pdf_gridding.columns = pdf_gridding.columns.astype(int)
pdf_gridding = pdf_gridding.loc[:, 1950:]
# pdf_gridding = pdf_gridding.loc[pix.ismatch(variable="**OC**"), :]
# pdf_gridding

# %%
# # If you need to look at negative values only, use this
# variable_regions_to_plot = harmonise_res.timeseries[harmonise_res.timeseries.min(axis="columns") < 0].index.droplevel(
#   harmonise_res.timeseries.index.names.difference(["variable", "region"])
# ).drop_duplicates()
# # variable_regions_to_plot
# pdf_gridding = pdf_gridding.openscm.mi_loc(variable_regions_to_plot)
# pdf_gridding.pix.unique("variable")

# %%
if pdf_gridding.empty:
    raise AssertionError

# pdf_gridding

# %%
pdf_sectors = split_sectors(pdf_gridding)
# pdf_sectors

# %%
regions = ["World", *sorted([r for r in pdf_sectors.index.get_level_values("region").unique() if r != "World"])]
regions

# %%
species_l = sorted(pdf_sectors.pix.unique("species"))

if make_region_sector_plots:
    if output_to_pdf:
        ctx_manager = PdfPages(output_dir_model / f"harmonisation-results_{model}.pdf")

        pn = 1
        toc_l = ["Table of contents", "=================", ""]

        for region in regions:
            toc_l.append(f"{region}")
            toc_l.append("-" * len(region))
            for species in species_l:
                pad = 10 - len(species)
                toc_l.append(f"    {species}:{' ' * pad}{pn}")
                pn += 1

            toc_l.append("")

        toc = "\n".join(toc_l)
        # toc

    else:
        ctx_manager = nullcontext()

    with ctx_manager as output_pdf_file:
        for region in tqdm.auto.tqdm(regions, desc="regions"):
            pdf_r = pdf_sectors.loc[pix.isin(region=region)]
            for species in tqdm.auto.tqdm(species_l, desc="species", leave=False):
                sdf = pdf_r.loc[pix.isin(species=species)]
                snsdf = sdf.openscm.to_long_data().dropna()
                fg = sns.relplot(
                    data=snsdf,
                    x="time",
                    y="value",
                    hue="scenario",
                    hue_order=sorted(snsdf["scenario"].unique()),
                    style="stage",
                    dashes={
                        "history": "",
                        "harmonised": "",
                        "pre-processed": (3, 3),
                    },
                    col="sectors",
                    col_wrap=min(3, len(snsdf["sectors"].unique())),
                    kind="line",
                    facet_kws=dict(sharey=False),
                )
                fg.fig.suptitle(f"{species} - {region}", y=1.02)
                for ax in fg.axes.flatten():
                    ax.axvline(HARMONISATION_YEAR, linestyle="--", color="gray", alpha=0.3, zorder=1.2)

                    if species == "CO2":
                        ax.axhline(0.0, linestyle="--", color="tab:gray")
                    else:
                        ax.set_ylim(ymin=0.0)

                if output_to_pdf:
                    output_pdf_file.savefig(bbox_inches="tight")
                    plt.close()
                else:
                    plt.show()

            # # Don't plot all for now
            # if region != "World":
            #     break
            # break

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Create combination to use for simple climate models
#
# Use the aggregate of the gridding emissions where we can,
# except for CO<sub>2</sub> AFOLU,
# globally harmonised timeseries otherwise.

# %%
from_gridding = harmonised_gridding_aggregate.reset_index(["workflow", "stage"], drop=True)
from_gridding = from_gridding.loc[~pix.isin(variable="Emissions|CO2|AFOLU")]
# from_gridding

# %%
from_global = res["global"].timeseries.loc[~pix.isin(variable=from_gridding.pix.unique("variable"))]
if from_global.empty:
    for_scms = from_gridding
else:
    for_scms = pix.concat([from_gridding, from_global])

for_scms

# %% [markdown]
# ## Save

# %%
if output_to_pdf:
    with open(output_dir_model / f"harmonisation-results_{model}_table-of-contents.txt", "w") as fh:
        fh.write(toc)

# %%
for idr, res_h in res.items():
    res_h.overrides.to_csv(output_dir_model / f"harmonisation-methods_{idr}_{model}.csv")
    HARMONISED_SCENARIO_DB.save(res_h.timeseries.pix.assign(workflow=idr), allow_overwrite=True)

# %%
HARMONISED_SCENARIO_DB.save(for_scms.pix.assign(workflow="for_scms"), allow_overwrite=True)
