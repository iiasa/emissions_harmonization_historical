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
# # Check internal consistency
#
# Here we check the internal consistency of the reporting of a given model.

# %% [markdown]
# ## Imports

# %%
import difflib
from collections.abc import Callable
from functools import partial

import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
import pandas as pd
import pandas.io.excel
import pandas_indexing as pix
import pandas_openscm
import tqdm.auto
from gcages.index_manipulation import combine_sectors, combine_species, split_sectors
from gcages.testing import compare_close
from openpyxl.styles.fonts import Font
from pandas_openscm.indexing import multi_index_lookup

from emissions_harmonization_historical.constants_5000 import (
    DATA_ROOT,
    DOWNLOAD_SCENARIOS_ID,
    RAW_SCENARIO_DB,
)
from emissions_harmonization_historical.excel_writing import set_cell

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""}
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "MESSAGE"

# %%
output_dir_model = DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / model
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %%
out_file = output_dir_model / f"internal-consistency-checking_{model}.xlsx"
out_file

# %%
pd.set_option("display.max_colwidth", None)

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
model_raw = RAW_SCENARIO_DB.load(pix.ismatch(model=f"*{model}*"), progress=True)
if model_raw.empty:
    raise AssertionError

# model_raw

# %% [markdown]
# ### Properties

# %%
properties = pd.concat([pd.read_csv(f) for f in output_dir_model.glob("*properties.csv")]).drop(
    "metadata", axis="columns"
)
# properties

# %% [markdown]
# ## Check internal consistency

# %% [markdown]
# Extract the model data, keeping:
#
# - only reported timesteps
# - only data from 2015 onwards (we don't care about data before this)
# - only data up to 2100 (while reporting is still weird for data post 2100)

# %%
model_df = model_raw.loc[:, 2015:2100].dropna(how="all", axis="columns")
if model_df.empty:
    raise AssertionError

model_df.columns.name = "year"
# model_df

# %%
# Interpolate before checking to handle issues with reporting on different timesteps
model_df = model_df.T.interpolate(method="index").T
# model_df

# %% [markdown]
# ### Figure out the model-specific regions

# %%
model_regions = [r for r in model_df.pix.unique("region") if r.startswith(model.split(" ")[0])]
if not model_regions:
    raise AssertionError
# model_regions

# %% [markdown]
# ### Define the internal consistency checking index
#
# If your model reports domestic aviation at the regional level, this will be fine.
# If not, we'll need to add some different code.

# %%
internal_consistency_checking_index = (
    gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.get_internal_consistency_checking_index(model_regions)
)
# internal_consistency_checking_index

# %%
internal_consistency_checking_variables = internal_consistency_checking_index.get_level_values("variable").unique()
# internal_consistency_checking_variables

# %% [markdown]
# ### Check the internal consistency

# %% [markdown]
# #### Set up

# %%
model_df_considered = multi_index_lookup(
    model_df,
    internal_consistency_checking_index,
)
# model_df_considered

# %%
model_df_considered_split_sectors = split_sectors(model_df_considered)

# %%
with pd.ExcelWriter(out_file, engine="openpyxl", mode="w") as writer:
    workbook = writer.book
    sheet_name = "scenario_database_info"
    set_cell(
        "Scenario properties as downloaded from the database",
        row=0,
        col=0,
        ws=workbook.create_sheet(title=sheet_name),
        font=Font(bold=True),
    )
    properties.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=2,
        index=False,
    )

# %% [markdown]
# ### Calculate the totals

# %%
totals_exp = combine_species(
    model_df_considered_split_sectors.openscm.groupby_except(["region", "sectors"]).sum()
).pix.assign(region="World")
# totals_exp

# %% [markdown]
# ### Get the reported totals

# %%
totals_reported = multi_index_lookup(model_df, totals_exp.index)
# totals_reported

# %% [markdown]
# ### Helper function


# %%
def write_errors_sheet(  # noqa: PLR0913
    set_cell_ws: Callable[[str, int, int, Font], None],
    variable: str,
    comparison_variable: pd.DataFrame,
    model_df_considered_split_sectors: pd.DataFrame,
    internal_consistency_checking_variables: pd.Index,
    writer: pd.io.excel.ExcelWriter,
    ignore_vars: set[str] = {"Emissions|CO2|AFOLU [NGHGI]"},
) -> None:
    """
    Write a sheet of errors
    """
    set_cell_ws(
        f"Internal consistency errors for {variable}",
        row=0,
        col=0,
        font=Font(bold=True),
    )

    set_cell_ws(
        "Reported totals compared to the sum of region-sector information",
        row=2,
        col=0,
        font=Font(bold=True),
    )
    set_cell_ws(
        f"Relative tolerance applied while checking for consistency: {tols[variable]['rtol']}",
        row=3,
        col=0,
    )
    set_cell_ws(
        f"Absolute tolerance applied while checking for consistency: {tols[variable]['atol'].to(unit)}",
        row=4,
        col=0,
    )
    set_cell_ws(
        "NaN indicates that there is no inconsistency issue for this data point for the given tolerances",
        row=5,
        col=0,
    )
    current_row = 6

    # Get row by row comparison and list possible causes for difference
    comparison_variable.columns.name = "source"
    row_by_row_comparison = comparison_variable.unstack().stack("source", future_stack=True).sort_index()

    components_included_in_sum = (
        combine_sectors(model_df_considered_split_sectors.loc[pix.isin(species=species)])
        .pix.assign(source="reported")
        .reorder_levels(row_by_row_comparison.index.names)
        .sort_index()
    )

    all_reported_vars = model_df.loc[pix.ismatch(variable=f"{variable}|**")].pix.unique("variable")
    all_included_vars = components_included_in_sum.pix.unique("variable")
    not_in_sum_vars = all_reported_vars.difference(all_included_vars).difference(ignore_vars)

    for v in not_in_sum_vars:
        if v == variable:
            continue

        included_lower_in_hierarchy = any(v in iv for iv in all_included_vars)
        if included_lower_in_hierarchy:
            continue

        included_higher_in_hierarchy = any(iv in v for iv in all_included_vars)
        if included_higher_in_hierarchy:
            continue

        possible_replacements = set(
            difflib.get_close_matches(
                v,
                [v for v in internal_consistency_checking_variables if v.startswith(f"{variable}|")],
                n=3,
                cutoff=0.9,
            )
        )
        if possible_replacements:
            msg = f"{v} is not included in the sum anywhere. " f"Some possible renamings: {possible_replacements}"
        else:
            msg = (
                f"{v} is not included in the sum anywhere. "
                f"We don't have any suggested renamings "
                "(maybe it is not included for a reason, please double check)"
            )

        set_cell_ws(
            msg,
            row=current_row,
            col=0,
        )
        current_row += 1

    current_row += 1
    row_by_row_comparison.to_excel(
        writer,
        sheet_name=variable,
        startrow=current_row,
    )
    current_row = current_row + 1 + row_by_row_comparison.shape[0]

    current_row += 1
    set_cell_ws(
        "The following rows were used to create the sector-region sum",
        row=current_row,
        col=0,
        font=Font(bold=True),
    )
    components_included_in_sum.to_excel(
        writer,
        sheet_name=variable,
        startrow=current_row + 2,
    )


# %% [markdown]
# ### Do the checking

# %%
tols = gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.get_default_internal_conistency_checking_tolerances()
# tols

# %% editable=true slideshow={"slide_type": ""}
for variable in tqdm.auto.tqdm(sorted(totals_exp.pix.unique("variable")), desc="Reported total variables"):
    species = variable.split("|")[1]

    totals_exp_variable = totals_exp.loc[pix.isin(variable=variable)]
    totals_reported_variable = totals_reported.loc[pix.isin(variable=variable)]

    unit_l = totals_exp_variable.index.get_level_values("unit").unique()
    if len(unit_l) > 1:
        raise AssertionError(unit_l)

    unit = unit_l[0]
    tols_use = {k: v.to(unit).m if k == "atol" else v for k, v in tols[variable].items()}

    comparison_variable = compare_close(
        left=totals_exp_variable,
        right=totals_reported_variable.reorder_levels(totals_exp_variable.index.names),
        left_name="derived_from_input",
        right_name="reported_total",
        **tols_use,
    )
    with pd.ExcelWriter(out_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        workbook = writer.book
        sheet_name = variable
        set_cell_ws = partial(set_cell, ws=workbook.create_sheet(title=variable))

        if comparison_variable.empty:
            print(f"    {variable}: internally consistent")
            set_cell_ws(
                f"Data is internally consistent for {variable}",
                row=0,
                col=0,
                font=Font(bold=True),
            )

        else:
            print(f"{variable}: internal consistency errors")
            write_errors_sheet(
                set_cell_ws=set_cell_ws,
                variable=variable,
                comparison_variable=comparison_variable,
                model_df_considered_split_sectors=model_df_considered_split_sectors,
                internal_consistency_checking_variables=internal_consistency_checking_variables,
                writer=writer,
            )
