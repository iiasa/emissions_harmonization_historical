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

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pandas_openscm.io

from emissions_harmonization_historical.constants_5000 import INFILLED_SCENARIOS_DB, INFILLED_SCENARIOS_DB_EXTENSIONS

# %%
pandas_openscm.register_pandas_accessor()

# %%
# Downloaded from https://zenodo.org/records/18497404
zenodo_df = pd.read_excel("ScenarioMIP_emissions_marker_scenarios_v0.1.xlsx", sheet_name="data")
zenodo_df = zenodo_df.set_index(["model", "scenario", "region", "variable", "unit"])
zenodo_df.columns = zenodo_df.columns.astype(int)
zenodo_df.columns.name = "year"
print(zenodo_df.head(2))

# %%
local_result = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete"))
print(local_result.head(2))

local_result_extentsion = INFILLED_SCENARIOS_DB_EXTENSIONS.load(pix.isin(stage="extended"))
local_result_extension = local_result_extentsion.loc[pix.ismatch(variable="**Emissions**")]
for col in local_result_extension.columns:
    if col < local_result.columns[0] or col > local_result.columns[-1]:
        local_result_extension = local_result_extension.drop(columns=col)

print(local_result_extension.head(2))

# %%
zenodo_df_compare = zenodo_df.loc[pix.ismatch(variable="**Emissions**")]
zenodo_df_compare = zenodo_df_compare.openscm.update_index_levels(
    {"variable": lambda x: x.replace("Climate Assessment|Harmonized and Infilled|", "")}
)
zenodo_df_compare = zenodo_df_compare.loc[pix.ismatch(variable="Emissions**")]

# %%
zenodo_df_compare.head(2)

# %%
print("Comparing local results from infilled database with Zenodo data...")

compare_df = pix.concat(
    [
        local_result.reset_index(["scenario", "stage"], drop=True).pix.assign(source="local"),
        zenodo_df_compare.reset_index("scenario", drop=True).pix.assign(source="zenodo"),
    ]
).sort_index(axis=1)

for model, mdf in compare_df.groupby(["model"]):
    print(f"Checking {model}")
    if model[0].startswith("COFFEE"):
        continue
    for variable, ts_df in mdf.groupby("variable"):
        # print(variable)
        if "local" not in ts_df.index.get_level_values("source"):
            continue

        tmp = ts_df.dropna(how="all", axis="columns")
        try:
            pd.testing.assert_frame_equal(
                tmp.loc[pix.isin(source="local")].reset_index(["source", "unit"], drop=True),
                tmp.loc[pix.isin(source="zenodo")].reset_index(["source", "unit"], drop=True),
                rtol=1e-5,
                atol=1e-6,
            )
        except AssertionError as exc:
            print(exc)
            ax = tmp.pix.project(["source", "model", "variable", "unit"]).T.plot()
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
            plt.show()


# %%
# Compare with what is in the extensions database,
# should be the same as the zenodo data for the overlapping variables and years
print("Comparing local results from infilled extensions database with Zenodo data...")
compare_df = pix.concat(
    [
        local_result_extension.reset_index(["scenario", "stage"], drop=True).pix.assign(source="local"),
        zenodo_df_compare.reset_index("scenario", drop=True).pix.assign(source="zenodo"),
    ]
).sort_index(axis=1)

for model, mdf in compare_df.groupby(["model"]):
    print(f"Checking {model}")
    if model[0].startswith("COFFEE"):
        continue
    for variable, ts_df in mdf.groupby("variable"):
        # print(variable)
        if "local" not in ts_df.index.get_level_values("source"):
            continue

        tmp = ts_df.dropna(how="all", axis="columns")
        try:
            pd.testing.assert_frame_equal(
                tmp.loc[pix.isin(source="local")].reset_index(["source", "unit"], drop=True),
                tmp.loc[pix.isin(source="zenodo")].reset_index(["source", "unit"], drop=True),
                rtol=1e-5,
                atol=1e-6,
            )
        except AssertionError as exc:
            print(exc)
            print(f"Model: {model}, Variable: {variable}")
            print(tmp.loc[pix.isin(source="local")])
            print(tmp.loc[pix.isin(source="zenodo")])
            ax = tmp.pix.project(["source", "model", "variable", "unit"]).T.plot()
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
            plt.show()
