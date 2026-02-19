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
# TODO: get file from Zenodo here.
# Right now all I have is the raw file from Jarmo,
# which is hopefully correct...

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pandas_openscm.io

from emissions_harmonization_historical.constants_5000 import (
    INFILLED_SCENARIOS_DB,
)

# %%
pandas_openscm.register_pandas_accessor()

# %%
zenodo_df = pd.read_excel("/Users/znicholls/Downloads/climate_assessment_with_README.xlsx", sheet_name="data")
zenodo_df = zenodo_df.set_index(["model", "scenario", "region", "variable", "unit"])
zenodo_df.columns = zenodo_df.columns.astype(int)
zenodo_df.head(2)

# %%
local_result = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete"))
local_result.head(2)

# %%
zenodo_df_compare = zenodo_df.loc[pix.ismatch(variable="**Emissions**")]
zenodo_df_compare = zenodo_df_compare.openscm.update_index_levels(
    {"variable": lambda x: x.replace("Climate Assessment|Harmonized and Infilled|", "")}
)
zenodo_df_compare = zenodo_df_compare.loc[pix.ismatch(variable="Emissions**")]

# %%
zenodo_df_compare.head(2)

# %%
compare_df = pix.concat(
    [
        local_result.reset_index(["scenario", "stage"], drop=True).pix.assign(source="local"),
        zenodo_df_compare.reset_index("scenario", drop=True).pix.assign(source="zenodo"),
    ]
).sort_index(axis=1)

for model, mdf in compare_df.groupby(["model"]):
    print(f"Checking {model}")
    for variable, ts_df in mdf.groupby("variable"):
        # print(variable)
        if "local" not in ts_df.index.get_level_values("source"):
            continue

        tmp = ts_df.dropna(how="all", axis="columns")
        try:
            pd.testing.assert_frame_equal(
                tmp.loc[pix.isin(source="local")].reset_index(["source", "unit"], drop=True),
                tmp.loc[pix.isin(source="zenodo")].reset_index(["source", "unit"], drop=True),
            )
        except AssertionError as exc:
            print(exc)
            ax = tmp.pix.project(["source", "model", "variable", "unit"]).T.plot()
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
            plt.show()
