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
import logging
import warnings

import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pandas_openscm.io

from emissions_harmonization_historical.constants_5000 import (
    INFILLED_SCENARIOS_DB,
)

# Suppress expected MAGICC warnings about extending solar forcing to 2500
# The solar RF data is already extended to 2500, but MAGICC's Fortran code
# warns that it's using extrapolated (not observed) data beyond 2100
warnings.filterwarnings("ignore", message=".*Extending solar RF.*")
warnings.filterwarnings("ignore", message=".*magicc logged a WARNING message.*")
warnings.filterwarnings(
    "ignore", message=r"magicc logged a WARNING message\. Check the 'stderr' key.*", category=UserWarning
)
warnings.filterwarnings("ignore", category=UserWarning, module="pymagicc.core")
# Also suppress at the source
logging.getLogger("pymagicc").setLevel(logging.ERROR)

# %%
marco = pandas_openscm.io.load_timeseries_csv(
    "../infilled_CGEM_marco.csv",
    index_columns=["model", "scenario", "variable", "unit", "region"],
    out_columns_type=int,
    out_columns_name="year",
)
marco

# %%
here = INFILLED_SCENARIOS_DB.load(pix.isin(stage="complete") & pix.ismatch(model="*CGEM*")).reset_index(
    "stage", drop=True
)
here.columns = here.columns.astype(int)

# %%
differing_variables = []
for variable in here.pix.unique("variable"):
    marco_v = marco.loc[pix.isin(variable=variable), 2023:].reorder_levels(here.index.names)
    here_v = here.loc[pix.isin(variable=variable), :]

    try:
        pd.testing.assert_frame_equal(
            marco_v,
            here_v,
            check_like=True,
        )
    except AssertionError as exc:
        print(f"difference for {variable=}")
        print(exc)
        print()
        differing_variables.append(variable)

print(f"{sorted(differing_variables)=}")

# %%
