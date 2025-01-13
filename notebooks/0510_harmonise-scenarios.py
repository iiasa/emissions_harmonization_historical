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
#
# Underlying scenario database: https://data.ece.iiasa.ac.at/ssp-submission/#/workspaces

# %%
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import seaborn as sns
import tqdm.autonotebook as tqdman

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HISTORICAL_COMPOSITE_PROCESSING_ID,
)

# %%
SCENARIO_TIME_ID = "20250113-182449"

# %%
HISTORICAL_GLOBAL_COMPOSITE_PATH = (
    DATA_ROOT / "global-composite" / f"historical-global-composite_{HISTORICAL_COMPOSITE_PROCESSING_ID}.csv"
)


# %%
def load_csv(fp):
    out = pd.read_csv(fp)
    out.columns = out.columns.str.lower()
    out = out.set_index(["model", "scenario", "variable", "region", "unit"])
    out.columns = out.columns.astype(int)

    return out


# %%
history = load_csv(HISTORICAL_GLOBAL_COMPOSITE_PATH)
history_cut = history.loc[:, 1990:]
history_cut

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
scenarios_raw = pix.concat([load_csv(f) for f in tqdman.tqdm(scenario_files)])
scenarios_raw_global = scenarios_raw.loc[
    pix.ismatch(region="World") & pix.isin(variable=history_cut.pix.unique("variable"))
]
scenarios_raw_global

# %%
# pandas-indexing is so well done
# scenarios_raw_global.pix.extract(variable="Emissions|{species}|{sector}|{subsector}", dropna=False, keep=True).index.to_frame(index=False)

# %%
scenarios_raw_global.pix.unique(["model", "scenario"]).to_frame(index=False)


# %%
def get_sns_df(indf):
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
    data=get_sns_df(history_cut),
    kind="line",
    hue="scenario",
    style="model",
)

# %%
make_all_var_plot(
    data=get_sns_df(scenarios_raw_global),
    kind="scatter",
    hue="scenario",
    style="model",
)

# %%
for model, mdf in scenarios_raw_global.groupby("model"):
    pdf = pd.concat([get_sns_df(mdf), get_sns_df(history_cut)])
    make_all_var_plot(
        data=pdf,
        kind="line",
        hue="scenario",
        style="model",
    )
    plt.show()

# %%
# TODO: move this into gcages?
import multiprocessing

from attrs import define
from gcages.ar6.harmonisation import harmonise_scenario
from gcages.parallelisation import run_parallel
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


@define
class PreProcessor:
    emissions_out: tuple[str, ...]

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        # TODO: add checks:
        # - no rows should be all zero or all nan
        # - data should be available for all required years
        # - no negative values for non-CO2
        res: pd.DataFrame = in_emissions.loc[pix.isin(variable=self.emissions_out)]

        res = strip_pint_incompatible_characters_from_units(res, units_index_level="unit")

        return res


@define
class Harmoniser:
    historical_emissions: pd.DataFrame
    harmonisation_year: int
    calc_scaling_year: int
    aneris_overrides: pd.DataFrame | None
    n_processes: int = multiprocessing.cpu_count()

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        # TODO: add checks back in
        # TODO: swap logic, blow up for scenarios that don't have the
        # harmonisation year and push that logic into the layer above
        # so we can remove calc_sccalc_scaling_year here.
        harmonised_df = pix.concat(
            run_parallel(
                func_to_call=harmonise_scenario,
                iterable_input=(gdf for _, gdf in in_emissions.groupby(["model", "scenario"])),
                input_desc="model-scenario combinations to harmonise",
                n_processes=self.n_processes,
                history=self.historical_emissions,
                year=self.harmonisation_year,
                overrides=self.aneris_overrides,
                calc_scaling_year=self.calc_scaling_year,
            )
        )

        # Not sure why this is happening, anyway
        harmonised_df.columns = harmonised_df.columns.astype(int)

        return harmonised_df


# %%
pre_processor = PreProcessor(emissions_out=tuple(history.pix.unique("variable")))

# %%
# As at 2024-01-13, just the list from AR6.
# We can tweak from here.
aneris_overrides = pd.DataFrame(
    [
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|BC'}, # depending on the decision tree in aneris/method.py
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
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },  # high historical variance, but using offset method to prevent diff from increasing when going negative rapidly (cov=23.2)
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'}, # depending on the decision tree in aneris/method.py
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
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'}, # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'}, # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'}, # depending on the decision tree in aneris/method.py
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|OC",
        },  # high historical variance (cov=18.5)
        {
            "method": "constant_ratio",
            "variable": "Emissions|SF6",
        },  # minor f-gas with low model reporting confidence
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'}, # depending on the decision tree in aneris/method.py
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|VOC",
        },  # high historical variance (cov=12.0)
    ]
)

# %%
import scipy.stats

# %%
harmonisation_year = 2021
# To discuss with Jarmo
calc_scaling_year = 2015
history_values = history_cut.loc[:, calc_scaling_year:harmonisation_year].copy()

# TODO: add use of an average for variables with high variability here
high_variability_variables = ("Emissions|BC", "Emissions|CO", "Emissions|OC")
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
    n_processes=multiprocessing.cpu_count(),
    # n_processes = 1,
)

# %%
# This business of having some scenarios end in 2150 and others in 2100
# is causing some issues. Will fix in next steps.
scenarios_raw_global_cut = scenarios_raw_global.loc[:, :2100]
scenarios_raw_global_cut

# %%
pre_processed = pre_processor(scenarios_raw_global_cut)
harmonised = harmoniser(pre_processed)
harmonised

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
    fg = make_all_var_plot(
        data=get_sns_df(pdf_wide),
        kind="line",
        hue="scenario",
        style="stage",
    )
    fg.fig.suptitle(model, y=1.01)
    plt.show()
