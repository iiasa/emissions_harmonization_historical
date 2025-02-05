"""
Infilling configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pyam
import silicone.database_crunchers
import tqdm.auto
from attrs import define
from gcages.infilling import infill_scenario
from gcages.io import load_timeseries_csv
from gcages.parallelisation import run_parallel


def wrap_silicone_infiller(
    infiller: Callable[[pyam.IamDataFrame], pyam.IamDataFrame],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Wrap a silicone infiller so it can take a plain `pandas.DataFrame` as input"""

    def res(inp: pd.DataFrame) -> pd.DataFrame:
        res_h = infiller(pyam.IamDataFrame(inp)).timeseries()
        # The fact that this is needed suggests there's a bug in silicone
        res_h = res_h.loc[:, inp.dropna(axis="columns", how="all").columns]

        return res_h

    return res


def get_direct_copy_infiller(variable: str, copy_from: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Get an infiller which just copies the scenario from another scenario"""

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        model = inp.pix.unique("model")
        if len(model) != 1:
            raise AssertionError(model)
        model = model[0]

        scenario = inp.pix.unique("scenario")
        if len(scenario) != 1:
            raise AssertionError(scenario)
        scenario = scenario[0]

        res = copy_from.loc[pix.isin(variable=variable)].pix.assign(model=model, scenario=scenario)

        return res

    return infiller


def get_direct_scaling_infiller(  # noqa: PLR0913
    leader: str,
    follower: str,
    scaling_factor: float,
    l_0: float,
    f_0: float,
    f_unit: str,
    calculation_year: int,
    f_calculation_year: float,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller which just scales one set of emissions to create the next set

    This is basically silicone's constant ratio infiller
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        lead_df = inp.loc[pix.isin(variable=[leader])]

        follow_df = (scaling_factor * (lead_df - l_0) + f_0).pix.assign(variable=follower, unit=f_unit)
        if not np.isclose(follow_df[calculation_year], f_calculation_year).all():
            raise AssertionError

        return follow_df

    return infiller


@define
class AR7FTInfiller:
    """
    AR7 fast-track infiller
    """

    infillers_silicone: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
    """
    Functions to use for infilling variable's with silicone.

    The keys define the variable that can be infilled.
    The variables define the function which,
    given inputs with the expected lead variables,
    returns the infilled time series.
    """

    infillers_wmo: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
    """
    Functions to use for infilling variable's with WMO data.
    """

    infillers_scaling: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
    """
    Functions to use for infilling variable's by scaling following another pathway.
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    n_processes: int = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to 1 to process in serial.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Infill

        Parameters
        ----------
        in_emissions
            Emissions to infill

        Returns
        -------
        :
            Infilled emissions
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        # The most efficient parallelisation is not always obvious.
        # We have the 'most obvious' solution below.
        # If we needed, we could open things up to better injection
        # to better support other patterns.
        infilled_sillicone = pix.concat(
            [
                v
                for v in run_parallel(
                    func_to_call=infill_scenario,
                    iterable_input=(gdf for _, gdf in in_emissions.groupby(["model", "scenario"])),
                    input_desc="model-scenario combinations to infill with silicone",
                    n_processes=self.n_processes,
                    infillers=self.infillers_silicone,
                )
                if v is not None
            ]
        )

        overlap = set(self.infillers_wmo.keys()).intersection(in_emissions.pix.unique("variable"))
        if overlap:
            msg = f"Expect to have no overlap between WMO variables and IAM variables. {overlap=}"
            raise AssertionError(msg)

        infilled_wmo = pix.concat(
            [
                v
                for v in run_parallel(
                    func_to_call=infill_scenario,
                    iterable_input=(gdf for _, gdf in in_emissions.groupby(["model", "scenario"])),
                    input_desc="model-scenario combinations to infill with WMO scenarios",
                    n_processes=self.n_processes,
                    infillers=self.infillers_wmo,
                )
                if v is not None
            ]
        )

        full_set_except_scaling = pix.concat([in_emissions, infilled_sillicone, infilled_wmo])

        infilled_scaling = pix.concat(
            [
                v
                for v in run_parallel(
                    func_to_call=infill_scenario,
                    iterable_input=(gdf for _, gdf in full_set_except_scaling.groupby(["model", "scenario"])),
                    input_desc="model-scenario combinations to infill with direct scaling",
                    n_processes=self.n_processes,
                    infillers=self.infillers_scaling,
                )
                if v is not None
            ]
        )

        infilled = pix.concat([infilled_sillicone, infilled_wmo, infilled_scaling])

        # TODO:
        #   - enable optional checks for:
        #       - input and output metadata is identical
        #         (except maybe a stage indicator)
        #           - no mangled variable names
        #           - no mangled units
        #           - output timesteps are from harmonisation year onwards,
        #             otherwise identical to input
        #       - all scenarios are now complete

        return infilled

    @classmethod
    def from_default_config(cls, harmonised: pd.DataFrame, data_root: Path) -> AR7FTInfiller:
        """
        Initialise from default, hard-coded configuration
        """
        from emissions_harmonization_historical.constants import FOLLOWER_SCALING_FACTORS_ID, WMO_2022_PROCESSING_ID

        # TODO: think some of these through a bit more
        lead_vars_crunchers = {
            "Emissions|BC": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|C2F6": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
            "Emissions|C6F14": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CF4": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
            "Emissions|CO": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|HFC|HFC125": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC134a": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC143a": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC227ea": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC23": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC245fa": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC32": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC43-10": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|NH3": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|NOx": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|OC": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|SF6": ("Emissions|CO2|Energy and Industrial Processes", silicone.database_crunchers.RMSClosest),
            "Emissions|Sulfur": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|VOC": (
                "Emissions|CO2|Energy and Industrial Processes",
                silicone.database_crunchers.QuantileRollingWindows,
            ),
        }

        all_iam_variables = harmonised.pix.unique("variable")
        variables_to_infill_l = []
        for (model, scenario), msdf in harmonised.groupby(["model", "scenario"]):
            to_infill = all_iam_variables.difference(msdf.pix.unique("variable"))
            variables_to_infill_l.extend(to_infill.tolist())

        variables_to_infill = set(variables_to_infill_l)

        infillers_silicone = {}
        for v_infill in tqdm.auto.tqdm(variables_to_infill, desc="Creating sillicone-based infillers"):
            leader, cruncher = lead_vars_crunchers[v_infill]

            # In this set up, infilling and harmonisation are coupled:
            # if we update the harmonisation, we also update the infilling DB
            # and therefore the infilling.
            # This is a choice, rather than being the only solution/option.
            v_infill_db = harmonised.loc[pix.isin(variable=[v_infill, leader])]
            infillers_silicone[v_infill] = wrap_silicone_infiller(
                cruncher(pyam.IamDataFrame(v_infill_db)).derive_relationship(
                    variable_follower=v_infill,
                    variable_leaders=[leader],
                )
            )

        wmo_2022_scenarios = load_timeseries_csv(
            data_root / "global" / "wmo-2022" / "processed" / f"wmo-2022_cmip7_global_{WMO_2022_PROCESSING_ID}.csv",
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_column_type=int,
        ).loc[pix.isin(scenario="WMO 2022 projections v20250129"), harmonised.columns]

        infillers_wmo = {}
        for wmo_var in wmo_2022_scenarios.pix.unique("variable"):
            infillers_wmo[wmo_var] = get_direct_copy_infiller(
                variable=wmo_var,
                copy_from=wmo_2022_scenarios,
            )

        scaling_factors_file = (
            data_root / "global-composite" / f"follower-scaling-factors_{FOLLOWER_SCALING_FACTORS_ID}.json"
        )
        with open(scaling_factors_file) as fh:
            scaling_factors_raw = json.load(fh)

        scale_variables = set(scaling_factors_raw.keys()) - set(infillers_silicone.keys()) - set(infillers_wmo.keys())

        infillers_scaling = {}
        for sv in scale_variables:
            cfg = scaling_factors_raw[sv]
            infillers_scaling[sv] = get_direct_scaling_infiller(
                leader=cfg["leader"],
                follower=sv,
                scaling_factor=cfg["scaling_factor"],
                l_0=cfg["l_0"],
                f_0=cfg["f_0"],
                f_unit=cfg["f_unit"],
                calculation_year=cfg["calculation_year"],
                f_calculation_year=cfg["f_calculation_year"],
            )

        return AR7FTInfiller(
            infillers_silicone=infillers_silicone,
            infillers_wmo=infillers_wmo,
            infillers_scaling=infillers_scaling,
            # TODO: see if we can avoid using local functions
            # hence open this back up
            n_processes=1,
            # TODO: implement and turn on
            run_checks=False,
        )
