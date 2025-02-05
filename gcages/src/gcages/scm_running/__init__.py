"""
Tools for running simple climate models (SCMs)
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import openscm_runner.adapters
import openscm_runner.run
import pandas as pd
import pandas_indexing as pix
import pymagicc.definitions
import scmdata
import tqdm.auto
from attrs import define, field

from gcages.database import GCDB, EmptyDBError
from gcages.pandas_helpers import multi_index_lookup
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


def transform_iamc_to_openscm_runner_variable(v):
    """
    Transform IAMC variable to OpenSCM-Runner variable

    Parameters
    ----------
    v
        Variable name to transform

    Returns
    -------
    :
        OpenSCM-Runner equivalent of `v`
    """
    res = v

    replacements = (
        ("CFC|", ""),
        ("HFC|", ""),
        ("PFC|", ""),
        ("|Montreal Gases", ""),
        (
            "HFC43-10",
            "HFC4310mee",
        ),
        (
            "AFOLU",
            "MAGICC AFOLU",
        ),
        (
            "Energy and Industrial Processes",
            "MAGICC Fossil and Industrial",
        ),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


def convert_openscm_runner_output_names_to_magicc_output_names(
    openscm_runner_names: Iterable[str],
) -> tuple[str, ...]:
    """
    Get output names for the call to MAGICC

    Parameters
    ----------
    openscm_runner_names
        OpenSCM-Runner output names

    Returns
    -------
    :
        MAGICC output names
    """
    res_l = []
    for openscm_runner_variable in openscm_runner_names:
        if openscm_runner_variable == "Surface Air Temperature Change":
            # A fun bug in pymagicc
            res_l.append("SURFACE_TEMP")
        elif openscm_runner_variable == "Effective Radiative Forcing|HFC4310mee":
            # Another fun bug in pymagicc
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                "Effective Radiative Forcing|HFC4310",
                inverse=True,
            )
            res_l.append(magicc_var)
        else:
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                openscm_runner_variable,
                inverse=True,
            )
            res_l.append(magicc_var)

    return tuple(res_l)


def run_scm(  # noqa: PLR0913
    scenarios: pd.DataFrame,
    climate_models_cfgs: dict[str, list[dict[str, Any]]],
    output_variables: tuple[str, ...],
    n_processes: int,
    db: GCDB | None = None,
    batch_size_scenarios: int | None = None,
    force_interpolate_to_yearly: bool = True,
    force_rerun: bool = False,
) -> pd.DataFrame:
    if db is None:
        db_use = GCDB(Path(tempfile.mkdtemp()))
    else:
        db_use = db

    if force_interpolate_to_yearly:
        # Interpolate to ensure no nans.
        for y in range(scenarios.columns.min(), scenarios.columns.max() + 1):
            if y not in scenarios:
                scenarios[y] = np.nan

        scenarios = scenarios.sort_index(axis="columns").T.interpolate("index").T

    mod_scens_to_run = scenarios.pix.unique(["model", "scenario"])
    for climate_model, cfg in tqdm.auto.tqdm(
        climate_models_cfgs.items(), desc="Climate models"
    ):
        if climate_model == "MAGICC7":
            # Avoid MAGICC's last year jump
            magicc_extra_years = 3
            cfg_use = [
                {**c, "endyear": scenarios.columns.max() + magicc_extra_years}
                for c in cfg
            ]
            os.environ["MAGICC_WORKER_NUMBER"] = str(n_processes)

        else:
            cfg_use = cfg

        if batch_size_scenarios is None:
            scenario_batches = [scenarios]
        else:
            scenario_batches = []
            for i in range(0, mod_scens_to_run.size, batch_size_scenarios):
                start = i
                stop = min(i + batch_size_scenarios, mod_scens_to_run.shape[0])

                scenario_batches.append(
                    multi_index_lookup(scenarios, mod_scens_to_run[start:stop])
                )

        for scenario_batch in tqdm.auto.tqdm(scenario_batches, desc="Scenario batch"):
            if force_rerun:
                run_all = True

            else:
                try:
                    existing_metadata = db_use.load_metadata()
                    run_all = False

                except EmptyDBError:
                    run_all = True

            if run_all:
                scenario_batch_to_run = scenario_batch

            else:
                db_mod_scen = existing_metadata.pix.unique(["model", "scenario"])

                already_run = multi_index_lookup(scenario_batch, db_mod_scen)
                scenario_batch_to_run = scenario_batch.loc[
                    scenario_batch.index.difference(already_run.index)
                ]

                already_run_mod_scen = already_run.pix.unique(["model", "scenario"])
                if not already_run_mod_scen.empty:
                    print(
                        "Not re-running already run scenarios:\n"
                        f"{already_run_mod_scen.to_frame(index=False)}"
                    )

                if scenario_batch_to_run.empty:
                    continue

            if climate_model == "MAGICC7":
                # Avoid MAGICC's last year jump
                scenario_batch_to_run = scenario_batch_to_run.copy()
                last_year = scenario_batch_to_run.columns.max()
                scenario_batch_to_run[last_year + magicc_extra_years] = (
                    scenario_batch_to_run[last_year]
                )
                scenario_batch_to_run = (
                    scenario_batch_to_run.sort_index(axis="columns")
                    .T.interpolate("index")
                    .T
                )

            batch_res = openscm_runner.run.run(
                scenarios=scmdata.ScmRun(scenario_batch_to_run, copy_data=True),
                climate_models_cfgs={climate_model: cfg_use},
                output_variables=output_variables,
            ).timeseries(time_axis="year")

            if climate_model == "MAGICC7":
                # Chop off the extra years
                batch_res = batch_res.iloc[:, :-magicc_extra_years]
                # Chop out regional results
                batch_res = batch_res.loc[pix.isin(region=["World"])]

            for _, df in tqdm.auto.tqdm(
                batch_res.groupby(["model", "scenario", "variable", "region"]),
                desc="Saving SCM results for the batch",
                leave=False,
            ):
                db_use.save(df)

    res = db_use.load(mod_scens_to_run, progress=True)

    if db is None:
        db_use.delete()

    return res


@define
class SCMRunner:
    """
    Simple climate model runner
    """

    climate_models_cfgs: dict[str, list[dict[str, Any]]] = field(
        repr=lambda x: ", ".join(
            (
                f"{climate_model}: {len(cfgs)} configurations"
                for climate_model, cfgs in x.items()
            )
        )
    )
    """
    Climate models to run and the configuration to use with them
    """

    output_variables: tuple[str, ...]
    """
    Variables to include in the output
    """

    force_interpolate_to_yearly: bool = True
    """
    Should we interpolate scenarios we run to yearly steps before running the SCMs.
    """

    db: GCDB | None = None
    """
    Database in which to store the output of the runs

    If not supplied, a default temporary database will be used.
    """

    res_column_type: type = int
    """
    Type to cast the result's column type to
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

    def __call__(
        self,
        in_emissions: pd.DataFrame,
        batch_size_scenarios: int | None = None,
        force_rerun: bool = False,
    ) -> pd.DataFrame:
        """
        Run the simple climate model

        Parameters
        ----------
        in_emissions
            Emissions to run

        batch_size_scenarios
            The number of scenarios to run at once.

            If not supplied, all scenarios are run simultaneously.
            For a large number of scenarios, this may lead to you running out of memory.
            Setting `batch_size_scenarios=1` means each scenario will be run separately,
            which is slower but you (almost definitely) won't run out of memory.

        force_rerun
            Force scenarios to re-run (i.e. disable caching).

        Returns
        -------
        :
            Raw results from the simple climate model
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in in_emissions
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        to_run = strip_pint_incompatible_characters_from_units(in_emissions)
        # Transform variable names
        to_run = to_run.pix.assign(
            variable=to_run.index.get_level_values("variable")
            .map(transform_iamc_to_openscm_runner_variable)
            .values
        )

        # # Deal with the HFC245 bug
        # to_run = to_run.pix.assign(
        #     variable=to_run.index.get_level_values("variable")
        #     .str.replace("HFC245ca", "HFC245fa")
        #     .values,
        #     unit=to_run.index.get_level_values("unit")
        #     .str.replace("HFC245ca", "HFC245fa")
        #     .values,
        # )

        scm_results = run_scm(
            to_run,
            climate_models_cfgs=self.climate_models_cfgs,
            output_variables=self.output_variables,
            n_processes=self.n_processes,
            db=self.db,
            batch_size_scenarios=batch_size_scenarios,
            force_interpolate_to_yearly=self.force_interpolate_to_yearly,
            force_rerun=force_rerun,
        )

        out = scm_results
        out.columns = out.columns.astype(self.res_column_type)

        # TODO:
        #   - enable optional checks for:
        #       - input and output scenarios are the same

        return out
