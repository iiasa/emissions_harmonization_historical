"""
SCM-running configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import openscm_runner.adapters
import pandas as pd
from attrs import define, field
from gcages.database import GCDB
from gcages.scm_running import (
    convert_openscm_runner_output_names_to_magicc_output_names,
    run_scm,
    transform_iamc_to_openscm_runner_variable,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


@define
class AR7FTSCMRunner:
    """
    AR7 fast-track simple climate model (SCM) runner
    """

    climate_models_cfgs: dict[str, list[dict[str, Any]]] = field(
        repr=lambda x: ", ".join((f"{climate_model}: {len(cfgs)} configurations" for climate_model, cfgs in x.items()))
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
            variable=to_run.index.get_level_values("variable").map(transform_iamc_to_openscm_runner_variable).values
        )

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

    @classmethod
    def from_default_config(
        cls,
        magicc_exe_path: Path,
        magicc_prob_distribution_path: Path,
        output_path: Path,
        n_processes: int = multiprocessing.cpu_count(),
    ) -> AR7FTSCMRunner:
        """
        Initialise from default, hard-coded configuration
        """
        os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
        if openscm_runner.adapters.MAGICC7.get_version() != "v7.6.0a3":
            raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())

        # Check MAGICC prob. distribution
        with open(magicc_prob_distribution_path) as fh:
            hash = hashlib.sha256(fh.read().encode("utf-8")).hexdigest()
        if hash != "90ff13c82f552bba51051bb88dead7102ec94fee95408d416413bf352816b7dc":
            # Calculated the hash with
            # `openssl dgst -sha256 magicc/magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json`
            raise AssertionError

        with open(magicc_prob_distribution_path) as fh:
            cfgs_raw = json.load(fh)

        scm_output_variables = (
            # GSAT
            "Surface Air Temperature Change",
            # # GMST
            # "Surface Air Ocean Blended Temperature Change",
            # ERFs
            "Effective Radiative Forcing",
            "Effective Radiative Forcing|Anthropogenic",
            "Effective Radiative Forcing|Aerosols",
            "Effective Radiative Forcing|Aerosols|Direct Effect",
            "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
            "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
            "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
            "Effective Radiative Forcing|Aerosols|Indirect Effect",
            "Effective Radiative Forcing|Greenhouse Gases",
            "Effective Radiative Forcing|CO2",
            "Effective Radiative Forcing|CH4",
            "Effective Radiative Forcing|N2O",
            "Effective Radiative Forcing|F-Gases",
            "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
            "Effective Radiative Forcing|Ozone",
            "Effective Radiative Forcing|Aviation|Cirrus",
            "Effective Radiative Forcing|Aviation|Contrail",
            "Effective Radiative Forcing|Aviation|H2O",
            "Effective Radiative Forcing|Black Carbon on Snow",
            # 'Effective Radiative Forcing|CH4 Oxidation Stratospheric',
            "CH4OXSTRATH2O_ERF",
            "Effective Radiative Forcing|Land-use Change",
            # "Effective Radiative Forcing|CFC11",
            # "Effective Radiative Forcing|CFC12",
            # "Effective Radiative Forcing|HCFC22",
            # "Effective Radiative Forcing|HFC125",
            # "Effective Radiative Forcing|HFC134a",
            # "Effective Radiative Forcing|HFC143a",
            # "Effective Radiative Forcing|HFC227ea",
            # "Effective Radiative Forcing|HFC23",
            # "Effective Radiative Forcing|HFC245fa",
            # "Effective Radiative Forcing|HFC32",
            # "Effective Radiative Forcing|HFC4310mee",
            # "Effective Radiative Forcing|CF4",
            # "Effective Radiative Forcing|C6F14",
            # "Effective Radiative Forcing|C2F6",
            # "Effective Radiative Forcing|SF6",
            # # Heat uptake
            # "Heat Uptake",
            # "Heat Uptake|Ocean",
            # Atmospheric concentrations
            "Atmospheric Concentrations|CO2",
            "Atmospheric Concentrations|CH4",
            "Atmospheric Concentrations|N2O",
            # # Carbon cycle
            # "Net Atmosphere to Land Flux|CO2",
            # "Net Atmosphere to Ocean Flux|CO2",
            # # permafrost
            # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
            # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
        )

        startyear = 1750
        endyear = 2100
        out_dynamic_vars = [
            f"DAT_{v}" for v in convert_openscm_runner_output_names_to_magicc_output_names(scm_output_variables)
        ]
        scm_results_db = GCDB(output_path / "db")

        base_cfgs = [
            {
                "run_id": c["paraset_id"],
                **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
            }
            for c in cfgs_raw["configurations"]
        ]

        common_cfg = {
            "startyear": startyear,
            "endyear": endyear,
            "out_dynamic_vars": out_dynamic_vars,
            "out_ascii_binary": "BINARY",
            "out_binary_format": 2,
        }

        run_config = [{**common_cfg, **base_cfg} for base_cfg in base_cfgs]

        res = cls(
            climate_models_cfgs={"MAGICC7": run_config},
            output_variables=scm_output_variables,
            force_interpolate_to_yearly=True,
            db=scm_results_db,
            res_column_type=int,
            # TODO: implement and activate
            run_checks=False,
            n_processes=n_processes,
        )

        return res
