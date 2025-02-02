"""
AR6 workflow
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from attrs import define

from gcages.ar6.harmonisation import AR6Harmoniser
from gcages.ar6.infilling import AR6Infiller
from gcages.ar6.post_processing import AR6PostProcessor
from gcages.ar6.pre_processing import AR6PreProcessor
from gcages.ar6.scm_running import (
    AR6SCMRunner,
    convert_openscm_runner_output_names_to_magicc_output_names,
)
from gcages.database import GCDB


@define
class AR6WorkflowRunResult:
    """Results of an AR6 workflow run"""

    input_emissions: pd.DataFrame
    """The input emissions"""

    pre_processed_emissions: pd.DataFrame
    """The pre-processed emissions"""

    harmonised_emissions: pd.DataFrame
    """The harmonised emissions"""

    infilled_emissions: pd.DataFrame
    """The infilled emissions"""

    complete_scenarios: pd.DataFrame
    """
    The complete scenarios

    I.e. scenarios with the complete set of emissions needed to run the SCMs.
    """

    scm_results_raw: pd.DataFrame
    """
    The raw SCM results
    """

    post_processed_timeseries: pd.DataFrame
    """
    Post-processed timeseries
    """

    post_processed_scenario_metadata: pd.DataFrame
    """
    Post-processed scenario-level metadata
    """


def run_ar6_workflow(  # noqa: PLR0913
    input_emissions: pd.DataFrame,
    *,
    magicc_exe_path: Path,
    magicc_prob_distribution_path: Path,
    batch_size_scenarios: int = 4,
    scm_output_variables: tuple[str, ...] | None = None,
    scm_results_db: GCDB | None = None,
    n_processes: int = 1,
    run_checks: bool = True,
) -> AR6WorkflowRunResult:
    """
    Run the workflow as it was run in AR6

    Parameters
    ----------
    input_emissions
        Input emissions

    magicc_exe_path
        Path to the MAGICC executable

    magicc_prob_distribution_path
        Path to MAGICC's probabilistic distribution from AR6

    batch_size_scenarios
        Batch size to use when running scenarios

    scm_output_variables
        SCM variables to output

    scm_results_db
        Database in which to save SCM raw results.

        If not supplied, the results aren't saved.

    n_processes
        Number of parallel processes to use for each step in the workflow.

    run_checks
        Whether to run the checks at each stage or not.

    Returns
    -------
    :
        Results of the workflow.
    """
    # Deliberately thin to avoid users having too many choices.
    # If users want to open things up, use this code as inspiration
    # then just create their own worklow.
    pre_processor = AR6PreProcessor.from_ar6_like_config(
        run_checks=run_checks, n_processes=n_processes
    )
    harmoniser = AR6Harmoniser.from_ar6_like_config(
        run_checks=run_checks, n_processes=n_processes
    )
    infiller = AR6Infiller.from_ar6_like_config(
        run_checks=run_checks,
        n_processes=n_processes,
    )

    scm_runner = AR6SCMRunner.from_ar6_like_config(
        magicc_exe_path=magicc_exe_path,
        magicc_prob_distribution_path=magicc_prob_distribution_path,
        db=scm_results_db,
        run_checks=run_checks,
        n_processes=n_processes,
    )
    if scm_output_variables is not None:
        scm_runner.output_variables = scm_output_variables
        new_config = [
            {
                **c,
                "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(  # noqa: E501
                    scm_output_variables
                ),
            }
            for c in scm_runner.climate_models_cfgs["MAGICC7"]
        ]
        scm_runner.climate_models_cfgs = {"MAGICC7": new_config}

    post_processor = AR6PostProcessor.from_ar6_like_config(
        run_checks=run_checks, n_processes=n_processes
    )

    pre_processed = pre_processor(input_emissions)
    harmonised = harmoniser(pre_processed)
    infilled = infiller(harmonised)
    scm_results = scm_runner(infilled, batch_size_scenarios=batch_size_scenarios)
    post_processed_timeseries, post_processed_metadata = post_processor(scm_results)

    res = AR6WorkflowRunResult(
        input_emissions=input_emissions,
        pre_processed_emissions=pre_processed,
        harmonised_emissions=harmonised,
        infilled_emissions=infilled,
        scm_results_raw=scm_results,
        post_processed_timeseries=post_processed_timeseries,
        post_processed_scenario_metadata=post_processed_metadata,
    )

    return res
