"""
Our processing workflow

Will be split out into climate-assessment or similar in future
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
from attrs import define
from gcages.post_processing import PostProcessingResult

from emissions_harmonization_historical.harmonisation import AR7FTHarmoniser, load_default_history
from emissions_harmonization_historical.infilling import AR7FTInfiller
from emissions_harmonization_historical.post_processing import AR7FTPostProcessor
from emissions_harmonization_historical.pre_processing import AR7FTPreProcessor
from emissions_harmonization_historical.scm_running import AR7FTSCMRunner


@define
class AR7FTWorkflowUpToInfillingRunResult:
    """Results of running the AR7 fast-track workflow up to the end of infilling"""

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
    The complete scenarios, i.e. scenarios with the complete set of emissions needed to run the SCMs
    """


@define
class AR7FTMAGICCRunResult:
    """Results of running MAGICC within the AR7 fast-track"""

    scenarios_run: pd.DataFrame
    """The scenarios that were run by MAGICC"""

    magicc_results: pd.DataFrame
    """The raw results from MAGICC"""

    post_processed: PostProcessingResult
    """Post-processing results"""


@define
class AR7FTWorkflowSCMRunResult:
    """Results of running an SCM as part of the AR7 fast-track workflow"""

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


def run_workflow_up_to_infilling(  # noqa: PLR0913
    input_emissions: pd.DataFrame,
    *,
    run_checks: bool = True,
    pre_processor: AR7FTPreProcessor | None = None,
    harmoniser: AR7FTHarmoniser | None = None,
    infiller: AR7FTInfiller | None = None,
    data_root: Path | None = None,
    n_processes: int = multiprocessing.cpu_count(),
) -> AR7FTWorkflowUpToInfillingRunResult:
    """
    Run the workflow up to the end of infilling

    Parameters
    ----------
    input_emissions
        Input emissions

    n_processes
        Number of parallel processes to use for each step in the workflow.

    run_checks
        Whether to run the checks at each stage or not.

    pre_processor
        Pre-processor to use.

        If not supplied, we use the default
        `AR7FTPreProcessor.from_default_config`.

    harmoniser
        Harmoniser to use.

        If not supplied, we use the default
        `AR7FTHarmoniser.from_default_config`.

    infiller
        Infiller to use.

        If not supplied, we use the default
        `AR7FTInfiller.from_default_config`.

    data_root
        Root data path

    n_processes
        Number of parallel processes to use throughout

    Returns
    -------
    :
        Results of running the workflow up to the end of infilling.
    """
    if pre_processor is None:
        pre_processor = AR7FTPreProcessor.from_default_config()

    if harmoniser is None:
        if data_root is None:
            raise TypeError(data_root)

        harmoniser = AR7FTHarmoniser.from_default_config(data_root=data_root, n_processes=n_processes)

    pre_processed = pre_processor(input_emissions)
    harmonised = harmoniser(pre_processed)

    if infiller is None:
        infiller = AR7FTInfiller.from_default_config(harmonised=harmonised, data_root=data_root)

    infilled = infiller(harmonised)

    complete_scenarios = pix.concat([harmonised, infilled])

    res = AR7FTWorkflowUpToInfillingRunResult(
        input_emissions=input_emissions,
        pre_processed_emissions=pre_processed,
        harmonised_emissions=harmonised,
        infilled_emissions=infilled,
        complete_scenarios=complete_scenarios,
    )

    return res


def add_in_data_from_historical(
    scenarios: pd.DataFrame, history: pd.DataFrame, include_historical_data_after: int
) -> pd.DataFrame:
    history_cut = history.loc[:, include_historical_data_after : scenarios.columns.min() - 1]
    if history_cut.isnull().any().any():
        raise AssertionError

    exp_n_variables = 52
    if history_cut.shape[0] != exp_n_variables:
        raise AssertionError

    a, b = history_cut.reset_index(["model", "scenario"], drop=True).align(scenarios)
    scenarios_full = pix.concat(
        [a.dropna(how="all", axis="columns"), b.dropna(how="all", axis="columns")], axis="columns"
    ).sort_index(axis="columns")

    # Interpolate so it's clearer what went into MAGICC
    scenarios_full = scenarios_full.T.interpolate("index").T
    if scenarios_full.isnull().any().any():
        raise AssertionError

    return scenarios_full


def run_magicc_and_post_processing(  # noqa: PLR0913
    complete_scenarios: pd.DataFrame,
    *,
    run_checks: bool = True,
    n_processes: int = multiprocessing.cpu_count(),
    post_processor: AR7FTPostProcessor | None = None,
    scm_runner: AR7FTSCMRunner | None = None,
    magicc_exe_path: Path | None = None,
    magicc_prob_distribution_path: Path | None = None,
    output_path: Path | None = None,
    history: pd.DataFrame | None = None,
    data_root: Path | None = None,
    batch_size_scenarios: int = 10,
    include_historical_data_after: int = 2015,  # End of MAGICC's internal historical emissions
) -> AR7FTMAGICCRunResult:
    if post_processor is None:
        post_processor = AR7FTPostProcessor.from_default_config()

    if scm_runner is None:
        if magicc_exe_path is None:
            raise TypeError

        if magicc_prob_distribution_path is None:
            raise TypeError

        if output_path is None:
            raise TypeError

        scm_runner = AR7FTSCMRunner.from_default_config(
            magicc_exe_path=magicc_exe_path,
            magicc_prob_distribution_path=magicc_prob_distribution_path,
            output_path=output_path,
            n_processes=n_processes,
        )

    if history is None:
        if data_root is None:
            raise TypeError(data_root)

        history = load_default_history(data_root=data_root)

    # MAGICC fun
    scenarios_run = add_in_data_from_historical(
        complete_scenarios,
        history=history,
        include_historical_data_after=include_historical_data_after,
    )

    magicc_results = scm_runner(scenarios_run, batch_size_scenarios=batch_size_scenarios)
    post_processed = post_processor(magicc_results)

    res = AR7FTMAGICCRunResult(
        scenarios_run=scenarios_run,
        magicc_results=magicc_results,
        post_processed=post_processed,
    )

    return res
