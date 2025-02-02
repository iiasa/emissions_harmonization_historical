"""
Our processing workflow

Will be split out into climate-assessment or similar in future
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
from attrs import define

from emissions_harmonization_historical.harmonisation import AR7FTHarmoniser
from emissions_harmonization_historical.pre_processing import AR7FTPreProcessor


@define
class AR7FTWorkflowUpToInfillingRunResult:
    """Results of running the AR7 fast-track workflow up to the end of infilling"""

    input_emissions: pd.DataFrame
    """The input emissions"""

    pre_processed_emissions: pd.DataFrame
    """The pre-processed emissions"""

    harmonised_emissions: pd.DataFrame
    """The harmonised emissions"""

    # infilled_emissions: pd.DataFrame
    # """
    # The infilled emissions, i.e. the complete set of emissions needed to run the SCMs
    # """


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
    data_root: Path | None = None,
    # infiller: Infiller | None = None,
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

    # if infiller is None:
    #     infiller = Infiller()

    pre_processed = pre_processor(input_emissions)
    harmonised = harmoniser(pre_processed)
    # infilled = infiller(harmonised)

    res = AR7FTWorkflowUpToInfillingRunResult(
        input_emissions=input_emissions,
        pre_processed_emissions=pre_processed,
        harmonised_emissions=harmonised,
        # infilled_emissions=infilled,
    )

    return res
