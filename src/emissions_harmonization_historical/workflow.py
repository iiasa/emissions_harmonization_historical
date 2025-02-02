"""
Our processing workflow

Will be split out into climate-assessment or similar in future
"""

from __future__ import annotations

import pandas as pd
from attrs import define
from gcages.ar6.harmonisation import Harmoniser
from gcages.ar6.infilling import Infiller
from gcages.ar6.pre_processing import PreProcessor


@define
class WorkflowUpToInfillingRunResult:
    """Results of running the workflow up to the end of infilling"""

    input_emissions: pd.DataFrame
    """The input emissions"""

    pre_processed_emissions: pd.DataFrame
    """The pre-processed emissions"""

    # harmonised_emissions: pd.DataFrame
    # """The harmonised emissions"""
    #
    # infilled_emissions: pd.DataFrame
    # """
    # The infilled emissions, i.e. the complete set of emissions needed to run the SCMs
    # """


@define
class WorkflowSCMRunResult:
    """Results of running an SCM as part of the workflow"""

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
    n_processes: int = 1,
    run_checks: bool = True,
    pre_processor: PreProcessor | None = None,
    harmoniser: Harmoniser | None = None,
    infiller: Infiller | None = None,
) -> WorkflowUpToInfillingRunResult:
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

    Returns
    -------
    :
        Results of running the workflow up to the end of infilling.
    """
    if pre_processor is None:
        pre_processor = PreProcessor()

    # if harmoniser is None:
    #     harmoniser = Harmoniser()
    #
    # if infiller is None:
    #     infiller = Infiller()

    pre_processor = PreProcessor.from_ar6_like_config(run_checks=run_checks, n_processes=n_processes)
    # harmoniser = Harmoniser.from_ar6_like_config(run_checks=run_checks, n_processes=n_processes)
    # infiller = Infiller.from_ar6_like_config(
    #     run_checks=run_checks,
    #     n_processes=n_processes,
    # )

    pre_processed = pre_processor(input_emissions)
    # harmonised = harmoniser(pre_processed)
    # infilled = infiller(harmonised)

    res = WorkflowUpToInfillingRunResult(
        input_emissions=input_emissions,
        pre_processed_emissions=pre_processed,
        # harmonised_emissions=harmonised,
        # infilled_emissions=infilled,
    )

    return res
