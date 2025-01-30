"""
Post-processing of results
"""

from __future__ import annotations

import multiprocessing

import pandas as pd
import pandas_indexing as pix
from attrs import define


def get_temperatures_in_line_with_assessment(
    raw_temperatures: pd.DataFrame,
    assessment_median: float,
    assessment_time_period: tuple[int, ...],
    assessment_pre_industrial_period: tuple[int, ...],
) -> pd.DataFrame:
    """
    Get temperatures in line with the historical assessment

    Parameters
    ----------
    raw_temperatures
        Raw temperatures

    assessment_median
        Median of the assessment to match

    assessment_time_period
        Time period over which the assessment applies

    assessment_pre_industrial_period
        Pre-industrial period used for the assessment


    Returns
    -------
    :
        Temperatures,
        adjusted so their medians are in line with the historical assessment.
    """
    rel_pi_temperatures = raw_temperatures.subtract(
        raw_temperatures.loc[:, list(assessment_pre_industrial_period)].mean(
            axis="columns"
        ),
        axis="rows",
    )
    mod_scen_medians = (
        rel_pi_temperatures.loc[:, list(assessment_time_period)]
        .mean(axis="columns")
        .groupby(["climate_model", "model", "scenario"])
        .median()
    )
    res = (
        rel_pi_temperatures.subtract(mod_scen_medians, axis="rows") + assessment_median
    )
    # Checker:
    # res.loc[:, list(assessment_time_period)].mean(axis="columns").groupby( ["model", "scenario"]).median()  # noqa: E501

    return res


def categorise_scenarios(
    temperatures_in_line_with_assessment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Categorise scenarios

    Parameters
    ----------
    temperatures_in_line_with_assessment
        Temperatures in line with the historical assessment

    Returns
    -------
    :
        Scenario categorisation
    """
    if len(temperatures_in_line_with_assessment.pix.unique("climate_model")) > 1:
        raise NotImplementedError

    peak_warming_quantiles = (
        temperatures_in_line_with_assessment.max(axis="columns")
        .groupby(["model", "scenario"])
        .quantile([0.33, 0.5, 0.67])
        .unstack()
    )
    eoc_warming_quantiles = (
        temperatures_in_line_with_assessment[2100]
        .groupby(["model", "scenario"])
        .quantile([0.5])
        .unstack()
    )
    categories = pd.Series(
        "C8: exceed warming of 4°C (>=50%)",
        index=peak_warming_quantiles.index,
        name="category_name",
    )
    categories[peak_warming_quantiles[0.5] < 4.0] = "C7: limit warming to 4°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.5] < 3.0] = "C6: limit warming to 3°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.5] < 2.5] = "C5: limit warming to 2.5°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.5] < 2.0] = "C4: limit warming to 2°C (>50%)"  # noqa: PLR2004
    categories[peak_warming_quantiles[0.67] < 2.0] = "C3: limit warming to 2°C (>67%)"  # noqa: PLR2004
    categories[
        (peak_warming_quantiles[0.33] > 1.5) & (eoc_warming_quantiles[0.5] < 1.5)  # noqa: PLR2004
    ] = "C2: return warming to 1.5°C (>50%) after a high overshoot"
    categories[
        (peak_warming_quantiles[0.33] <= 1.5) & (eoc_warming_quantiles[0.5] < 1.5)  # noqa: PLR2004
    ] = "C1: limit warming to 1.5°C (>50%) with no or limited overshoot"

    out = categories.to_frame()
    out["category"] = out["category_name"].apply(lambda x: x.split(":")[0])

    return out


def get_exceedance_probability(indf: pd.DataFrame, warming_level: float) -> float:
    """
    Get exceedance probability

    For exceedance probability over time
    (i.e. at each timestep, rather than at any point in the simulation),
    see `get_exceedance_probability_over_time`.

    This assumes that the data has already been grouped appropriately
    e.g. by scenario.

    Parameters
    ----------
    indf
        Temperature data from which to calculate the exceedance probability

    warming_level
        Warming level that defines exceedance

    Returns
    -------
    :
        Exceedance probability for `indf`.
    """
    peaks = indf.max(axis="columns")
    n_above_level = (peaks > warming_level).sum(axis="rows")
    ep = n_above_level / peaks.shape[0] * 100

    return ep


def get_exceedance_probability_over_time(
    indf: pd.DataFrame, warming_level: float
) -> pd.Series[float]:
    """
    Get exceedance probability over time

    For exceedance probability at any point in the simulation,
    see `get_exceedance_probability`

    This assumes that the data has already been grouped appropriately
    e.g. by scenario.

    Parameters
    ----------
    indf
        Input data

    warming_level
        Warming level that defines exceedance

    Returns
    -------
    :
        Exceedance probability over time for `indf`.
    """
    gt_wl = (indf > warming_level).sum(axis="rows")
    ep = gt_wl / indf.shape[0] * 100

    return ep


@define
class PostProcessingResult:
    """
    Results of post-processing
    """

    timeseries: pd.DataFrame
    """Timeseries, still including individual runs"""

    timeseries_aggregate: pd.DataFrame
    """Timeseries aggregate i.e. not including individual runs"""

    timeseries_percentiles: pd.DataFrame
    """Timeseries percentiles"""

    metadata: pd.DataFrame
    """Metadata for each scenario"""


@define
class PostProcessor:
    """
    Post-processor that follows the same logic as was used in AR6
    """

    gsat_variable_name: str
    """The name of the GSAT variable"""

    gsat_in_line_with_assessment_variable_name: str
    """The name of the GSAT variable once its been aligned with the assessment"""

    gsat_assessment_median: float
    """
    Median of the GSAT assessment
    """

    gsat_assessment_time_period: tuple[int, ...]
    """
    Time period over which the GSAT assessment applies
    """

    gsat_assessment_pre_industrial_period: tuple[int, ...]
    """
    Pre-industrial time period used for the GSAT assessment
    """

    percentiles_to_calculate: tuple[float, ...] = (0.05, 0.33, 0.5, 0.67, 0.95)
    """Percentiles to calculate and include in the output"""

    exceedance_global_warming_levels: tuple[float, ...] = (1.5, 2.0, 2.5)
    """
    Global-warming levels against which to calculate exceedance probabilities
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

    def __call__(self, in_df: pd.DataFrame) -> PostProcessingResult:
        """
        Do the post-processing

        Parameters
        ----------
        in_df
            Data to post-process

        Returns
        -------
        :
            Post-processed results
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in the output
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        temperatures_in_line_with_assessment = get_temperatures_in_line_with_assessment(
            in_df.loc[pix.isin(variable=[self.gsat_variable_name])],
            assessment_median=self.gsat_assessment_median,
            assessment_time_period=self.gsat_assessment_time_period,
            assessment_pre_industrial_period=self.gsat_assessment_pre_industrial_period,
        ).pix.assign(variable=self.gsat_in_line_with_assessment_variable_name)

        categories = categorise_scenarios(temperatures_in_line_with_assessment)
        peak_warming_quantiles = (
            temperatures_in_line_with_assessment.max(axis="columns")
            .groupby(["model", "scenario"])
            .quantile(list(self.percentiles_to_calculate))
            .unstack()
        )
        peak_warming_quantiles.columns = peak_warming_quantiles.columns.map(
            lambda x: f"Peak warming {x * 100}"
        )
        eoc_warming_quantiles = (
            temperatures_in_line_with_assessment[2100]
            .groupby(["model", "scenario"])
            .quantile(list(self.percentiles_to_calculate))
            .unstack()
        )
        eoc_warming_quantiles.columns = eoc_warming_quantiles.columns.map(
            lambda x: f"EOC warming {x * 100}"
        )

        exceedance_probabilities_l = []
        for gwl in self.exceedance_global_warming_levels:
            gwl_exceedance_probabilities_l = []
            for (
                model,
                scenario,
            ), msdf in temperatures_in_line_with_assessment.groupby(
                ["model", "scenario"]
            ):
                if len(msdf.pix.unique("climate_model")) != 1:
                    raise NotImplementedError

                ep = get_exceedance_probability(msdf, warming_level=gwl)
                ep_s = pd.Series(
                    ep,
                    name=f"{gwl:.2f}°C exceedance probability",
                    index=pd.MultiIndex.from_tuples(
                        ((model, scenario),), names=["model", "scenario"]
                    ),
                )
                gwl_exceedance_probabilities_l.append(ep_s)

            exceedance_probabilities_l.append(
                pix.concat(gwl_exceedance_probabilities_l)
            )

        exceedance_probabilities = (
            pix.concat(exceedance_probabilities_l, axis="columns").melt(
                ignore_index=False
            )
            # .pix.assign(unit="%")
        )
        exceedance_probabilities = exceedance_probabilities.pivot_table(
            values="value",
            columns="variable",
            index=exceedance_probabilities.index.names,
        )
        exceedance_probabilities.columns.name = None

        exceedance_probabilities_l = []
        for gwl in [1.5, 2.0, 2.5]:
            ep = (
                temperatures_in_line_with_assessment.groupby(
                    temperatures_in_line_with_assessment.index.names.difference(
                        ["variable", "unit", "run_id"]
                    )
                )
                .apply(get_exceedance_probability_over_time, warming_level=gwl)
                .pix.assign(unit="%", variable=f"{gwl:.2f}°C exceedance probability")
            )

            exceedance_probabilities_l.append(ep)

        exceedance_probabilities_over_time = pix.concat(exceedance_probabilities_l)

        temperatures_in_line_with_assessment_percentiles = (
            temperatures_in_line_with_assessment.groupby(
                ["climate_model", "model", "scenario", "variable", "region", "unit"]
            ).quantile(list(self.percentiles_to_calculate))
        )
        temperatures_in_line_with_assessment_percentiles.index.names = [
            *temperatures_in_line_with_assessment_percentiles.index.names[:-1],
            "quantile",
        ]
        temperatures_in_line_with_assessment_percentiles = (
            temperatures_in_line_with_assessment_percentiles.reset_index("quantile")
        )
        temperatures_in_line_with_assessment_percentiles["percentile"] = (
            (100 * temperatures_in_line_with_assessment_percentiles["quantile"])
            .round(1)
            .astype(str)
        )
        temperatures_in_line_with_assessment_percentiles = (
            temperatures_in_line_with_assessment_percentiles.drop(
                "quantile", axis="columns"
            ).set_index("percentile", append=True)
        )

        timeseries_l = [temperatures_in_line_with_assessment]
        timeseries = pix.concat(timeseries_l)
        timeseries.columns = timeseries.columns.astype(int)

        timeseries_percentiles_l = [temperatures_in_line_with_assessment_percentiles]
        timeseries_percentiles = pix.concat(timeseries_percentiles_l)
        timeseries_percentiles.columns = timeseries_percentiles.columns.astype(int)

        timeseries_aggregate_l = [exceedance_probabilities_over_time]
        timeseries_aggregate = pix.concat(timeseries_aggregate_l)
        timeseries_aggregate.columns = timeseries_aggregate.columns.astype(int)

        metadata_l = [
            categories,
            peak_warming_quantiles,
            eoc_warming_quantiles,
            exceedance_probabilities,
        ]
        metadata = pix.concat(metadata_l, axis="columns")

        # TODO:
        #   - enable optional checks for:
        #       - input and output scenarios are the same

        return PostProcessingResult(
            timeseries=timeseries,
            timeseries_aggregate=timeseries_aggregate,
            timeseries_percentiles=timeseries_percentiles,
            metadata=metadata,
        )
