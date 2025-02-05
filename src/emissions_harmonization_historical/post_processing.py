"""
Post-processing configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import multiprocessing

import pandas as pd
import pandas_indexing as pix
from attrs import define
from gcages.post_processing import (
    PostProcessingResult,
    categorise_scenarios,
    get_exceedance_probability,
    get_exceedance_probability_over_time,
    get_temperatures_in_line_with_assessment,
)


@define
class AR7FTPostProcessor:
    """
    AR7 fast-track post-processor
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
        peak_warming_quantiles.columns = peak_warming_quantiles.columns.map(lambda x: f"Peak warming {x * 100}")
        eoc_warming_quantiles = (
            temperatures_in_line_with_assessment[2100]
            .groupby(["model", "scenario"])
            .quantile(list(self.percentiles_to_calculate))
            .unstack()
        )
        eoc_warming_quantiles.columns = eoc_warming_quantiles.columns.map(lambda x: f"EOC warming {x * 100}")

        exceedance_probabilities_l = []
        for gwl in self.exceedance_global_warming_levels:
            gwl_exceedance_probabilities_l = []
            for (
                model,
                scenario,
            ), msdf in temperatures_in_line_with_assessment.groupby(["model", "scenario"]):
                if len(msdf.pix.unique("climate_model")) != 1:
                    raise NotImplementedError

                ep = get_exceedance_probability(msdf, warming_level=gwl)
                ep_s = pd.Series(
                    ep,
                    name=f"{gwl:.2f}°C exceedance probability",
                    index=pd.MultiIndex.from_tuples(((model, scenario),), names=["model", "scenario"]),
                )
                gwl_exceedance_probabilities_l.append(ep_s)

            exceedance_probabilities_l.append(pix.concat(gwl_exceedance_probabilities_l))

        exceedance_probabilities = (
            pix.concat(exceedance_probabilities_l, axis="columns").melt(ignore_index=False)
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
                    temperatures_in_line_with_assessment.index.names.difference(["variable", "unit", "run_id"])
                )
                .apply(get_exceedance_probability_over_time, warming_level=gwl)
                .pix.assign(unit="%", variable=f"{gwl:.2f}°C exceedance probability")
            )

            exceedance_probabilities_l.append(ep)

        exceedance_probabilities_over_time = pix.concat(exceedance_probabilities_l)

        timeseries_percentiles = pix.concat(
            [
                in_df,
                temperatures_in_line_with_assessment,
            ]
        )

        timeseries_percentiles = timeseries_percentiles.groupby(
            ["climate_model", "model", "scenario", "variable", "region", "unit"]
        ).quantile(list(self.percentiles_to_calculate))
        timeseries_percentiles.index.names = [
            *timeseries_percentiles.index.names[:-1],
            "quantile",
        ]
        timeseries_percentiles = timeseries_percentiles.reset_index("quantile")
        timeseries_percentiles["percentile"] = (100 * timeseries_percentiles["quantile"]).round(1).astype(str)
        timeseries_percentiles = timeseries_percentiles.drop("quantile", axis="columns").set_index(
            "percentile", append=True
        )
        timeseries_percentiles.columns = timeseries_percentiles.columns.astype(int)

        timeseries_l = [temperatures_in_line_with_assessment]
        timeseries = pix.concat(timeseries_l)
        timeseries.columns = timeseries.columns.astype(int)

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

    @classmethod
    def from_default_config(cls) -> AR7FTPostProcessor:
        """
        Initialise from default, hard-coded configuration
        """
        return cls(
            gsat_variable_name="Surface Air Temperature Change",
            gsat_in_line_with_assessment_variable_name="Assessed Surface Air Temperature Change",
            gsat_assessment_median=0.85,
            gsat_assessment_time_period=range(1995, 2014 + 1),
            gsat_assessment_pre_industrial_period=range(1850, 1900 + 1),
            percentiles_to_calculate=(0.05, 0.33, 0.5, 0.67, 0.95),
            exceedance_global_warming_levels=(1.5, 2.0, 2.5),
            # TODO: implement and activate
            run_checks=False,
        )
