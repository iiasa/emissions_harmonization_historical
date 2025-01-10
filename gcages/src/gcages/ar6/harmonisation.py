"""
Harmonisation part of the AR6 workflow
"""

from __future__ import annotations

import importlib
import multiprocessing

import pandas as pd
import pandas_indexing as pix  # type: ignore
from aneris.harmonize import Harmonizer
from attrs import define

from gcages.parallelisation import (
    assert_only_working_on_variable_unit_variations,
    run_parallel,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


def load_ar6_historical_emissions() -> pd.DataFrame:
    """
    Load the historical emissions that were used in AR6

    The data is massaged to what is expected by our harmonisation,
    it isn't the raw data (at least not raw metadata).

    Returns
    -------
    :
        Historical emissions used in AR6
    """
    res: pd.DataFrame = pd.read_csv(
        importlib.resources.files("gcages") / "history_ar6.csv"
    )
    res.columns = res.columns.str.lower()
    res = res.set_index(["model", "scenario", "variable", "unit", "region"])
    res.columns = res.columns.astype(int)

    res = pix.assignlevel(
        res,
        variable=res.pix.unique("variable").map(  # type: ignore
            lambda x: x.replace("AR6 climate diagnostics|", "").replace(
                "|Unharmonized", ""
            )
        ),
    )

    # Strip out any units that won't play nice with pint
    res = strip_pint_incompatible_characters_from_units(res, units_index_level="unit")

    # We only care about data from 1990 onwards (really, only 2015, but ok).
    res = res.loc[:, 1990:]

    return res


def add_historical_year_based_on_scaling(
    year_to_add: int,
    year_calc_scaling: int,
    emissions: pd.DataFrame,
    emissions_historical: pd.DataFrame,
    ms: tuple[str, ...] = ("model", "scenario"),
) -> pd.DataFrame:
    """
    Add a historical emissions year based on scaling

    Parameters
    ----------
    year_to_add
        Year to add

    year_calc_scaling
        Year to use to calculate the scaling

    emissions
        Emissions to which to add data for `year_to_add`

    emissions_historical
        Historical emissions to use to calculate
        the fill values based on scaling

    ms
        Name of the model and scenario columns.

        These have to be dropped from `emissions_historical`
        before everything will line up.

    Returns
    -------
    :
        `emissions` with data for `year_to_add`
        based on the scaling between `emissions`
        and `emissions_historical` in `year_calc_scaling`.
    """
    if emissions.pix.unique(["model", "scenario"]).shape[0] > 1:  # type: ignore
        # Processing is much trickier with multiple scenarios
        raise NotImplementedError

    ms = ("model", "scenario")
    # emissions_no_ms = emissions.reset_index(ms, drop=True)
    emissions_historical_common_vars = emissions_historical.loc[
        pix.isin(variable=emissions.pix.unique("variable"))  # type: ignore
    ]

    emissions_historical_no_ms = emissions_historical_common_vars.reset_index(
        ms, drop=True
    )

    scale_factor = emissions[year_calc_scaling].divide(
        emissions_historical_no_ms[year_calc_scaling]
    )
    fill_value = scale_factor.multiply(emissions_historical_no_ms[year_to_add])
    fill_value.name = year_to_add

    out = pd.concat([emissions, fill_value], axis="columns").sort_index(axis="columns")

    return out


def _check_data(hist, scen, year):
    # TODO: push back upstream
    check = ["region", "variable"]

    def downselect(df):
        return pix.projectlevel(df.index, check)

    s = downselect(scen)
    h = downselect(hist)
    if h.empty:
        raise ValueError("No historical data in harmonization year")
        # raise MissingHarmonisationYear("No historical data in harmonization year")

    if not s.difference(h).empty:
        # raise MissingHistoricalError(
        raise ValueError(
            "Historical data does not match scenario data in harmonization "
            f"year for\n {s.difference(h)}"
        )


def _convert_units_to_match(
    start: pd.DataFrame, match: pd.DataFrame, copy_on_entry: bool = True
):
    # TODO: push back upstream
    if copy_on_entry:
        out = start.copy()

    else:
        out = start

    differences = pix.projectlevel(match.index, ["variable", "unit"]).difference(
        pix.projectlevel(start.index, ["variable", "unit"])
    )
    if not differences.empty:
        updated = []
        for variable, target_unit in differences:
            v_loc = pix.isin(variable=variable)
            updated.append(out.loc[v_loc].pix.convert_unit(target_unit))
            out = out.loc[~v_loc]

        out = pix.concat([out, *updated])

    return out


def _knead_overrides(
    overrides: pd.DataFrame, scen: pd.DataFrame, harm_idx: pd.MultiIndex
) -> pd.DataFrame:
    # TODO: push back upstream
    """
    Process overrides to get a form readable by aneris, supporting many different use cases.

    Parameters
    ----------
    overrides : pd.DataFrame or pd.Series
    scen : pyam.IamDataFrame with data for single scenario and model instance
    """
    if overrides is None:
        return None

    # massage into a known format
    # check if no index and single value - this should be the override for everything
    if overrides.index.names == [None] and len(overrides["method"]) == 1:
        _overrides = pd.Series(
            overrides["method"].iloc[0],
            index=pd.Index(scen.region, name=harm_idx[-1]),  # only need to match 1 dim
            name="method",
        )
    # if data is provided per model and scenario, get those explicitly
    elif set(["model", "scenario"]).issubset(set(overrides.index.names)):
        _overrides = overrides.loc[
            pix.isin(model=scen.model, scenario=scen.scenario)
        ].droplevel(["model", "scenario"])
    # some of expected idx in cols, make it a multiindex
    elif isinstance(overrides, pd.DataFrame) and set(harm_idx) & set(overrides.columns):
        idx = list(set(harm_idx) & set(overrides.columns))
        _overrides = overrides.set_index(idx)["method"]
    else:
        _overrides = overrides

    # do checks
    if isinstance(_overrides, pd.DataFrame) and _overrides.isnull().any(axis=None):
        missing = _overrides.loc[_overrides.isnull().any(axis=1)]
        # raise AmbiguousHarmonisationMethod(
        raise ValueError(f"Overrides are missing for provided data:\n" f"{missing}")
    if _overrides.index.to_frame().isnull().any(axis=None):
        missing = _overrides[_overrides.index.to_frame().isnull().any(axis=1)]
        # raise AmbiguousHarmonisationMethod(
        raise ValueError(f"Defined overrides are missing data:\n" f"{missing}")
    if _overrides.index.duplicated().any():
        # raise AmbiguousHarmonisationMethod(
        raise ValueError(
            "Duplicated values for overrides:\n"
            f"{_overrides[_overrides.index.duplicated()]}"
        )

    return _overrides


def harmonise_all(
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.DataFrame | None = None,
):
    """
    Harmonise all timeseries in `scenarios` to match `history`

    This is a re-write of aneris` version of the same.
    TODO: MR upstream.

    Parameters
    ----------
    scenarios
        `pd.DataFrame` containing the timeseries to be harmonised

    history
        `pd.DataFrame` containing the historical timeseries to which
        `scenarios` should be harmonised.

    year
        The year in which `scenarios` should be harmonised to `history`

    overrides
        If not provided, the default aneris decision tree is used.

        Otherwise, `overrides` must be a `pd.DataFrame`
        containing any specifications for overriding the default aneris methods.
        Each row specifies one override.
        The override method is specified in the "method" columns.
        The other columns specify which of the timeseries in
        `scenarios` should use this override by specifying metadata to match
        ( e.g. variable, region).
        If a cell has a null value (evaluated using `pd.isnull()`)
        then that scenario characteristic will not be used for
        filtering for that override.
        For example, if you have a row with "method" equal to "constant_ratio",
        region equal to "World" and variable is null
        then all timeseries in the "World" region will use the "constant_ratio" method.
        In contrast, if you have a row with "method" equal to "constant_ratio",
        region equal to "World" and variable is "Emissions|CO2"
        then only timeseries with variable equal to "Emissions|CO2"
        and region equal to "World" will use the "constant_ratio" method.

    Returns
    -------
    :
        The harmonised timeseries

    Notes
    -----
    This interface is nowhere near as sophisticated as aneris' other interfaces.
    It simply harmonises timeseries.
    It does not check sectoral sums or other possible errors which can arise when harmonising.
    If you need such features, do not use this interface.

    Raises
    ------
    MissingHistoricalError
        No historical data is provided for a given timeseries

    MissingHarmonisationYear
        A value for the harmonisation year is missing or is null in `history`

    AmbiguousHarmonisationMethod
        `overrides` do not uniquely specify the harmonisation method for a given timeseries.
    """
    sidx = scenarios.index  # save in case we need to re-add extraneous indicies later

    dfs = []
    for (model, scenario), msdf in scenarios.groupby(["model", "scenario"]):
        hist_msdf = history.loc[
            pix.isin(region=msdf.pix.unique("region"))
            & pix.isin(variable=msdf.pix.unique("variable"))
        ]
        _check_data(hist_msdf, msdf, year)
        # pix.set_openscm_registry_as_default()
        # hist_msdf = hist_msdf.pix.convert_unit({"Mt BC/yr": "Mt BC / yr"})
        hist_msdf = _convert_units_to_match(start=hist_msdf, match=msdf)
        # need to convert to internal datastructure
        level_order = ["model", "scenario", "region", "variable", "unit"]
        msdf_aneris = msdf.reorder_levels(level_order)
        hist_msdf_aneris = hist_msdf.reorder_levels(level_order)
        h = Harmonizer(msdf_aneris, hist_msdf_aneris, harm_idx=["variable", "region"])
        # knead overrides
        _overrides = _knead_overrides(overrides, msdf, harm_idx=["variable", "region"])
        result = h.harmonize(year=year, overrides=_overrides)
        # need to convert out of internal datastructure
        dfs.append(
            result.assign(model=model, scenario=scenario).set_index(
                ["model", "scenario"], append=True
            )
        )

    # realign indicies if more than standard IAMC_IDX were there originally
    result = pix.concat(dfs)
    result = pix.semijoin(result, sidx, how="right").reorder_levels(sidx.names)

    return result


def harmonise_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.DataFrame | None,
    calc_scaling_year: int,
) -> pd.DataFrame:
    """
    Harmonise a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    history
        History to harmonise to

    year
        Year to use for harmonisation

    overrides
        Overrides to pass to aneris

    calc_scaling_year
        Year to use for calculating scaling if `year` is not in `indf`

    Returns
    -------
    :
        Harmonised scenario
    """
    assert_only_working_on_variable_unit_variations(indf)

    # A bunch of other fix ups that were applied in AR6
    if year not in indf:
        emissions_to_harmonise = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf,
            emissions_historical=history,
        )

    elif indf[year].isnull().any():
        null_emms_in_harm_year = indf[year].isnull()

        dont_change = indf[~null_emms_in_harm_year]

        updated = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf[null_emms_in_harm_year].drop(year, axis="columns"),
            emissions_historical=history,
        )

        emissions_to_harmonise = pd.concat([dont_change, updated])

    else:
        emissions_to_harmonise = indf

    # In AR6, any emissions with zero in the harmonisation year were dropped
    emissions_to_harmonise = emissions_to_harmonise[
        ~(emissions_to_harmonise[year] == 0.0)
    ]

    # In AR6, we interpolated before harmonising
    # First, check that there are no nans in the max year.
    # I don't know what happens in that case.
    if emissions_to_harmonise[emissions_to_harmonise.columns.max()].isnull().any():
        raise NotImplementedError

    # Then interpolate
    out_interp_years = list(range(year, emissions_to_harmonise.columns.max() + 1))
    emissions_to_harmonise = emissions_to_harmonise.reindex(
        columns=out_interp_years
    ).interpolate(method="slinear", axis="columns")

    harmonised = harmonise_all(
        emissions_to_harmonise,
        history=history,
        year=year,
        overrides=overrides,
    )

    return harmonised


@define
class AR6Harmoniser:
    """
    Harmoniser that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
    """

    historical_emissions: pd.DataFrame
    """
    Historical emissions to use for harmonisation
    """

    harmonisation_year: int
    """
    Year in which to harmonise
    """

    calc_scaling_year: int
    """
    Year to use for calculating a scaling factor from historical

    This is only needed if `self.harmonisation_year`
    is not in the emissions to be harmonised.

    For example, if `self.harmonisation_year` is 2015
    and `self.calc_scaling_year` is 2010
    and we have a scenario without 2015 data,
    then we will use the difference from historical in 2010
    to infer a value for 2015.

    This logic was perculiar to AR6, it may not be repeated.
    """

    aneris_overrides: pd.DataFrame | None
    """
    Overrides to supply to `aneris.convenience.harmonise_all`

    For source code and docs,
    see e.g. https://github.com/iiasa/aneris/blob/v0.4.2/src/aneris/convenience.py.
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
        Harmonise

        Parameters
        ----------
        in_emissions
            Emissions to harmonise

        Returns
        -------
        :
            Harmonised emissions
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in raw_emissions
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        # May need to drop all nan times here
        # May need to drop out variables which are all zero
        # May need to drop out variables which are zero in the harmonisation year

        harmonised_df = pix.concat(
            run_parallel(  # type: ignore
                func_to_call=harmonise_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                ),
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

        # Apply AR6 naming scheme
        out: pd.DataFrame = harmonised_df.pix.format(
            variable="AR6 climate diagnostics|Harmonized|{variable}"
        )

        # TODO:
        #   - enable optional checks for:
        #       - input and output metadata is identical
        #         (except maybe a stage indicator)
        #           - no mangled variable names
        #           - no mangled units
        #           - output timesteps are from harmonisation year onwards,
        #             otherwise identical to input
        #       - output scenarios all have common starting point

        return out

    @classmethod
    def from_ar6_like_config(
        cls, run_checks: bool = True, n_processes: int = multiprocessing.cpu_count()
    ) -> AR6Harmoniser:
        """
        Initialise from config (exactly) like what was used in AR6

        Parameters
        ----------
        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        n_processes
            Number of processes to use for parallel processing.

            Set to 1 to process in serial.

        Returns
        -------
        :
            Initialised harmoniser
        """
        historical_emissions = load_ar6_historical_emissions()

        # All variables not mentioned here use aneris' default decision tree
        aneris_overrides_ar6 = pd.DataFrame(
            [
                {
                    # high historical variance (cov=16.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC",
                },
                {
                    # high historical variance (cov=16.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C2F6",
                },
                {
                    # high historical variance (cov=15.4)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|C6F14",
                },
                {
                    # high historical variance (cov=11.2)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|PFC|CF4",
                },
                {
                    # high historical variance (cov=15.4)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|CO",
                },
                {
                    # always ratio method by choice
                    "method": "reduce_ratio_2080",
                    "variable": "Emissions|CO2",
                },
                {
                    # high historical variance,
                    # but using offset method to prevent diff
                    # from increasing when going negative rapidly (cov=23.2)
                    "method": "reduce_offset_2150_cov",
                    "variable": "Emissions|CO2|AFOLU",
                },
                {
                    # always ratio method by choice
                    "method": "reduce_ratio_2080",
                    "variable": "Emissions|CO2|Energy and Industrial Processes",
                },
                {
                    # basket not used in infilling
                    # (sum of f-gases with low model reporting confidence)
                    "method": "constant_ratio",
                    "variable": "Emissions|F-Gases",
                },
                {
                    # basket not used in infilling
                    # (sum of subset of f-gases with low model reporting confidence)
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC125",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC134a",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC143a",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC227ea",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC23",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC32",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|HFC|HFC43-10",
                },
                {
                    # high historical variance (cov=18.5)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|OC",
                },
                {
                    # minor f-gas with low model reporting confidence
                    "method": "constant_ratio",
                    "variable": "Emissions|SF6",
                },
                {
                    # high historical variance (cov=12.0)
                    "method": "reduce_ratio_2150_cov",
                    "variable": "Emissions|VOC",
                },
            ]
        )

        # TODO: turn checks back on
        return cls(
            historical_emissions=historical_emissions,
            harmonisation_year=2015,
            calc_scaling_year=2010,
            aneris_overrides=aneris_overrides_ar6,
            run_checks=run_checks,
            n_processes=n_processes,
        )
