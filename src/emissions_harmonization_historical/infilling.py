"""
Infilling configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pyam
import tqdm.auto

if TYPE_CHECKING:
    import silicone.database_crunchers.base


def get_silicone_based_infiller(
    infilling_db: pd.DataFrame,
    follower_variable: str,
    lead_variables: list[str],
    silicone_db_cruncher: silicone.database_crunchers.base._DatabaseCruncher,
    derive_relationship_kwargs: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller based on silicone

    Parameters
    ----------
    infilling_db
        Infilling database

    follower_variable
        The variable to infill

    lead_variables
        The variables used to infill `follower_variable`

    silicone_db_cruncher
        Silicone cruncher to use

    derive_relationship_kwargs
        Passed to `silicone_db_cruncher.derive_relationship`

    Returns
    -------
    :
        Function which can be used to infill `follower_variable` in scenarios
    """
    if derive_relationship_kwargs is None:
        derive_relationship_kwargs = {}

    silicone_infiller = silicone_db_cruncher(pyam.IamDataFrame(infilling_db)).derive_relationship(
        variable_follower=follower_variable, variable_leaders=lead_variables, **derive_relationship_kwargs
    )

    def res(inp: pd.DataFrame) -> pd.DataFrame:
        res_h = silicone_infiller(pyam.IamDataFrame(inp)).timeseries()
        # The fact that this is needed suggests there's a bug in silicone
        res_h = res_h.loc[:, inp.dropna(axis="columns", how="all").columns]

        return res_h

    return res


def get_direct_copy_infiller(variable: str, copy_from: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller which just copies the timeseries from another scenario

    Parameters
    ----------
    variable
        Variable to infill

    copy_from
        Scenario to copy from

    Returns
    -------
    :
        Infiller which can infill data for `variable`
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        model_l = inp.pix.unique("model")
        if len(model_l) != 1:
            raise AssertionError(model_l)
        model = model_l[0]

        scenario_l = inp.pix.unique("scenario")
        if len(scenario_l) != 1:
            raise AssertionError(scenario_l)
        scenario = scenario_l[0]

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
    with smarter handling of pre-industrial levels.
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        lead_df = inp.loc[pix.isin(variable=[leader])]

        follow_df = (scaling_factor * (lead_df - l_0) + f_0).pix.assign(variable=follower, unit=f_unit)
        if not np.isclose(follow_df[calculation_year], f_calculation_year).all():
            raise AssertionError

        return follow_df

    return infiller


def infill(indf: pd.DataFrame, infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame | None:
    """
    Infill an emissions scenario using the provided infillers

    Parameters
    ----------
    indf
        Emissions scenario to infill

    infillers
        Infillers to use

        Each key is the gas the infiller can infill.
        Each value is the function which does the infilling.

    Returns
    -------
    :
        Infilled timeseries.

        If nothing was infilled, `None` is returned
    """
    infilled_l = []
    for variable in tqdm.auto.tqdm(infillers):
        for (model, scenario), msdf in indf.groupby(["model", "scenario"]):
            if variable not in msdf.index.get_level_values("variable"):
                infilled_l.append(infillers[variable](msdf))

    if not infilled_l:
        return None

    return pix.concat(infilled_l)


def get_complete(indf: pd.DataFrame, infilled: pd.DataFrame | None) -> pd.DataFrame:
    """
    Get a complete set of timeseries

    This is just a convenience function to help deal with the fact
    that [infill][] can return `None`.

    Parameters
    ----------
    indf
        Input data

    infilled
        Results of infilling using [infill][]

    Returns
    -------
    :
        Complete data i.e. the combination of `indf` and `infilled`
    """
    if infilled is not None:
        complete = pix.concat([indf, infilled])

    else:
        complete = indf

    return complete
