"""
Tests of conversion to xarray
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import pandas_indexing as pix  # noqa: F401
import xarray as xr


def _metadata_to_xr(
    metadata: pd.DataFrame,
    timeseries_id_name: str = "ts_id",
) -> xr.Dataset:
    # Assumes timeseries are in rows
    ts_ids = np.arange(metadata.shape[0])
    timeseries_id_coord = xr.Coordinates({timeseries_id_name: ts_ids})
    metadata_columns = metadata.columns.tolist()
    index_darrays = {}
    for metadata_col in metadata_columns:
        metadata_col_unique_vals = metadata[metadata_col].unique()
        metadata_col_map = {}
        metadata_col_mapped_values = [""] * len(metadata_col_unique_vals)
        for i, v in enumerate(metadata_col_unique_vals):
            metadata_col_map[v] = i
            metadata_col_mapped_values[i] = i

        metadata_map_xr = xr.DataArray(
            metadata_col_unique_vals,
            # TODO: make naming injectable (?)
            dims=[f"{metadata_col}_map"],
            coords={f"{metadata_col}_map": metadata_col_mapped_values},
        )
        metadata_int_xr = xr.DataArray(
            # Not sure if converting to category first then mapping
            # is faster here or not, would have to profile.
            metadata[metadata_col].map(metadata_col_map),
            dims=[timeseries_id_name],
            coords=timeseries_id_coord,
        )

        index_darrays[metadata_col] = metadata_map_xr
        # TODO: make naming injectable (?)
        index_darrays[f"{metadata_col}_int"] = metadata_int_xr

    index_xr = xr.Dataset(index_darrays)

    return index_xr


def _data_to_xr(
    df: pd.DataFrame,
    timeseries_id_name: str = "ts_id",
    time_axis: str = "time",
    values_name: str = "values",
) -> xr.Dataset:
    # Possible to flatten the arrays,
    # but more than zero mucking around.
    # Really, just working in pandas seems like a much easier solution
    # rather than fighting xarray the whole time
    # (unless xarray with sparse data 'just works').
    data_rs = df.reset_index()
    data_index_xr = _metadata_to_xr(
        data_rs[df.index.names], timeseries_id_name=timeseries_id_name
    )
    data_values_xr = xr.DataArray(
        data_rs[df.columns],
        dims=[timeseries_id_name, time_axis],
        coords={timeseries_id_name: data_rs.index.values, time_axis: df.columns},
    )
    data_xr = xr.merge([data_index_xr, data_values_xr.to_dataset(name=values_name)])

    return data_xr


def _loc(
    ds: xr.Dataset,
    locs: dict[str, Any],
    timeseries_id_name: str = "ts_id",
) -> xr.Dataset:
    # You can do locs on the non-sparse format,
    # but it's more mucking around than I'd like.
    ts_ids = ds[timeseries_id_name]
    locs_map = {}
    for metadata_col, values in locs.items():
        values_1d = np.atleast_1d(values)

        # TODO: make naming injectable (?)
        values_int = ds[f"{metadata_col}_map"][ds[metadata_col].isin(values_1d)]
        # TODO: make naming injectable (?)
        locs_map[f"{metadata_col}_map"] = values_int
        ts_ids = np.intersect1d(
            ts_ids,
            # TODO: make naming injectable (?)
            ds[timeseries_id_name][ds[f"{metadata_col}_int"].isin(values_int)],
        )

    res = ds.loc[{timeseries_id_name: ts_ids} | locs_map]

    return res


def _to_df(ds: xr.Dataset, category_index: bool = True) -> pd.DataFrame:
    # For almost all operations, this is the pattern I would use:
    #
    # 1. Drop out to pandas
    # 2. Do the stuff
    #
    # In other words, I think using xarray for tabular data
    # just leads to fighting xarray.
    # Really, we should just use pandas with some accessor.
    # I wonder how much of what we need is already in here:
    # https://github.com/coroa/pandas-indexing/blob/main/src/pandas_indexing/iamc/resolver.py
    data = ds["values"].to_pandas()

    # TODO: make naming injectable (?)
    metadata_columns = [v.replace("_map", "") for v in ds.coords if v.endswith("_map")]

    index_cols = {}
    for metadata_col in metadata_columns:
        # TODO: make naming injectable (?)
        metadata_col_series_int = ds[f"{metadata_col}_int"].to_pandas()
        if category_index:
            metadata_col_series_int = metadata_col_series_int.astype("category")

        metadata_col_map = ds[metadata_col].to_pandas()

        index_cols[metadata_col] = metadata_col_series_int.map(metadata_col_map)

    index = pd.DataFrame(index_cols)

    res = pd.concat([index, data], axis="columns").set_index(metadata_columns)

    return res


def _to_array_like(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    # Unstack into a more standard array-like form
    tmp = _to_df(ds)
    tmp.columns.name = time_name

    res_d = {}
    for variable, vdf in tmp.groupby("variable", observed=True):
        unit_u = vdf.pix.unique("units")
        if len(unit_u) != 1:
            raise AssertionError(unit_u)
        unit = unit_u[0]

        res_d[variable] = (
            vdf.reset_index("units", drop=True)
            .unstack("variable")
            .stack(time_name, future_stack=True)
            .to_xarray()[variable]
        )
        res_d[variable].attrs["units"] = unit
        # # The above allows the below
        # import pint_xarray
        #
        # res_d[variable].pint.quantify()

    res = xr.Dataset(res_d)

    return res


def to_xarray(df: pd.DataFrame) -> xr.Dataset:
    # # I don't like this approach because, yes, it's stacked,
    # # but the reduction in the size of the stacked dimensions
    # # is non-existent: they all have length 3 * 4 * 5,
    # # rather than the length of their unique values.
    # xr.DataArray(
    #     np.random.default_rng().random((3, 4, 5)),
    #     dims=("a", "b", "c"),
    #     coords=dict(
    #         a=["s1", "s2", "s3"],
    #         b=np.arange(4),
    #         c=np.arange(5.0),
    #     ),
    # )
    data_xr = _data_to_xr(df)

    return data_xr


def test_non_sparse():
    n_scenarios = 3
    n_variables = 4
    n_runs = 5
    units = "Mt"
    timepoints = np.arange(2005.0, 2010.0, 1.0)
    idx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            (
                (s, v, r, units)
                for s, v, r in itertools.product(
                    [f"scenario_{i}" for i in range(n_scenarios)],
                    [f"variable_{i}" for i in range(n_variables)],
                    [i for i in range(n_runs)],
                )
            ),
            columns=["scenario", "variable", "run", "units"],
        )
    )

    n_ts = n_scenarios * n_variables * n_runs
    start = pd.DataFrame(
        50.0
        * np.linspace(0.3, 1, n_ts)[:, np.newaxis]
        * np.linspace(0, 1, timepoints.size)[np.newaxis, :]
        + np.random.default_rng().random((n_ts, timepoints.size)),
        columns=timepoints,
        index=idx,
    )

    res = to_xarray(start)

    assert not res["values"].isnull().any()
    for col in start.index.names:
        assert set(res[col].values) == set(start.pix.unique(col))
        lookup_val = start.pix.unique(col)[0]
        loced = _loc(res, {col: lookup_val})
        assert set(loced[col].values) == set([lookup_val])

    loced_h = _loc(res, {"scenario": ["scenario_0", "scenario_2"], "run": [1, 3]})
    _to_df(loced_h)
    _to_df(_loc(res, {"variable": "variable_1"}))
    _to_df(_loc(res, {"variable": ["variable_1", "variable_0"], "run": 1}))

    array_like = _to_array_like(res)

    null_count = array_like.isnull().sum()
    not_null_count = (~array_like.isnull()).sum()

    null_fraction = null_count / (null_count + not_null_count)

    # No sparsity, hence no nulls even when we go to a more standard array-like
    assert all(v for v in null_fraction == 0.0)

    import pint
    import pint_xarray  # noqa: F401

    array_like_q = array_like.pint.quantify()
    assert all(
        isinstance(array_like_q[v].data, pint.Quantity) for v in array_like_q.data_vars
    )


def test_sparse():
    timepoints = np.arange(1750.0, 2100.0, 1.0)
    idx = pd.MultiIndex.from_tuples(
        (
            ("scenario_a", "model_a", "v_1", 1, "Mt"),
            ("scenario_aa", "model_a", "v_1", 1, "Mt"),
            ("scenario_b", "model_b", "v_1", 1, "Mt"),
            ("scenario_bb", "model_b", "v_1", 1, "Mt"),
        ),
        names=["scenario", "model", "variable", "run", "units"],
    )

    start = pd.DataFrame(
        np.random.default_rng().random((4, timepoints.size)),
        columns=timepoints,
        index=idx,
    )

    res = to_xarray(start)

    assert not res["values"].isnull().any()
    for col in start.index.names:
        assert set(res[col].values) == set(start.pix.unique(col))
        lookup_val = start.pix.unique(col)[0]
        loced = _loc(res, {col: lookup_val})
        assert set(loced[col].values) == set([lookup_val])

    loced_h = _loc(res, {"scenario": ["scenario_a", "scenario_b"], "run": [1]})
    _to_df(loced_h)
    _to_df(_loc(res, {"variable": "v_1"}))
    _to_df(_loc(res, {"variable": ["v_1"], "run": 1}))
    _to_df(
        _loc(res, {"scenario": ["scenario_a", "scenario_b"], "model": "model_a"})
    ).iloc[:, :5]
    _to_df(
        _loc(res, {"model": "model_a", "scenario": ["scenario_a", "scenario_b"]})
    ).iloc[:, :5]

    array_like = _to_array_like(res)

    null_count = array_like.isnull().sum()
    not_null_count = (~array_like.isnull()).sum()

    null_fraction = null_count / (null_count + not_null_count)

    # Half the values are null because of sparsity
    assert all(v for v in null_fraction == 0.5)

    import pint
    import pint_xarray  # noqa: F401

    array_like_q = array_like.pint.quantify()
    assert all(
        isinstance(array_like_q[v].data, pint.Quantity) for v in array_like_q.data_vars
    )
