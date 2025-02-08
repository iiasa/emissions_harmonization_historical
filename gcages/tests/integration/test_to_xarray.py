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
    convert_to_category: bool = True,
    timeseries_id_name: str = "ts_id",
    metadata_name: str = "metadata",
    metadata_column_int_values_name: str = "metadata_column_int",
    metadata_int_values_name: str = "metadata_int",
) -> xr.Dataset:
    if convert_to_category:
        metadata = metadata.astype("category")

    # Assumes timeseries are in rows
    ts_ids = np.arange(metadata.shape[0])
    metadata_columns = metadata.columns.tolist()
    index_darrays = {}
    for metadata_col in metadata_columns:
        metadata_col_unique_vals = metadata[metadata_col].values.categories
        metadata_col_map = {}
        metadata_col_mapped_values = [""] * len(metadata_col_unique_vals)
        for i, v in enumerate(metadata_col_unique_vals):
            metadata_col_map[v] = i
            metadata_col_mapped_values[i] = i

        metadata_xr = xr.DataArray(
            metadata_col_mapped_values,
            dims=[metadata_col],
            coords={metadata_col: metadata_col_unique_vals.values},
        )

        # TODO: make this injectable
        metadata_col_id = f"{metadata_col}_int"
        index_darrays[metadata_col_id] = metadata_xr
        metadata[metadata_col] = metadata[metadata_col].map(metadata_col_map)

    index_darrays[metadata_int_values_name] = xr.DataArray(
        metadata.values,
        dims=[timeseries_id_name, "metadata"],
        coords={timeseries_id_name: ts_ids, "metadata": metadata_columns},
    )

    index_xr = xr.Dataset(index_darrays)

    return index_xr


def _data_to_xr(
    df: pd.DataFrame,
    timeseries_id_name: str = "ts_id",
) -> xr.Dataset:
    data_rs = df.reset_index()
    data_index_xr = _metadata_to_xr(
        data_rs[df.index.names], timeseries_id_name=timeseries_id_name
    )
    data_values_xr = xr.DataArray(
        data_rs[df.columns],
        dims=[timeseries_id_name, "time"],
        coords={timeseries_id_name: data_rs.index.values, "time": df.columns},
    )
    data_xr = xr.merge([data_index_xr, data_values_xr.to_dataset(name="values")])

    return data_xr


def _loc(ds: xr.Dataset, locs: dict[str, Any]) -> xr.Dataset:
    res = ds
    ts_ids = res["ts_id"]
    for metadata_col, values in locs.items():
        values_1d = np.atleast_1d(values)

        metadata_col_int_values = res[f"{metadata_col}_int"].loc[
            {metadata_col: values_1d}
        ]
        metadata_col_idx = res["metadata"].values.tolist().index(metadata_col)
        ts_locs = (
            ds["metadata_int"][:, metadata_col_idx]
            .isin(metadata_col_int_values)
            .drop("metadata")
        )
        ts_ids = np.intersect1d(ts_ids, res["ts_id"][ts_locs])

    res = ds.loc[{"ts_id": ts_ids, **{c: np.atleast_1d(v) for c, v in locs.items()}}]

    return res


def _to_df(ds: xr.Dataset) -> pd.DataFrame:
    data = ds["values"].to_pandas()

    index = ds["metadata_int"].to_pandas().astype("category")
    index.columns = ds.metadata
    for column in index:
        col_map = {v: k for k, v in ds[f"{column}_int"].to_pandas().to_dict().items()}
        index[column] = index[column].map(col_map)

    index.columns.name = None
    index.index.name = None
    # # TODO: think through the category handling more
    # for c in index:
    #     index[c] = index[c].astype(index[c].dtype.categories.dtype)
    # # TODO: think through the type handling more
    # for c in index:
    #     if isinstance(index[c].values[0], np.int32):
    #         index[c] = index[c].astype(np.int64)

    res = pd.concat([index.loc[ds["ts_id"]], data], axis="columns").set_index(
        ds["metadata"].values.tolist()
    )

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
    # If you have a non-sparse array, this is all fine
    tmp = start.copy()
    tmp.columns.name = "year"
    tmp.unstack("variable").stack("year").to_xarray()

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


def test_sparse():
    timepoints = np.arange(1750.0, 2100.0, 1.0)
    idx = pd.MultiIndex.from_tuples(
        (
            ("scenario_a", "model_a", "v_1", 1, "Mt"),
            ("scenario_a", "model_aa", "v_1", 1, "Mt"),
            ("scenario_b", "model_b", "v_1", 1, "Mt"),
            ("scenario_b", "model_bb", "v_1", 1, "Mt"),
        ),
        names=["scenario", "model", "variable", "run", "units"],
    )

    start = pd.DataFrame(
        np.random.default_rng().random((4, timepoints.size)),
        columns=timepoints,
        index=idx,
    )

    res = to_xarray(start)
    # If you have a sparse array, this creates heaps of nans
    # so memory usage is much higher.
    tmp = start.copy()
    tmp.columns.name = "year"
    tmp.unstack("variable").stack("year").to_xarray()

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


# - variables with different units
# - unique model-scenario pairs
# - other weird sparsity combos
