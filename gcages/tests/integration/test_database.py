"""
Tests of our database (`gcages.database`)
"""

from __future__ import annotations

import itertools
import re
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.database import GCDB, AlreadyInDBError, EmptyDBError, GCDBDataFormat


def create_test_df(
    *,
    n_scenarios: int,
    n_variables: int,
    n_runs: int,
    timepoints: np.typing.NDArray[np.floating],
    units: str,
) -> pd.DataFrame:
    """
    Create a [`pd.DataFrame`][pandas.DataFrame] to use in testing

    This uses the idea of simple climate model runs,
    where you have a number of scenarios,
    each of which has a number of variables
    from a number of different model runs
    with output for a number of different time points.
    """
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
    df = pd.DataFrame(
        50.0
        * np.linspace(0.3, 1, n_ts)[:, np.newaxis]
        * np.linspace(0, 1, timepoints.size)[np.newaxis, :]
        + np.random.default_rng().random((n_ts, timepoints.size)),
        columns=timepoints,
        index=idx,
    )

    return df


@pytest.mark.parametrize(
    "format", (pytest.param(format, id=str(format)) for format in GCDBDataFormat)
)
def test_save_and_load(format, tmpdir):
    start = create_test_df(
        n_scenarios=20,
        n_variables=15,
        n_runs=60,
        # n_scenarios=90,
        # n_variables=15,
        # n_runs=600,
        timepoints=np.arange(1750, 2100),
        # n_scenarios=10,
        # n_variables=1,
        # n_runs=3,
        # timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    db = GCDB(tmpdir, format=format)

    db.save(start)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        start.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=int)

    pd.testing.assert_frame_equal(start, loaded)


def test_save_multiple_and_load(tmpdir):
    db = GCDB(tmpdir)

    all_saved_l = []
    for units in ["Mt", "Gt", "Tt"]:
        to_save = create_test_df(
            n_scenarios=10,
            n_variables=1,
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
            units=units,
        )

        db.save(to_save)
        all_saved_l.append(to_save)

    all_saved = pix.concat(all_saved_l)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        all_saved.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(all_saved, loaded)


def test_save_overwrite_error(tmpdir):
    db = GCDB(tmpdir)

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_variables=1,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    dup = cdf(units="m")
    db.save(dup)

    to_save = pix.concat([dup, cdf(units="km")])

    error_msg = re.escape(
        "The following rows are already in the database:\n"
        f"{dup.index.to_frame(index=False)}"
    )
    with pytest.raises(AlreadyInDBError, match=error_msg):
        db.save(to_save)


def test_save_overwrite_force(tmpdir):
    db = GCDB(Path(tmpdir))

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_variables=1,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    original = cdf(units="m")
    db.save(original)

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(original, loaded)

    original_overwrite = cdf(units="m")
    updated = pix.concat([original_overwrite, cdf(units="km")])

    # With force, we can overwrite
    db.save(updated, allow_overwrite=True)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the new file, but not the original file.
    db_files = list(db.db_dir.glob("*.csv"))
    assert set([f.name for f in db_files]) == {"1.csv", "index.csv", "filemap.csv"}

    # Check that the data was overwritten with new data
    try:
        pd.testing.assert_frame_equal(original, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same DataFrame,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        updated.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(updated, loaded)


def test_save_overwrite_force_half_overlap(tmpdir):
    db = GCDB(Path(tmpdir))

    cdf = partial(
        create_test_df,
        n_variables=5,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="m",
    )

    original = cdf(n_scenarios=6)
    db.save(original)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the file map file plus written file.
    db_files = list(db.db_dir.glob("*.csv"))
    assert set([f.name for f in db_files]) == {
        "0.csv",
        "index.csv",
        "filemap.csv",
    }

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(original, loaded)

    original_overwrite = cdf(n_scenarios=3)

    # With force, we can overwrite
    db.save(original_overwrite, allow_overwrite=True)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the file map file plus the newly written file
    # plus the re-written data file
    # (to handle the need to split the original data so we can keep only what we need),
    # but not the original file.
    db_files = list(db.db_dir.glob("*.csv"))
    assert set([f.name for f in db_files]) == {
        "1.csv",
        "2.csv",
        "index.csv",
        "filemap.csv",
    }

    # Check that the data was overwritten with new data
    overlap_idx = original.index.isin(original_overwrite.index)
    overlap = original.loc[overlap_idx]
    try:
        pd.testing.assert_frame_equal(overlap, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same values,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    update_exp = pix.concat([original.loc[~overlap_idx], original_overwrite])
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        update_exp.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load(out_columns_type=float)

    pd.testing.assert_frame_equal(update_exp, loaded)


@pytest.mark.parametrize(
    "meth, args",
    (
        ("delete", []),
        ("load", []),
        ("load_metadata", []),
        ("regroup", [["scenarios"]]),
        (
            "save",
            [
                create_test_df(
                    n_scenarios=1,
                    n_runs=1,
                    n_variables=1,
                    units="t",
                    timepoints=np.array([1.2]),
                )
            ],
        ),
    ),
)
def test_locking(tmpdir, meth, args):
    db = GCDB(Path(tmpdir))

    # Put some data in the db so there's something to lock
    db.save(
        create_test_df(
            n_scenarios=1,
            n_variables=1,
            n_runs=1,
            timepoints=np.array([10.0]),
            units="Mt",
        )
    )

    # Acquire the lock
    with db.index_file_lock:
        # Check that we can't re-acquire the lock to use the method
        with pytest.raises(filelock.Timeout):
            getattr(db, meth)(
                # Can't use defaults here as default is no timeout
                lock_context_manager=db.index_file_lock.acquire(timeout=0.0),
                *args,
            )

        with pytest.raises(filelock.Timeout):
            getattr(db, meth)(
                # Can't use defaults here as default is no timeout
                lock_context_manager=db.index_file_lock.acquire(timeout=0.1),
                *args,
            )

        # Unless we pass in a different context manager
        getattr(db, meth)(lock_context_manager=nullcontext(), *args)


def test_load_with_loc(tmpdir):
    db = GCDB(tmpdir)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

    for selector in [
        pix.isin(scenario=["scenario_1", "scenario_3"]),
        pix.isin(scenario=["scenario_1", "scenario_3"], variable=["variable_2"]),
        (
            pix.isin(scenario=["scenario_1", "scenario_3"])
            & pix.ismatch(variable=["variable_1*"])
        ),
        pix.isin(scenario=["scenario_1", "scenario_3"], variable=["variable_2"]),
    ]:
        loaded = db.load(selector, out_columns_type=float)
        exp = full_db.loc[selector]

        pd.testing.assert_frame_equal(loaded, exp)


def test_load_with_index_all(tmpdir):
    db = GCDB(tmpdir)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

    idx = full_db.index
    exp = full_db

    loaded = db.load(idx, out_columns_type=float)

    pd.testing.assert_frame_equal(loaded, exp)


@pytest.mark.parametrize(
    "slice",
    (slice(None, None, None), slice(None, 3, None), slice(2, 4, None), slice(1, 15, 2)),
)
def test_load_with_index_slice(tmpdir, slice):
    db = GCDB(tmpdir)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

    idx = full_db.index[slice]
    exp = full_db[slice]

    loaded = db.load(idx, out_columns_type=float)

    pd.testing.assert_frame_equal(loaded, exp)


@pytest.mark.parametrize(
    "levels",
    (
        pytest.param(["scenario"], id="first_level"),
        pytest.param(["variable"], id="not_first_level"),
        pytest.param(["scenario", "variable"], id="multi_level_in_order"),
        pytest.param(["scenario", "variable"], id="multi_level_non_adjacent"),
        pytest.param(["variable", "scenario"], id="multi_level_out_of_order"),
        pytest.param(["run", "variable"], id="multi_level_out_of_order_not_first"),
    ),
)
def test_load_with_pix_unique_levels(tmpdir, levels):
    db = GCDB(tmpdir)

    full_db = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    for _, pdf in full_db.groupby(["scenario"]):
        db.save(pdf)

    locator = None
    for level in levels:
        if locator is None:
            locator = pix.isin(**{level: full_db.pix.unique(level)[:2]})
        else:
            locator &= pix.isin(**{level: full_db.pix.unique(level)[:2]})

    exp = full_db.loc[locator]
    idx = exp.pix.unique(levels)

    loaded = db.load(idx, out_columns_type=float)

    pd.testing.assert_frame_equal(loaded, exp)


def test_deletion(tmpdir):
    db = GCDB(Path(tmpdir))

    db.save(
        create_test_df(
            n_scenarios=10,
            n_variables=3,
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
            units="Mt",
        )
    )

    assert isinstance(db.load(), pd.DataFrame)

    db.delete()

    with pytest.raises(EmptyDBError):
        db.load_metadata()

    with pytest.raises(EmptyDBError):
        db.load()


def test_regroup(tmpdir):
    db = GCDB(Path(tmpdir))

    all_dat = create_test_df(
        n_scenarios=10,
        n_variables=3,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    db.save(all_dat)

    pd.testing.assert_frame_equal(db.load(out_columns_type=float), all_dat)
    # Testing implementation but ok as a helper for now
    assert len(list(db.db_dir.glob("*.csv"))) == 3

    for new_grouping in (
        ["scenario"],
        ["scenario", "variable"],
        ["variable", "run"],
    ):
        db.regroup(new_grouping)

        # Make sure data unchanged
        pd.testing.assert_frame_equal(
            db.load(out_columns_type=float), all_dat, check_like=True
        )
        # Testing implementation but ok as a helper for now
        assert (
            len(list(db.db_dir.glob("*.csv")))
            == 2 + all_dat.pix.unique(new_grouping).shape[0]
        )
