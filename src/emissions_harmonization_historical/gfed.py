"""Here we specify functions related to processing GFED (Global Fire Emissions Database) data."""

from pathlib import Path

import cftime
import dask
import h5py
import numpy as np
import pandas as pd
import xarray as xr


def read_var(dataset, coords):
    """
    Read a dataset and converts it into an xarray DataArray using Dask for delayed computation.

    Parameters
    ----------
        dataset: The dataset to be converted (e.g., an h5py dataset or similar).
        coords: A list of coordinates for the DataArray.

    Returns
    -------
        xr.DataArray: The resulting DataArray with delayed loading via Dask, preserving
        the shape and data type of the original dataset.
    """
    return xr.DataArray(
        dask.array.from_delayed(dask.delayed(np.asarray)(dataset), shape=dataset.shape, dtype=dataset.dtype),
        coords=coords,
    )


def read_coords(file):
    """
    Read latitude and longitude coordinates from a file and converts them to pandas Index objects.

    Parameters
    ----------
        file: The file (likely HDF5) from which to read the latitude and longitude coordinates.

    Returns
    -------
        list: A list containing two pandas Index objects for latitude ('lat') and longitude ('lon').
    """
    return [
        pd.Index(file["lat"][:, 0], name="lat"),
        pd.Index(file["lon"][0, :], name="lon"),
    ]


def concat_group(group, concat_dim, coords, sep=None):
    """
    Concatenate datasets with possible sub-dimensions separated by 'sep'.

    Concatenate multiple datasets along a specified dimension and optionally splits this
    dimension into multiple sub-dimensions based on a separator.

    Parameters
    ----------
        group: A dictionary-like object containing datasets to concatenate.
        concat_dim: The dimension along which to concatenate the datasets.
        coords: Coordinates for the resulting concatenated DataArray.
        sep (optional): A string separator to split the concatenated dimension into sub-dimensions.

    Returns
    -------
        xr.DataArray: The concatenated DataArray, with the concatenated dimension optionally
        split into multiple dimensions if `sep` is provided.
    """
    names = list(group)
    da = xr.concat(
        [read_var(group[n], coords) for n in names],
        dim=pd.Index(names, name=concat_dim),
    )
    if sep is None:
        return da

    return da.assign_coords(
        {concat_dim: da.indexes[concat_dim].str.split(sep, expand=True).rename(concat_dim.split(sep))}
    ).unstack(concat_dim)


def read_monthly(group, coords):
    """
    Read and process monthly data.

    Read and process monthly data from a group of datasets, handling partitioned
    datasets where necessary.

    Parameters
    ----------
        group: A dictionary-like object containing monthly data, where each key represents a month.
        coords: Coordinates for the resulting datasets.

    Returns
    -------
        xr.Dataset: A concatenated dataset across all months.

    Special Handling:
        If the dataset contains a 'partitioning' group, it multiplies specific variables
        by the partitioning dataset to handle sectoral variables.
    """
    months = list(group)
    month_ds = []
    for month in months:
        if "partitioning" in group[month]:
            partition = concat_group(group[month]["partitioning"], "var_sector", coords, sep="_")
            sectoral_vars = frozenset(partition.indexes["var"])
        else:
            sectoral_vars = frozenset()

        monthly_das = {}
        for name in group[month]:
            h5ds = group[month][name]
            if not isinstance(h5ds, h5py.Dataset):
                continue

            da = read_var(h5ds, coords)
            if name in sectoral_vars:
                da = da * partition.sel(var=name, drop=True)
            monthly_das[name] = da

        month_ds.append(xr.Dataset(monthly_das))

    return xr.concat(month_ds, dim=pd.Index(months, name="month").astype(int))


def month_to_cftime(ds, year):
    """
    Convert month to CFTimeIndex using first day of each month.

    Convert the month dimension in a dataset to a CFTimeIndex representing the first day
    of each month, using a given year.

    Parameters
    ----------
        ds: The dataset containing a 'month' dimension.
        year: The year to use when generating the CFTimeIndex.

    Returns
    -------
        xr.Dataset: The dataset with the 'month' dimension replaced by a 'time' dimension in CFTime format.
    """
    return (
        ds.assign_coords(
            time=(
                "month",
                xr.CFTimeIndex([cftime.datetime(year, m, 1) for m in ds.indexes["month"]]),
            )
        )
        .swap_dims(month="time")
        .drop("month")
    )


def read_year(filename):
    """
    Read a GFED4 datafile for a specific year.

    Read data from a file for a specific year, processes monthly emissions data, and
    returns a merged dataset.

    Parameters
    ----------
       filename: The file (assumed to be HDF5 format) to read.

    Returns
    -------
       xr.Dataset: A dataset containing the emissions data for the given year, with
       time coordinates in CFTimeIndex format.

    Workflow:
       - Opens the file, reads the coordinates, reads the emissions data, and converts the
       month dimension to cftime with the given year.
    """
    file = h5py.File(filename)
    coords = read_coords(file)
    parts = [
        # read_monthly(file["biosphere"], coords),
        # read_monthly(file["burned_area"], coords),
        read_monthly(file["emissions"], coords)
    ]

    year = int(Path(filename).stem.split("_")[1])
    ds = month_to_cftime(xr.merge(parts), year)

    return ds


def read_cell_area(filename):
    """
    Read the grid cell area data from an HDF5 file.

    Read the grid cell area data from a file, converts it into an xarray DataArray, and assigns
    the unit attribute for the area.

    Parameters
    ----------
        filename: The file (assumed to be HDF5 format) containing the grid cell area data.

    Returns
    -------
        xr.DataArray: The grid cell area data as an xarray DataArray with coordinates, and
        an assigned unit of "m2" (square meters).

    Workflow:
        - Opens the file and reads the latitude and longitude coordinates using the read_coords function.
        - Reads the "grid_cell_area" dataset from the "ancill" group in the file.
        - Converts the dataset into an xarray DataArray with the specified coordinates.
        - Assigns the unit attribute to be "m2" (square meters).
    """
    file = h5py.File(filename)
    coords = read_coords(file)
    return read_var(file["ancill"]["grid_cell_area"], coords).assign_attrs(unit="m2")


def add_global(df, groups=["em", "unit", "sector"]):
    """
    Add a "Global" or "World" aggregate row to the DataFrame by summing over all countries.

    Parameters
    ----------
        df: A pandas DataFrame containing emissions data, indexed by "em" (emission type),
            "unit", "sector", and other dimensions such as "country".

    Returns
    -------
        pd.DataFrame: The original DataFrame with an additional row representing global
        emissions for each emission type, unit, and sector.

    Workflow:
        - Groups the DataFrame by "em", "unit", and "sector" columns, ignoring the "country" dimension.
        - Sums the emissions across all countries for each group.
        - Assigns the "country" value as "World" to represent global totals.
        - Adds these new global rows to the original DataFrame using `pd.concat`.
        - Returns the resulting DataFrame with the added "World" aggregate rows.
    """
    return pd.concat([df, df.groupby(groups).sum().idx.assign(country="World", order=df.index.names)])
