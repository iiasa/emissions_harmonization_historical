"""Here we specify functions related to processing CEDS (Community Emissions Data System) data."""

from pathlib import Path

import pandas as pd


def get_map(mapping, sector_column, sector_output_column_name="sector_59"):
    """
    Create a MultiIndex mapping between a given sector column and the harmonized sectors.

    This function takes a DataFrame (`mapping`) and creates a pandas `MultiIndex` from
    the `sector_column` and the "Harmonization Sectors" column. It drops duplicate rows,
    renames the levels of the index to `sector_59` and `sector`, and removes any null
    values from the index.

    Parameters
    ----------
    mapping : pd.DataFrame
        A DataFrame containing sector data, including a column for the sectors that need
        to be mapped and a column for the harmonized sectors.

    sector_column : str
        The name of the column in `mapping` that contains the sectors you want to map to
        the harmonized sectors.

    Returns
    -------
    pd.MultiIndex
        A pandas `MultiIndex` where the first level is the values from `sector_column`,
        and the second level is the corresponding harmonized sectors. Any null values in
        the index are dropped.
    """
    return (
        pd.MultiIndex.from_frame(mapping[[sector_column, "Harmonization Sectors"]].drop_duplicates())
        .rename([sector_output_column_name, "sector"])
        .idx.dropna()
    )


def read_CEDS(path: Path, num_index=4, sector_output_column_name="sector_59"):
    """
    Read a CEDS (Community Emissions Data System) CSV file into a pandas DataFrame.

    This function reads a CEDS dataset from a CSV file located at `path` and processes it
    into a pandas DataFrame. The file is read with `num_index` columns as the index, and
    the column names are transformed by converting them into integers (removing any leading
    characters, e.g., "X2010" becomes 2010). The index levels "country" and "sector" are
    renamed to "region" and "sector_59", respectively.

    Parameters
    ----------
    path : Path
        The path to the CSV file containing the CEDS data.

    num_index : int, optional (default=4)
        The number of columns from the CSV file to be used as the index for the DataFrame.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the processed CEDS data. The index will include `num_index`
        columns, with "country" renamed to "region" and "sector" renamed to "sector_59".
        The columns will represent years and will be renamed as integers.
    """
    df = (
        pd.read_csv(path, index_col=list(range(num_index)), engine="pyarrow")
        .rename(columns=lambda s: int(s[1:]))
        .rename_axis(index={"country": "region", "sector": sector_output_column_name})
    )
    return df