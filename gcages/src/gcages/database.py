"""
Local database implementation

Super simple, mainly designed as a caching helper for simple climate model runs
"""

from __future__ import annotations

import contextlib
import os
from contextlib import nullcontext
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import filelock
import numpy as np
import pandas as pd
import pandas_indexing as pix
from attrs import define

from gcages.pandas_helpers import multi_index_lookup, multi_index_match

if TYPE_CHECKING:
    import xarray as xr


class AlreadyInDBError(ValueError):
    """
    Raised when saving data would overwrite data which is already in the database
    """

    def __init__(self, already_in_db: pd.DataFrame) -> None:
        """
        Initialise the error

        Parameters
        ----------
        already_in_db
            data that is already in the database
        """
        error_msg = (
            "The following rows are already in the database:\n"
            f"{already_in_db.index.to_frame(index=False)}"
        )
        super().__init__(error_msg)


class EmptyDBError(ValueError):
    """
    Raised when trying to access data from a database that is empty
    """

    def __init__(self, db: GCDB) -> None:
        """
        Initialise the error

        Parameters
        ----------
        db
            The database
        """
        error_msg = f"The database is empty: {db=}"
        super().__init__(error_msg)


class GCDBDataFormat(StrEnum):
    """Options for the data format to use with `GCDB`"""

    CSV = "csv"
    Feather = "feather"
    netCDF = "nc"
    # Other options to consider:
    #
    # HDF5: https://pandas.pydata.org/docs/user_guide/io.html#hdf5-pytables
    # HDF5 = auto()


@define
class GCDB:
    """
    A database for gcages (GCDB)
    """

    db_dir: Path
    """
    Path in which the database is stored

    Both the index and the data files will be written in this directory.
    """

    format: GCDBDataFormat = GCDBDataFormat.Feather
    """
    Format to use for saving the data
    """

    @property
    def index_file(self) -> Path:
        """
        The file in which the database's index is stored

        Returns
        -------
        :
            Path to the index file
        """
        if self.format == GCDBDataFormat.CSV:
            return self.db_dir / "index.csv"

        if self.format == GCDBDataFormat.Feather:
            return self.db_dir / "index.feather"

        if self.format == GCDBDataFormat.netCDF:
            return self.db_dir / "index.nc"

        raise NotImplementedError(self.format)

    @property
    def file_map_file(self) -> Path:
        if self.format == GCDBDataFormat.CSV:
            return self.db_dir / "filemap.csv"

        if self.format == GCDBDataFormat.Feather:
            return self.db_dir / "filemap.feather"

        if self.format == GCDBDataFormat.netCDF:
            return self.db_dir / "filemap.nc"

        raise NotImplementedError(self.format)

    @property
    def index_file_lock_path(self) -> Path:
        return f"{self.index_file}.lock"

    @property
    def index_file_lock(self) -> Path:
        return filelock.FileLock(self.index_file_lock_path)

    def delete(
        self,
        *,
        progress: bool = False,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> None:
        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            if self.format == GCDBDataFormat.CSV:
                to_remove = self.db_dir.glob("*.csv")
            elif self.format == GCDBDataFormat.Feather:
                to_remove = self.db_dir.glob("*.feather")
            elif self.format == GCDBDataFormat.netCDF:
                to_remove = netCDFBackend.get_all_db_files(self.db_dir)
            else:
                raise NotImplementedError(self.format)

            if progress:
                import tqdm.auto as tqdm

                to_remove = tqdm.auto.tqdm(to_remove, desc="DB files")

            for f in to_remove:
                os.remove(f)

    def get_new_data_file_path(self, file_id: int) -> Path:
        """
        Get the path in which to write a new data file

        Parameters
        ----------
        file_id
            ID to associate with the file

        Returns
        -------
        :
            File in which to write the new data

        Raises
        ------
        FileExistsError
            A file already exists for the given `file_id`
        """
        if self.format == GCDBDataFormat.CSV:
            file_path = self.db_dir / f"{file_id}.csv"

        elif self.format == GCDBDataFormat.Feather:
            file_path = self.db_dir / f"{file_id}.feather"

        elif self.format == GCDBDataFormat.netCDF:
            file_path = self.db_dir / f"{file_id}.nc"

        else:
            raise NotImplementedError(self.format)

        if file_path.exists():
            raise FileExistsError(file_path)

        return file_path

    def load(
        self,
        selector: pd.Index | pd.MultiIndex | pix.selectors.Selector | None = None,
        *,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
        progress: bool = False,
        out_columns_type: type | None = None,
    ) -> pd.DataFrame:
        if not self.index_file.exists():
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        def idx_obj(inobj):
            if selector is None:
                res = inobj

            elif isinstance(selector, pd.MultiIndex):
                res = multi_index_lookup(inobj, selector)

            elif isinstance(selector, pd.Index):
                res = inobj[inobj.index.isin(selector.values, level=selector.name)]

            else:
                res = inobj.loc[selector]

            return res

        with lock_context_manager:
            if self.format == GCDBDataFormat.CSV:
                index_raw = pd.read_csv(self.index_file)
            elif self.format == GCDBDataFormat.Feather:
                index_raw = pd.read_feather(self.index_file)
            elif self.format == GCDBDataFormat.netCDF:
                index_raw = netCDFBackend.load_index(self.index_file)
            else:
                raise NotImplementedError(self.format)

            # Don't need to copy as index_raw is only used internally.
            # The different name is just to help understand the order of operations.
            index = index_raw
            index.index = pd.MultiIndex.from_frame(index_raw)
            file_map = self.load_file_map(lock_context_manager=nullcontext())

            index_to_load = idx_obj(index)
            files_to_load = file_map[index_to_load["file_id"].unique()].map(Path)

            if progress:
                import tqdm.auto as tqdm

                files_to_load = tqdm.tqdm(files_to_load, desc="Files to load")

            if self.format == GCDBDataFormat.CSV:
                data_l = [pd.read_csv(f) for f in files_to_load]
            elif self.format == GCDBDataFormat.Feather:
                data_l = [pd.read_feather(f) for f in files_to_load]
            elif self.format == GCDBDataFormat.netCDF:
                data_l = [netCDFBackend.load_data(f) for f in files_to_load]
            else:
                raise NotImplementedError(self.format)

        loaded = pix.concat(data_l).set_index(index.index.droplevel("file_id").names)

        res = idx_obj(loaded)
        if out_columns_type is not None:
            res.columns = res.columns.astype(out_columns_type)

        return res

    def load_file_map(
        self,
        *,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> pd.MultiIndex:
        if not self.index_file.exists():
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            if self.format == GCDBDataFormat.CSV:
                file_map = pd.read_csv(self.file_map_file, index_col="file_id")[
                    "file_path"
                ]
            elif self.format == GCDBDataFormat.Feather:
                file_map_raw = pd.read_feather(self.file_map_file)
                file_map = file_map_raw.set_index("file_id")["file_path"]
            elif self.format == GCDBDataFormat.netCDF:
                file_map = netCDFBackend.load_file_map(self.file_map_file)
            else:
                raise NotImplementedError(self.format)

        return file_map

    def load_metadata(
        self,
        *,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> pd.MultiIndex:
        if not self.index_file.exists():
            raise EmptyDBError(self)

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            if self.format == GCDBDataFormat.CSV:
                db_index = pd.read_csv(self.index_file)
            elif self.format == GCDBDataFormat.Feather:
                db_index = pd.read_feather(self.index_file)
            elif self.format == GCDBDataFormat.netCDF:
                db_index = netCDFBackend.load_index(self.index_file)
            else:
                raise NotImplementedError(self.format)

        res = pd.MultiIndex.from_frame(db_index).droplevel("file_id")
        return res

    def regroup(
        self,
        new_groups: list[str],
        *,
        progress: bool = False,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> None:
        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            all_dat = self.load(progress=progress, lock_context_manager=nullcontext())

            self.delete(lock_context_manager=nullcontext())

            grouper = all_dat.groupby(new_groups)
            if progress:
                import tqdm.auto as tqdm

                grouper = tqdm.tqdm(grouper)

            for _, df in grouper:
                self.save(df, allow_overwrite=True, lock_context_manager=nullcontext())

    def save(
        self,
        data: pd.DataFrame,
        *,
        allow_overwrite: bool = False,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> None:
        """
        Save data into the database

        This saves all of the data in a single file.
        If you want to put the data into different files,
        group `data` before calling [save][(c)].

        Parameters
        ----------
        data
            Data to add to the database

        allow_overwrite
            Should overwrites of data that is already in the database be allowed?

        lock_context_manager
            Context manager to use when obtaining the database's lock file.

            If not supplied, we use `self.index_file_lock.acquire()`.
        """
        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            if self.index_file.exists():
                if self.format == GCDBDataFormat.CSV:
                    index_db = pd.read_csv(self.index_file)
                elif self.format == GCDBDataFormat.Feather:
                    index_db = pd.read_feather(self.index_file)
                elif self.format == GCDBDataFormat.netCDF:
                    index_db = netCDFBackend.load_index(self.index_file)
                else:
                    raise NotImplementedError(self.format)

                metadata = pd.MultiIndex.from_frame(index_db).droplevel("file_id")
                file_map = self.load_file_map(lock_context_manager=nullcontext())

                already_in_db = multi_index_lookup(data, metadata)
                if not already_in_db.empty and not allow_overwrite:
                    raise AlreadyInDBError(already_in_db=already_in_db)

                file_id = index_db["file_id"].max() + 1
                data_file_path = self.get_new_data_file_path(file_id=file_id)
                index_data = data.index.to_frame(index=False)
                index_data["file_id"] = file_id
                file_map[file_id] = data_file_path

                if already_in_db.empty:
                    # No clashes, so we can simply join
                    index = pd.concat([index_db, index_data])
                else:
                    if not allow_overwrite:  # pragma: no cover
                        msg = "Should have already raised above"
                        raise AssertionError(msg)

                    index_db_keep, file_map = _update_index_for_overwrite(
                        db=self,
                        file_ids_existing=index_db.set_index(metadata.names)["file_id"],
                        data_to_write=data,
                        already_in_db=already_in_db,
                        file_map=file_map,
                    )

                    index = pd.concat([index_db_keep, index_data])

            else:
                # index file doesn't exist i.e. we're starting from nothing
                file_id = 0
                data_file_path = self.get_new_data_file_path(file_id=file_id)
                index = data.index.to_frame(index=False)
                index["file_id"] = file_id
                file_map = pd.Series({file_id: data_file_path}, name="file_path")
                file_map.index.name = "file_id"

            if self.format == GCDBDataFormat.CSV:
                backend = CSVBackend

            elif self.format == GCDBDataFormat.Feather:
                backend = FeatherBackend

            elif self.format == GCDBDataFormat.netCDF:
                backend = netCDFBackend

            else:
                raise NotImplementedError(self.format)

            backend._save_to_database(
                index=index,
                index_file=self.index_file,
                file_map=file_map,
                file_map_file=self.file_map_file,
                data=data,
                data_file_path=data_file_path,
            )


def _update_index_for_overwrite(
    db: GCDB,
    file_ids_existing: pd.Series[int],
    data_to_write: pd.DataFrame,
    already_in_db: pd.DataFrame,
    file_map: pd.Series[Path],
) -> tuple[pd.MultiIndex, pd.Series[Path]]:
    remove_loc = multi_index_match(file_ids_existing, data_to_write.index)

    file_ids_remove = set(file_ids_existing[remove_loc])
    file_ids_keep = set(file_ids_existing[~remove_loc])
    file_ids_overlap = file_ids_remove.intersection(file_ids_keep)

    file_map_out = file_map.copy()

    if not file_ids_overlap:
        # Nice and simple, just remove the old files
        index_out = file_ids_existing[~remove_loc].reset_index()
        for rfid in file_ids_remove:
            os.remove(file_map_out.pop(rfid))

    else:
        # More complicated: for some files,
        # some of the data needs to be removed
        # while other parts need to be kept.
        # Hence we have to re-write that data into files
        # that are separate from the data we want to keep
        # before we can continue.
        file_ids_out = file_ids_existing.copy()
        for ofid in file_ids_overlap:
            file = file_map_out.pop(ofid)

            if db.format == GCDBDataFormat.CSV:
                overlap_file_data_raw = pd.read_csv(file)
            elif db.format == GCDBDataFormat.Feather:
                overlap_file_data_raw = pd.read_feather(file)
            elif db.format == GCDBDataFormat.netCDF:
                overlap_file_data_raw = netCDFBackend.load_data(file)
            else:
                raise NotImplementedError(db.format)

            overlap_file_data = overlap_file_data_raw.set_index(
                file_ids_existing.index.names
            )

            data_not_being_overwritten = overlap_file_data.loc[
                ~multi_index_match(overlap_file_data, data_to_write.index)
            ]

            # Ensure we use a file ID we haven't already used
            data_not_being_overwritten_file_id = (
                max(file_map_out.index.max(), file_map.index.max()) + 1
            )
            data_not_being_overwritten_file_path = db.get_new_data_file_path(
                file_id=data_not_being_overwritten_file_id
            )

            # Re-write the data we want to keep
            if db.format == GCDBDataFormat.CSV:
                data_not_being_overwritten.to_csv(data_not_being_overwritten_file_path)
            elif db.format == GCDBDataFormat.Feather:
                # Feather doesn't support writing indexes,
                # see https://pandas.pydata.org/docs/user_guide/io.html#feather.
                data_to_write = data_not_being_overwritten.reset_index()
                data_to_write.to_feather(data_not_being_overwritten_file_path)
            elif db.format == GCDBDataFormat.netCDF:
                netCDFBackend.save_data(
                    data_not_being_overwritten, data_not_being_overwritten_file_path
                )
            else:
                raise NotImplementedError(db.format)

            # Update the file map (already popped the old file above)
            file_map_out[data_not_being_overwritten_file_id] = (
                data_not_being_overwritten_file_path
            )

            # Update the file ids of the data we're keeping
            data_not_being_overwritten_idx = multi_index_match(
                file_ids_out, data_not_being_overwritten.index
            )
            file_ids_out.loc[data_not_being_overwritten_idx] = (
                data_not_being_overwritten_file_id
            )
            # Remove the rows that still refer to the data we're dropping
            file_ids_out = file_ids_out.loc[file_ids_out != ofid]
            # Remove the file that contained the data
            os.remove(file)

        index_out = file_ids_out.reset_index()

    return index_out, file_map_out


class CSVBackend:
    """CSV-based backend"""

    @staticmethod
    def _save_to_database(  # noqa: PLR0913
        index: pd.DataFrame,
        index_file: Path,
        file_map: pd.Series[Path],
        file_map_file: Path,
        data: pd.DataFrame,
        data_file_path: Path,
    ) -> None:
        index.to_csv(index_file, index=False)
        file_map.to_csv(file_map_file)
        data.to_csv(data_file_path)


class FeatherBackend:
    """Feather-based backend"""

    @staticmethod
    def _save_to_database(  # noqa: PLR0913
        index: pd.DataFrame,
        index_file: Path,
        file_map: pd.Series[Path],
        file_map_file: Path,
        data: pd.DataFrame,
        data_file_path: Path,
    ) -> None:
        index.to_feather(index_file)

        # Feather doesn't support
        # (see https://pandas.pydata.org/docs/user_guide/io.html#feather):
        # - writing indexes
        file_map_write = file_map.reset_index()
        # - writing non-native types (e.g. Path)
        file_map_write["file_path"] = file_map_write["file_path"].astype(str)
        file_map_write.to_feather(file_map_file)

        # - writing indexes
        data_write = data.reset_index()
        # - mixed column types
        data_write.columns = data_write.columns.astype(str)
        data_write.to_feather(data_file_path)


class netCDFBackend:
    """netCDF-based backend"""

    # All the categorical stuff here is probably fine, because it's abstracted,
    # but it could cause issues
    # (although it's probably worth it for performance reasons).
    # See https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a/.

    @staticmethod
    def _save_to_database(  # noqa: PLR0913
        index: pd.DataFrame,
        index_file: Path,
        file_map: pd.Series[Path],
        file_map_file: Path,
        data: pd.DataFrame,
        # TODO: rename to data_file for consistency
        data_file_path: Path,
    ) -> None:
        # TODO: MissingOptionalDependencyError
        import xarray as xr

        # Index
        index_xr = _metadata_to_xr(index, row_name="ts_id")
        index_xr.to_netcdf(index_file)

        # File map
        file_map_xr = xr.DataArray.from_series(file_map.astype(str))
        file_map_xr.to_netcdf(file_map_file)

        # Data
        netCDFBackend.save_data(data, data_file_path)

    @staticmethod
    def get_all_db_files(db_dir: Path) -> None:
        return db_dir.glob("*.nc")

    @staticmethod
    def load_data(data_file: Path) -> pd.MultiIndex:
        import xarray as xr

        raw = xr.load_dataset(data_file)

        data = raw["values"].to_pandas()
        index = netCDFBackend.metadata_idx_to_index(raw)

        res = pd.concat([index.loc[raw["ts_idx"]], data], axis="columns")

        return res

    @staticmethod
    def load_file_map(file_map_file: Path) -> pd.MultiIndex:
        import xarray as xr

        return xr.load_dataarray(file_map_file).to_pandas().map(Path)

    @staticmethod
    def load_index(index_file: Path) -> pd.MultiIndex:
        import xarray as xr

        raw = xr.load_dataset(index_file)

        # As a nice illustration of xarray's sparsity,
        # if you do the below, you blow your memory.
        # raw.to_dataframe()

        index = netCDFBackend.metadata_idx_to_index(raw)

        return index

    @staticmethod
    def metadata_idx_to_index(raw: xr.Dataset) -> pd.DataFrame:
        index = raw["metadata_idx"].to_pandas().astype("category")
        index.columns = index.columns.map(raw["metadata_column"].to_pandas())
        for column in index:
            index[column] = index[column].map(raw[column].to_pandas())

        index.columns.name = None
        index.index.name = None
        # TODO: think through the category handling more
        for c in index:
            index[c] = index[c].astype(index[c].dtype.categories.dtype)
        # TODO: think through the type handling more
        for c in index:
            if isinstance(index[c].values[0], np.int32):
                index[c] = index[c].astype(np.int64)

        return index

    @staticmethod
    def save_data(data: pd.DataFrame, data_file: Path) -> None:
        import xarray as xr

        data_rs = data.reset_index()
        data_index_xr = _metadata_to_xr(data_rs[data.index.names], row_name="ts_idx")
        data_values_xr = xr.DataArray(
            data_rs[data.columns],
            dims=["ts_idx", "time"],
            coords=dict(ts_idx=data_rs.index.values, time=data.columns),
        )
        data_xr = xr.merge([data_index_xr, data_values_xr.to_dataset(name="values")])
        data_xr.to_netcdf(data_file)


def _metadata_to_xr(
    start: pd.DataFrame, row_name: str, convert_to_category: bool = True
) -> xr.Dataset:
    import xarray as xr

    if convert_to_category:
        start = start.astype("category")

    # Assumes timeseries are in rows
    ts_ids = np.arange(start.shape[0])
    metadata_columns = start.columns.tolist()
    index_darrays = {}
    for metadata_col in metadata_columns:
        metadata_col_unique_vals = start[metadata_col].values.categories
        metadata_col_map = {}
        metadata_col_mapped_values = [""] * len(metadata_col_unique_vals)
        for i, v in enumerate(metadata_col_unique_vals):
            metadata_col_map[v] = i
            metadata_col_mapped_values[i] = i

        metadata_col_id = f"{metadata_col}_id"
        metadata_xr = xr.DataArray(
            metadata_col_unique_vals,
            dims=[metadata_col_id],
            coords={metadata_col_id: metadata_col_mapped_values},
        )

        index_darrays[metadata_col] = metadata_xr
        start[metadata_col] = start[metadata_col].map(metadata_col_map)

    metadata_column_idx_coord = dict(metadata_column_idx=range(len(metadata_columns)))
    index_darrays["metadata_column"] = xr.DataArray(
        metadata_columns,
        dims=["metadata_column_idx"],
        coords=metadata_column_idx_coord,
    )
    index_darrays["metadata_idx"] = xr.DataArray(
        start.values,
        dims=[row_name, "metadata_column_idx"],
        coords={row_name: ts_ids} | metadata_column_idx_coord,
    )

    index_xr = xr.Dataset(index_darrays)

    return index_xr
