"""
Local database implementation

Super simple, mainly designed as a caching helper for simple climate model runs
"""

from __future__ import annotations

import contextlib
import os
from contextlib import nullcontext
from enum import StrEnum, auto
from pathlib import Path

import filelock
import pandas as pd
import pandas_indexing as pix
from attrs import define

from gcages.pandas_helpers import multi_index_lookup, multi_index_match


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

    CSV = auto()
    Feather = auto()


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
        if self.format == GCDBDataFormat.CSV:
            return self.db_dir / "index.csv"

        raise NotImplementedError(self.format)

    @property
    def file_map_file(self) -> Path:
        if self.format == GCDBDataFormat.CSV:
            return self.db_dir / "filemap.csv"

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
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> None:
        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            for f in self.db_dir.glob("*.csv"):
                os.remove(f)

    def get_new_data_file_path(self, file_id: int) -> Path:
        file_path = self.db_dir / f"{file_id}.csv"
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
            index = pd.MultiIndex.from_frame(pd.read_csv(self.index_file)).to_frame()
            file_map = pd.read_csv(self.file_map_file, index_col="file_id")["file_path"]

            index_to_load = idx_obj(index)
            files_to_load = file_map[index_to_load["file_id"].unique()]

            if progress:
                import tqdm.autonotebook as tqdman

                files_to_load = tqdman.tqdm(files_to_load)

            data_l = [pd.read_csv(f) for f in files_to_load]

        loaded = pix.concat(data_l).set_index(index.index.droplevel("file_id").names)

        res = idx_obj(loaded)
        if out_columns_type is not None:
            res.columns = res.columns.astype(out_columns_type)

        return res

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
            db_index = pd.read_csv(self.index_file)

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
                import tqdm.autonotebook as tqdman

                grouper = tqdman.tqdm(grouper)

            for _, df in grouper:
                self.save(df, allow_overwrite=True, lock_context_manager=nullcontext())

    def save(
        self,
        data: pd.DataFrame,
        *,
        allow_overwrite: bool = False,
        lock_context_manager: contextlib.AbstractContextManager | None = None,
    ) -> None:
        # Save entire frame into single file.
        # If user wants things broken up,
        # they should do that before calling `save`.

        if lock_context_manager is None:
            lock_context_manager = self.index_file_lock.acquire()

        with lock_context_manager:
            if self.index_file.exists():
                index_db = pd.read_csv(self.index_file)
                metadata = pd.MultiIndex.from_frame(index_db).droplevel("file_id")
                file_map = pd.read_csv(self.file_map_file, index_col="file_id")[
                    "file_path"
                ]

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

            index.to_csv(self.index_file, index=False)
            file_map.to_csv(self.file_map_file)
            data.to_csv(data_file_path)


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
            overlap_file_data = pd.read_csv(file).set_index(
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
            data_not_being_overwritten.to_csv(data_not_being_overwritten_file_path)
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
