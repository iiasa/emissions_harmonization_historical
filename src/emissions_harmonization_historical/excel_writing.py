"""
Tools for helping with excel writing
"""

from __future__ import annotations

import openpyxl.styles.fonts
import openpyxl.worksheet.worksheet


def set_cell(
    value: str,
    row: int,
    col: int,
    ws: openpyxl.worksheet.worksheet.Worksheet,
    font: openpyxl.styles.fonts.Font | None = None,
) -> None:
    """
    Set a cell's value

    Parameters
    ----------
    value
        Value to set

    row
        Row to set (uses python indexing i.e. 0-indexing)

    col
        Column to set (uses python indexing i.e. 0-indexing)

    ws
        Worksheet to write in

    font
        Font to apply to the value
    """
    cell = ws.cell(row=row + 1, column=col + 1)
    cell.value = value
    if font is not None:
        cell.font = font
