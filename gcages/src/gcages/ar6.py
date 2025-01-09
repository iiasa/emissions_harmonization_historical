"""
Reproduction of the AR6 workflow
"""

from __future__ import annotations

AR6_RAW_VARIABLES: tuple[str, ...] = (
    "Emissions|CH4",
    "Emissions|N2O",
)
"""
Raw variables that were used in the AR6 workflow

Many variables were dropped before the workflow was entered.
For example, most sectoral detail.
"""


def get_ar6_harmoniser() -> None:
    """Docstring TBD"""
    raise NotImplementedError
