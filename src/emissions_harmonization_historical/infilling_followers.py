"""
Definition of the variables to follow for the infilling that needs it
"""

from __future__ import annotations

# TODO: double check with an expert
# whether these choices for 'co-production' make sense.
FOLLOW_LEADERS: dict[str, str] = {
    "Emissions|C3F8": "Emissions|C2F6",
    "Emissions|C4F10": "Emissions|C2F6",
    "Emissions|C5F12": "Emissions|C2F6",
    "Emissions|C7F16": "Emissions|C2F6",
    "Emissions|C8F18": "Emissions|C2F6",
    "Emissions|cC4F8": "Emissions|CF4",
    "Emissions|SO2F2": "Emissions|CF4",
    "Emissions|HFC|HFC236fa": "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC152a": "Emissions|HFC|HFC43-10",
    "Emissions|HFC|HFC365mfc": "Emissions|HFC|HFC134a",
    "Emissions|Montreal Gases|CH2Cl2": "Emissions|HFC|HFC134a",
    "Emissions|Montreal Gases|CHCl3": "Emissions|C2F6",
    "Emissions|Montreal Gases|CH3Br": "Emissions|C2F6",
    "Emissions|Montreal Gases|CH3Cl": "Emissions|CF4",
    "Emissions|NF3": "Emissions|SF6",
}
