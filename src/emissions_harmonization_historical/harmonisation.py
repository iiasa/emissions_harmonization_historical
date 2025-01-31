"""
Harmonisation configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import pandas as pd

HARMONISATION_YEAR = 2021

HARMONISATION_YEAR_MISSING_SCALING_YEAR = 2015
"""
Year to scale if the harmonisation year is missing from a submission
"""

# As at 2024-01-30, just the list from AR6.
# We can tweak from here.
aneris_overrides = pd.DataFrame(
    [
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|BC'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C2F6",
        },  # high historical variance (cov=16.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|C6F14",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|PFC|CF4",
        },  # high historical variance (cov=11.2)
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|CO",
        },  # high historical variance (cov=15.4)
        {
            "method": "reduce_ratio_2080",
            "variable": "Emissions|CO2",
        },  # always ratio method by choice
        {
            # high historical variance,
            # but using offset method to prevent diff from increasing
            # when going negative rapidly (cov=23.2)
            "method": "reduce_offset_2150_cov",
            "variable": "Emissions|CO2|AFOLU",
        },
        {
            "method": "reduce_ratio_2080",  # always ratio method by choice
            "variable": "Emissions|CO2|Energy and Industrial Processes",
        },
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|CH4'},
        {
            "method": "constant_ratio",
            "variable": "Emissions|F-Gases",
        },  # basket not used in infilling (sum of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC",
        },  # basket not used in infilling (sum of subset of f-gases with low model reporting confidence)
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC125",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC134a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC143a",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC227ea",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC23",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC32",
        },  # minor f-gas with low model reporting confidence
        {
            "method": "constant_ratio",
            "variable": "Emissions|HFC|HFC43-10",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|N2O'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NH3'},
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|NOx'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|OC",
        },  # high historical variance (cov=18.5)
        {
            "method": "constant_ratio",
            "variable": "Emissions|SF6",
        },  # minor f-gas with low model reporting confidence
        # depending on the decision tree in aneris/method.py
        #     {'method': 'default_aneris_tree', 'variable': 'Emissions|Sulfur'},
        {
            "method": "reduce_ratio_2150_cov",
            "variable": "Emissions|VOC",
        },  # high historical variance (cov=12.0)
    ]
)
"""
Overrides to pass to aneris
"""
