"""
emissions_harmonization_historical.extensions

=============================================

This package contains helper functions and workflow components used to construct
historical+future continuous emissions timeseries for non-CO2 species, CO2
component splitting, AFOLU handling, and general extension utilities.

Modules
-------
- :mod:`extension_functionality` -- Low-level mathematical helpers used to
        construct extension profiles (sigmoids, decays, splines, and ramps).
- :mod:`general_utils_for_extensions` -- Dataframe utilities for interpolation,
        harmonising scenario + historical time series, index helpers and CSV dumping.
- :mod:`extensions_functions_for_non_co2` -- High-level routines to extend
        non-CO2 species regionally and to combine per-sector extensions into totals.
- :mod:`finish_regional_extensions` -- Helpers to finalize and stitch regional
        extensions and to save/serialize outputs.
- :mod:`fossil_co2_storyline_functions` -- Storyline-driven CO2 (fossil/AFOLU)
        splitting and extension utilities (CS, ECS, CSCS storylines, etc.).
- :mod:`cdr_and_fossil_splits` -- Utilities for splitting and extending CDR and
        fossil CO2 components, and for constructing gross removal/positive emission
        variables in historical time series.
- :mod:`afolu_extension_functions` -- AFOLU-specific extension helpers (cumulative
        AFOLU computation and filling from historical records).

Usage
-----
Import the package and access submodules::

                from emissions_harmonization_historical import extensions
                extensions.general_utils_for_extensions.interpolate_to_annual(...)

For convenience, common submodules are exported at package level below.
"""

from . import (
    afolu_extension_functions,
    cdr_and_fossil_splits,
    extension_functionality,
    extensions_functions_for_non_co2,
    finish_regional_extensions,
    fossil_co2_storyline_functions,
    general_utils_for_extensions,
)

__all__ = [
    "extension_functionality",
    "general_utils_for_extensions",
    "extensions_functions_for_non_co2",
    "finish_regional_extensions",
    "fossil_co2_storyline_functions",
    "cdr_and_fossil_splits",
    "afolu_extension_functions",
]
