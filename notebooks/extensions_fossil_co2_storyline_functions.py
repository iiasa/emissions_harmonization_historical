import sys

import numpy as np

from emissions_harmonization_historical.extension_functionality import (
    make_linear_function_with_smooth_transition,
)


def extend_from_start_to_stop_with_value(value, start, stop, co2_fossil_extend, co2_total_extend):
    """
    Set total CO2 emissions to a constant value over a specified time period and adjust fossil emissions accordingly.

    This function modifies the total CO2 emissions array to have a constant value over a specified range,
    then adjusts the fossil CO2 emissions to maintain consistency (since total = fossil + AFOLU).

    Parameters
    ----------
    value : float or array-like
        The target total CO2 emissions value(s) to set for the specified period.
        Can be a scalar for constant emissions or an array for varying emissions.
    start : int
        Start index in the emissions arrays (inclusive).
    stop : int
        Stop index in the emissions arrays (exclusive).
    co2_fossil_extend : numpy.ndarray
        Array of fossil CO2 emissions to be modified in-place.
    co2_total_extend : numpy.ndarray
        Array of total CO2 emissions to be modified in-place.

    Returns
    -------
    co2_fossil_extend : numpy.ndarray
        Modified fossil CO2 emissions array.
    co2_total_extend : numpy.ndarray
        Modified total CO2 emissions array.

    Notes
    -----
    This function assumes that total CO2 = fossil CO2 + AFOLU CO2, and that the AFOLU
    component remains unchanged during the modification.
    """
    co2_fossil_extend[start:stop] = value - co2_total_extend[start:stop]
    co2_total_extend[start:stop] = value
    return co2_fossil_extend, co2_total_extend


def linear_to_target(target, from_val, start, stop, time):
    """
    Calculate linear interpolation between two values over a specified time period.

    This function computes a linear trajectory from an initial value to a target value
    over a specified time range, useful for creating transitions in emissions scenarios.

    Parameters
    ----------
    target : float
        The target value to reach at the end of the interpolation period.
    from_val : float
        The initial value at the start of the interpolation period.
    start : int or float
        The start time/year of the interpolation period.
    stop : int or float
        The end time/year of the interpolation period.
    time : int, float, or array-like
        The time point(s) at which to evaluate the linear function.
        Can be a single value or an array of time points.

    Returns
    -------
    float or numpy.ndarray
        The linearly interpolated value(s) at the specified time point(s).
        Returns a scalar if `time` is scalar, or an array if `time` is array-like.

    Examples
    --------
    >>> # Linear transition from 100 to 0 between years 2020 and 2050
    >>> linear_to_target(0, 100, 2020, 2050, 2035)
    50.0
    >>>
    >>> # Get values for multiple years
    >>> years = np.array([2020, 2030, 2040, 2050])
    >>> linear_to_target(0, 100, 2020, 2050, years)
    array([100.,  66.67,  33.33,   0.])
    """
    slope = (target - from_val) / (stop - start)
    return from_val + slope * (time - start)


def extend_co2_for_scen_storyline(df_extended_afolu, df_fossil, storyline, start=2023, end=2500):
    """
    Extend CO2 emissions time series for a given scenario storyline up to a specified end year.

    This function combines fossil and AFOLU (Agriculture, Forestry, and Other Land Use) CO2 emissions,
    and applies scenario-specific extensions (constant, linear, or sigmoid transitions) based on the
    provided storyline parameters. The extension logic supports multiple storyline types, including
    constant-then-sigmoid (CS), exponential-then-sigmoid (ECS), and double constant-sigmoid (CSCS).

    Parameters
    ----------
    df_extended_afolu : pandas.DataFrame
        DataFrame containing extended AFOLU CO2 emissions, indexed or columned by year.
    df_fossil : pandas.DataFrame
        DataFrame containing fossil CO2 emissions, indexed or columned by year.
    storyline : tuple
        Scenario storyline parameters. The first element specifies the storyline type ("CS", "ECS", "CSCS"),
        followed by integers specifying transition years and target values for the extension.
    start : int, optional
        Start year for the extension (default is 2023).
    end : int, optional
        End year for the extension (default is 2500).

    Returns
    -------
    co2_total_extend : numpy.ndarray
        Extended total CO2 emissions (fossil + AFOLU) for each year in the range [start, end].
    co2_fossil_extend : numpy.ndarray
        Extended fossil CO2 emissions for each year in the range [start, end].
    extended_years : numpy.ndarray
        Array of years corresponding to the extended emissions time series.

    Raises
    ------
    SystemExit
        If the provided storyline type is not recognized.

    Notes
    -----
    The function relies on helper functions such as `extend_from_start_to_stop_with_value` and
    `make_linear_function_with_smooth_transition` for constructing the extended emissions profiles.
    """
    # Implement the logic to extend CO2 emissions for the given scenario storyline
    print(storyline)
    extended_years = np.arange(start, 2501)
    co2_fossil_extend = np.zeros_like(extended_years)
    co2_fossil_extend[: 2101 - start] = df_fossil.loc[f"{start}" :, :].to_numpy().flatten()
    co2_total_extend = co2_fossil_extend + df_extended_afolu.loc[:, f"{start}" :].to_numpy().flatten()

    if storyline[0] == "CS":
        stop_const = storyline[1]
        end_sig = storyline[2]
        print("Doing constant then sigmoid evolution")
        co2_constant = co2_total_extend[2100 - start]
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            co2_constant, 2101 - start, stop_const - start + 1, co2_fossil_extend, co2_total_extend
        )
        linear_roll_extension = make_linear_function_with_smooth_transition(
            co2_total_extend[: stop_const + 1 - start],
            0,
            roll_start_length=20,
            roll_end_length=20,
            t_vals=np.arange(start, end_sig + 1),
            t_extend=stop_const - start,
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            linear_roll_extension, stop_const + 1 - start, end_sig + 1 - start, co2_fossil_extend, co2_total_extend
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            0, end_sig + 1 - start, end + 1 - start, co2_fossil_extend, co2_total_extend
        )
        # sigmoid_total_extension = sigmoid_function(
        # 0, co2_constant, stop_const +1, end_sig, np.arange(stop_const+1,end +1)
        # )
        # co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
        # sigmoid_total_extension,
        # stop_const +1 - start,
        # end +1 - start,
        # co2_fossil_extend,
        # co2_total_extend
        # )
    elif storyline[0] == "ECS":
        exp_end = storyline[1]
        exp_targ = storyline[2]
        sig_start = storyline[3]
        sig_end = storyline[4]
        # exp_decay_total = exp_decay(
        #     exp_targ,
        #     exp_targ - co2_total_extend[2100-start],
        #     exp_end-2100,
        #     time_0 = 2100,
        #     time = np.arange(2100, sig_start +1)
        #     )

        linear_decay_total = make_linear_function_with_smooth_transition(
            co2_total_extend[: 2101 - start], exp_targ, 20, 20, np.arange(start, exp_end + 1), 2100 - start
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            linear_decay_total, 2100 - start + 1, exp_end - start + 1, co2_fossil_extend, co2_total_extend
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            exp_targ, exp_end - start + 1, sig_start - start + 1, co2_fossil_extend, co2_total_extend
        )
        # linear_total_extension = linear_to_target(0, exp_targ, sig_start, sig_end, np.arange(sig_start, sig_end +1))
        linear_total_extension = make_linear_function_with_smooth_transition(
            co2_total_extend[: sig_start - start + 1], 0, 20, 20, np.arange(start, sig_end + 1), sig_start - start
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            linear_total_extension, sig_start - start + 1, sig_end + 1 - start, co2_fossil_extend, co2_total_extend
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            0, sig_end + 1 - start, end + 1 - start, co2_fossil_extend, co2_total_extend
        )
        # sigmoid_total_extension = sigmoid_function(0, exp_decay_total[-1],
        # sig_start, sig_end, extended_years[sig_start-start:])
        stop_const = storyline[1]
        sig_targ = storyline[2]
        end_sig1 = storyline[3]
        start_sig2 = storyline[4]
        end_sig2 = storyline[5]
        co2_constant = co2_total_extend[2100 - start]
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            co2_constant, 2101 - start, stop_const - start + 1, co2_fossil_extend, co2_total_extend
        )
        linear_total_extension = make_linear_function_with_smooth_transition(
            co2_total_extend[: stop_const + 1 - start],
            sig_targ,
            20,
            20,
            np.arange(start, end_sig1 + 1),
            stop_const - start,
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            linear_total_extension, stop_const + 1 - start, end_sig1 + 1 - start, co2_fossil_extend, co2_total_extend
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            sig_targ, end_sig1 + 1 - start, start_sig2 - start + 1, co2_fossil_extend, co2_total_extend
        )
        # linear_total_extension = linear_to_target(0, sig_targ, start_sig2, end_sig2, np.arange(start_sig2,end_sig2+1))
        linear_total_extension = make_linear_function_with_smooth_transition(
            co2_total_extend[: start_sig2 - start + 1], 0, 20, 20, np.arange(start, end_sig2 + 1), start_sig2 - start
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            linear_total_extension, start_sig2 - start + 1, end_sig2 + 1 - start, co2_fossil_extend, co2_total_extend
        )
        co2_fossil_extend, co2_total_extend = extend_from_start_to_stop_with_value(
            0, end_sig2 + 1 - start, end + 1 - start, co2_fossil_extend, co2_total_extend
        )
    else:
        print(f"Why am I here with {storyline}?")
        sys.exit(4)

    return co2_total_extend, co2_fossil_extend, extended_years
