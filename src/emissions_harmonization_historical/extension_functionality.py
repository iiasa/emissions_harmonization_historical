import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def sigmoid_function(to_val, from_val, tstart, tend, z, adjust_from=True):  # noqa: PLR0913
    """
    Calculate sigmoid function

    Calculate sigmoid function from from_val to to_val
    between tstart and tend for over t-values z
    """
    z_shift = (z - (tstart + tend) / 2.0) * 5 / 2.0 / (tend - tstart)
    if not adjust_from:
        from_inf = from_val
    else:
        from_inf = (from_val - to_val / (1.0 + np.exp(-z_shift[0]))) / (1 - 1 / (1.0 + np.exp(-z_shift[0])))
    scaling = to_val - from_inf
    sigmoid = (1.0 / (1.0 + np.exp(-z_shift))) * scaling + from_inf

    return sigmoid


def get_sigmoid_derivative(to_val, from_val, sigmoid_shift, sigmoid_len=50):
    """
    Calculate the derivative of the sigmoid function
    """
    scaling = to_val - from_val
    t_eval = sigmoid_shift * 5 / 2 / sigmoid_len
    derivative = scaling * (np.exp(-t_eval) / (1 + np.exp(t_eval))) * 5 / 2 / sigmoid_len
    return derivative


def exp_decay(end, scale, decay_time, time_0, time):
    """
    Calculate exponential decay

    Calculate exponential decay to end value, with scale, decay_time,
    time_0 and over the values of time
    """
    # print(f"{time_0:}, {time:}, {decay_time}")
    return end - scale * np.exp((time_0 - time) / decay_time)


def solve_LU_constants(cle_0, lu_0, dlu_0, cle_inf_0=False, min_tau=20):
    """
    Solve for Land use exponential decay values
    """
    if cle_inf_0:
        cle_inf = 0
        k_mult = -cle_0
    else:
        k_mult = -lu_0 / dlu_0
        cle_inf = cle_0 + k_mult
    tau = k_mult / lu_0
    print(f"direct tau value: {tau}")
    if tau < min_tau:
        tau = min_tau
    elif tau > min_tau * 5:
        tau = min_tau * 5
    k_mult = tau * lu_0
    cle_inf = cle_0 + k_mult
    return cle_inf, k_mult, tau


def find_func_form_lu_extension(function, cle_func, t_vals, t_extend, cle_inf_0=False):
    """
    Calculate extended land use
    """
    lu_0 = function[t_extend]
    cle_0 = cle_func[t_extend]
    if cle_inf_0:
        cle_inf, k_mult, tau = solve_LU_constants(cle_0, lu_0, 1, cle_inf_0=cle_inf_0)
    else:
        dlu_0 = get_derivative_using_spline(function, t_vals, t_extend)
        print(dlu_0)
        cle_inf, k_mult, tau = solve_LU_constants(cle_0, lu_0, dlu_0)
    ext_func = np.zeros(len(t_vals))
    ext_func[:t_extend] = function[:t_extend]
    print(f"Value of tau is {tau}")
    ext_func[t_extend:] = k_mult / tau * np.exp((t_vals[t_extend] - t_vals[t_extend:]) / tau)
    return ext_func, cle_inf


def extend_flat_evolution(function, t_vals):
    """
    Extend function flatly
    """
    data_extend = np.zeros_like(t_vals)
    data_extend[: len(function)] = function
    data_extend[len(function) :] = function[-1]
    return data_extend


def extend_flat_cumulative(function, t_vals):
    """
    Extend cumulative function so cumulative stays the same
    """
    data_extend = np.zeros_like(t_vals)
    data_extend[: len(function)] = function
    return data_extend


def extend_linear_rampdown(function, t_vals, rampdown_end=2150):
    """
    Extend cumulative function so cumulative stays the same
    """
    data_extend = np.zeros_like(t_vals)
    data_extend[: len(function)] = function
    data_extend[len(function) : rampdown_end - t_vals[0]] = np.linspace(
        function[-1], 0, rampdown_end - t_vals[len(function)]
    )
    return data_extend


def make_cubic_coefficients_from_end_points_and_derivatives(  # noqa: PLR0913
    x0, y0, dy0, x1, y1, dy1
):
    """
    Make cubic coefficients from end points and derivatives
    """
    dt = x1 - x0
    A = (dy1 + dy0 - 2 * (y1 - y0) / dt) / (dt**2)
    B = (3 * (y1 - y0) / dt - 2 * dy0 - dy1) / dt
    C = dy0
    D = y0
    return A, B, C, D


def make_combined_quadratic(x0, y0, dy0, x1, y1, dy1):  # noqa: PLR0913
    """
    Make combined quadratic coefficients from end points and derivatives
    """
    dt = x1 - x0
    A = ((y1 - y0) / dt + (dy1 - dy0) / 2) / (dt)
    B = (y1 - y0) / dt + (dy0 - dy1) / 2
    C = y0
    return A, B, C


def smooth_step(t_vals, t_start, t_end):
    """
    Make smooth step between two functions
    """
    S = np.zeros(len(t_vals))
    for i, t_val in enumerate(t_vals):
        if t_val <= t_start:
            S[i] = 0
        elif t_val >= t_end:
            S[i] = 1
        else:
            x = (t_val - t_start) / (t_end - t_start)
            S[i] = x**3 * (x * (x * 6 - 15) + 10)
            S[i] = 0.5 * (1 + np.tanh(x))
            # S[i] = 3 * x**2 - 2 * x**3
            # S[i] = x
    return S


def make_linear_function_with_smooth_transition(  # noqa: PLR0913
    function, target_val, roll_start_length, roll_end_length, t_vals, t_extend
):
    """
    Make linear function with smooth transition to target
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    slope_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    vert_dist = target_val - function[-1]
    vert_dist_high = target_val - function[-1] - slope_at_extension * roll_start_length

    slope_max = vert_dist_high / (t_vals[-1] - t_vals[t_extend] - roll_end_length * 0.3)
    slope_min = (vert_dist) / (t_vals[-1] - t_vals[t_extend])
    linear_slope = np.mean([slope_max, slope_min])
    # height_at_start_of_linear = target_val - linear_slope *lin_seg_length
    height_at_start_of_linear = target_val - linear_slope * (t_vals[-1] - t_vals[t_extend] - roll_end_length * 0.7)
    indices_roll_1 = np.arange(t_extend + 1, t_extend + 1 + roll_start_length)
    indices_roll_2 = np.arange(len(t_vals) - roll_end_length, len(t_vals))
    indices_linear = np.arange(t_extend + 1 + roll_start_length, len(t_vals) - roll_end_length)
    # data_extend[indices_linear] = function[-1] + linear_slope* (t_vals[indices_linear] - t_vals[t_extend])
    data_extend[indices_linear] = height_at_start_of_linear + linear_slope * (t_vals[indices_linear] - t_vals[t_extend])

    interpolator1 = PchipInterpolator(
        np.concatenate((t_vals[: t_extend + 1], t_vals[indices_linear])),
        np.concatenate((function[: t_extend + 1], data_extend[indices_linear])),
    )
    data_extend[indices_roll_1] = interpolator1(t_vals[indices_roll_1])
    interpolator2 = PchipInterpolator(
        np.concatenate((t_vals[indices_linear], np.arange(t_vals[-1], t_vals[-1] + 100))),
        np.concatenate((data_extend[indices_linear], np.ones(100) * target_val)),
    )
    data_extend[indices_roll_2] = interpolator2(t_vals[indices_roll_2])
    return data_extend[t_extend + 1 :]


def make_cubic_roll_to_linear_extension(  # noqa: PLR0913
    function, target_val, roll_start_length, roll_end_length, t_vals, t_extend, vert_dist_end_frac=0.01
):
    """
    Make a cubic roll to linear extension
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    slope_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    lin_seg_length = len(t_vals) - len(function) - roll_start_length - roll_end_length
    # Calculate vertical distance to cover in final cubic
    vert_dist = (target_val - function[-1]) * (1 - vert_dist_end_frac)

    slope_max = vert_dist / lin_seg_length
    slope_min = vert_dist / (lin_seg_length + roll_start_length)
    slope = np.mean([slope_max, slope_min])
    height_at_start_of_linear = target_val - slope * lin_seg_length

    a1, b1, c1, d1 = make_cubic_coefficients_from_end_points_and_derivatives(
        t_vals[t_extend],
        function[-1],
        slope_at_extension,
        t_vals[t_extend] + roll_start_length,
        height_at_start_of_linear,
        slope,
    )
    a2, b2, c2, d2 = make_cubic_coefficients_from_end_points_and_derivatives(
        t_vals[t_extend] + roll_start_length + lin_seg_length,
        target_val + vert_dist_end_frac,
        slope,
        t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length,
        target_val,
        0,
    )
    # First cubic segment
    for i, t_val in enumerate(t_vals):
        if t_val <= t_vals[t_extend]:
            continue
        dt = t_val - t_vals[t_extend]
        if t_val <= t_vals[t_extend] + roll_start_length:
            data_extend[i] = a1 * dt**3 + b1 * dt**2 + c1 * dt + d1
        elif t_val <= t_vals[t_extend] + roll_start_length + lin_seg_length:
            data_extend[i] = height_at_start_of_linear + slope * (t_val - (t_vals[t_extend] + roll_start_length))
        elif t_val <= t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length:
            dt2 = dt - (roll_start_length + lin_seg_length)
            data_extend[i] = a2 * dt2**3 + b2 * dt2**2 + c2 * dt2 + d2
        else:
            data_extend[i] = target_val
    return data_extend[t_extend + 1 :]


def get_exp_targ_from_current_data(function, exp_end, fraction_extend=0.1):
    """
    Get exponential target from current data
    """
    slope_at_extension = get_derivative_using_spline(function, np.arange(len(function)), len(function) - 1)
    exp_targ = function[-1] + slope_at_extension * (exp_end - (len(function) - 1)) * fraction_extend

    return exp_targ


def make_quadratic_roll_to_linear_extension(  # noqa: PLR0913
    function, target_val, roll_start_length, roll_end_length, t_vals, t_extend, vert_dist_end_frac=0.01
):
    """
    Make a cubic roll to linear extension
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    slope_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    lin_seg_length = len(t_vals) - len(function) - roll_start_length - roll_end_length
    # Calculate vertical distance to cover in final cubic
    vert_dist = (target_val - function[-1]) * (1 - vert_dist_end_frac)
    vert_dist_for_end_cubic = (target_val - function[-1]) * vert_dist_end_frac

    slope_max = vert_dist / lin_seg_length
    slope_min = vert_dist / (lin_seg_length + roll_start_length)
    slope = np.mean([slope_max, slope_min])
    height_at_start_of_linear = target_val - slope * lin_seg_length - vert_dist_for_end_cubic
    print(f"slope at extension: {slope_at_extension}, calculated slope: {slope}")
    print(f"height at start of linear: {height_at_start_of_linear}")
    print(
        f"lin seg length: {lin_seg_length}, roll start length: {roll_start_length}, roll end length: {roll_end_length}"
    )
    print(f"target val: {target_val}, function end val: {function[-1]}")
    print(f"vert dist: {vert_dist}, vert dist for end cubic: {vert_dist_for_end_cubic}")

    a1, b1, c1 = make_combined_quadratic(
        t_vals[t_extend],
        function[-1],
        slope_at_extension,
        t_vals[t_extend] + roll_start_length,
        height_at_start_of_linear,
        slope,
    )
    a2, b2, c2 = make_combined_quadratic(
        t_vals[t_extend] + roll_start_length + lin_seg_length,
        target_val + vert_dist_end_frac,
        slope,
        t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length,
        target_val,
        0,
    )
    # First cubic segment
    for i, t_val in enumerate(t_vals):
        if t_val <= t_vals[t_extend]:
            continue
        dt = t_val - t_vals[t_extend]
        if t_val <= t_vals[t_extend] + roll_start_length:
            data_extend[i] = a1 * dt**2 + b1 * dt + c1
        elif t_val <= t_vals[t_extend] + roll_start_length + lin_seg_length:
            data_extend[i] = height_at_start_of_linear + slope * (t_val - (t_vals[t_extend] + roll_start_length))
        elif t_val <= t_vals[t_extend] + roll_start_length + lin_seg_length + roll_end_length:
            dt2 = dt - (roll_start_length + lin_seg_length)
            data_extend[i] = a2 * dt2**2 + b2 * dt2 + c2
        else:
            data_extend[i] = target_val
    return data_extend[t_extend + 1 :]


def get_derivative_using_spline(function, t_vals, t_extend):
    """
    Get derivative of function using spline
    """
    print(function[t_extend - 50 : t_extend])
    print(t_extend)
    spline = np.interp(
        t_vals[t_extend - 10 : t_extend + 10], t_vals[t_extend - 50 : t_extend], function[t_extend - 50 : t_extend]
    )
    return np.gradient(spline)[8]


def do_simple_sigmoid_or_exponential_extension_to_target(
    function: np.ndarray, t_vals: np.ndarray, t_extend: int, target: float, sigmoid_shift=40
) -> np.ndarray:
    """
    Calculate extension function by calling sigmoid functionality to extend
    """
    data_extend = np.zeros(len(t_vals))
    data_extend[: len(function)] = function
    derivative_at_extension = get_derivative_using_spline(function, t_vals, t_extend)
    sigmoid_derivative_at_extension = get_sigmoid_derivative(target, function[-1], sigmoid_shift)
    # print(f"Arguments for sigmoid: {target: }, come from: {scen_full.values[0, -1]},
    # start_time: {scen_full.columns[-1] + sigmoid_shift},
    # transition over: {2150 + sigmoid_shift}")
    if (
        derivative_at_extension < 0
        and sigmoid_derivative_at_extension < 0
        and (np.abs(derivative_at_extension) > np.abs(sigmoid_derivative_at_extension))
    ):
        data_extend[len(function) :] = exp_decay(
            target,
            target - function[-1],
            np.min((50, (target - function[-1]) / derivative_at_extension)),
            time_0=t_vals[t_extend],
            time=t_vals[len(function) :],
        )
    else:
        data_extend[len(function) :] = sigmoid_function(
            target,
            function[-1],
            2100 + sigmoid_shift,
            2150 + sigmoid_shift,
            t_vals[len(function) :],
        )
    # print(data_extend)
    return data_extend


def quick_plot_check(x, y, name):
    """
    Make a quick check_plot, just for testing
    """
    plt.clf()
    plt.plot(x, y)
    plt.savefig(name)


if __name__ == "__main__":
    # x = np.arange(2050, 2200)
    # quick_plot_check(x, sigmoid_function(1, -1, 2100, 2150, x), "plot_vanilla_sigmoid.png")
    # Example 1: Function with positive derivative, cubic turns negative
    x = np.arange(1950, 2300)
    func1 = np.linspace(0, 10, 100)  # Increasing function
    result1 = make_quadratic_roll_to_linear_extension(
        func1, target_val=5, roll_start_length=5, roll_end_length=5, t_vals=x, t_extend=99
    )
    quick_plot_check(x, result1, "plot_cubic_roll_to_linear_1.png")

    # Example 2: Function with negative derivative, cubic retains negative sign
    func2 = np.linspace(10, 0, 100)  # Decreasing function
    result2 = make_quadratic_roll_to_linear_extension(
        func2, target_val=-5, roll_start_length=5, roll_end_length=5, t_vals=x, t_extend=99
    )
    quick_plot_check(x, result2, "plot_cubic_roll_to_linear_2.png")

    # Example 3: Function with positive derivative, cubic retains positive sign
    func3 = np.linspace(0, 20, 100)  # Steeply increasing function
    result3 = make_quadratic_roll_to_linear_extension(
        func3, target_val=25, roll_start_length=5, roll_end_length=5, t_vals=x, t_extend=99
    )
    quick_plot_check(x, result3, "plot_cubic_roll_to_linear_3.png")
