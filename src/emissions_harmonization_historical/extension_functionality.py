import matplotlib.pyplot as plt
import numpy as np


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


def get_derivative_using_spline(function, t_vals, t_extend):
    """
    Get derivative of function using spline
    """
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
    x = np.arange(2050, 2200)
    quick_plot_check(x, sigmoid_function(1, -1, 2100, 2150, x), "plot_vanilla_sigmoid.png")
