import matplotlib.pyplot as plt
import numpy as np


def sigmoid_function(to_val, from_val, tstart, tend, z):
    """
    Calculate sigmoid function

    Calculate sigmoid function from from_val to to_val
    between tstart and tend for over t-values z
    """
    z_shift = (z - (tstart + tend) / 2.0) * 5 / 2.0 / (tend - tstart)
    scaling = to_val - from_val
    sigmoid = (1.0 / (1.0 + np.exp(-z_shift))) * scaling + from_val

    return sigmoid


def exp_decay(end, scale, decay_time, time_0, time):
    """
    Calculate exponential decay

    Calculate exponential decay to end value, with scale, decay_time,
    time_0 and over the values of time
    """
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
    elif tau > min_tau * 10:
        tau = min_tau * 10
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
        spline = np.interp(
            t_vals[t_extend - 10 : t_extend + 10], t_vals[t_extend - 50 : t_extend], function[t_extend - 50 : t_extend]
        )
        dlu_0 = np.gradient(spline)[8]
        print(dlu_0)
        cle_inf, k_mult, tau = solve_LU_constants(cle_0, lu_0, dlu_0)
    ext_func = np.zeros(len(t_vals))
    ext_func[:t_extend] = function[:t_extend]
    print(f"Value of tau is {tau}")
    ext_func[t_extend:] = k_mult / tau * np.exp((t_vals[t_extend] - t_vals[t_extend:]) / tau)
    return ext_func, cle_inf


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
