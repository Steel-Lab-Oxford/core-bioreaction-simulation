from datetime import datetime
import numpy as np


def num_unsteadied(comparison, threshold):
    return np.sum(np.abs(comparison) > threshold)


def simulate_steady_states(y0, total_time, reverse_rates, sim_func, t0, t1,
                           threshold=0.1, disable_logging=False):
    """ Simulate a function sim_func for a chunk of time in steps of t0 - t1, starting at 
    t0 and running until either the steady states have been reached (specified via threshold) 
    or until the total_time as has been reached. """

    ti = t0
    iter_time = datetime.now()
    while True:
        if ti == t0:
            y00 = y0
        else:
            y00 = ys[:, -1, :]

        x_res = sim_func(y00, reverse_rates)

        if np.sum(np.argmax(x_res.ts >= np.inf)) > 0:
            ys = x_res.ys[:, :np.argmax(x_res.ts >= np.inf), :]
            ts = x_res.ts[:, :np.argmax(x_res.ts >= np.inf)] + ti
        else:
            ys = x_res.ys
            ts = x_res.ts + ti

        if ti == t0:
            ys_full = ys
            ts_full = ts
        else:
            ys_full = np.concatenate([ys_full, ys], axis=1)
            ts_full = np.concatenate([ts_full, ts], axis=1)

        if (num_unsteadied(ys[:, -1, :] - y00) == 0, threshold) or (ti >= total_time):
            break
        if not disable_logging:
            print('Steady states: ', ti, ' iterations. ', (num_unsteadied(
                ys[:, -1, :] - y00), threshold), ' left to steady out. ', datetime.now() - iter_time)

        ti += t1 - t0

    return np.array(ys_full), np.array(ts_full[0])
