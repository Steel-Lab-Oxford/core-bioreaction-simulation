from datetime import datetime
import numpy as np


def num_unsteadied(comparison, threshold):
    return np.sum(np.abs(comparison) > threshold)


def simulate_steady_states(y0, total_time, reverse_rates, sim_func, params, threshold = 0.1):
    ti = 0
    iter_time = datetime.now()
    while True:
        if ti == params.t_start:
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
            
        if ti == params.t_start:
            ys_full = ys
            ts_full = ts
        else:
            ys_full = np.concatenate([ys_full, ys], axis=1)
            ts_full = np.concatenate([ts_full, ts], axis=1)

        if (num_unsteadied(ys[:, -1, :] - y00) == 0) or (ti >= total_time):
            break
        print('Steady states: ', ti, ' iterations. ', (num_unsteadied(
            ys[:, -1, :] - y00)), ' left to steady out. ', datetime.now() - iter_time)

        ti += params.t_end - params.t_start

    return np.array(ys_full), np.array(ts_full[0])
