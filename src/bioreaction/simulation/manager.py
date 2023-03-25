from datetime import datetime
import numpy as np


def num_unsteadied(comparison, threshold):
    return np.sum(np.abs(comparison) > threshold)


def simulate_steady_states(y0, total_time, sim_func, t0, t1,
                           threshold=0.1, disable_logging=False, 
                           **sim_kwargs):
    """ Simulate a function sim_func for a chunk of time in steps of t1 - t0, starting at 
    t0 and running until either the steady states have been reached (specified via threshold) 
    or until the total_time as has been reached. Assumes batching.
    
    Args:
    y0: initial state, shape = (batch, time, vars)
    t0: initial time
    t1: simulation chunk end time
    total_time: total time to run the simulation function over
    sim_kwargs: any (batchable) arguments left to give the simulation function,
        for example rates or other parameters
    threshold: minimum difference between the final states of two consecutive runs 
        for the state to be considered steady
    """

    ti = t0
    iter_time = datetime.now()
    while True:
        if ti == t0:
            y00 = y0
        else:
            y00 = ys[:, -1, :]

        x_res = sim_func(y00, **sim_kwargs)

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
            print('Steady states: ', ti, ' iterations. ', num_unsteadied(
                ys[:, -1, :] - y00, threshold), ' left to steady out. ', datetime.now() - iter_time)

        ti += t1 - t0

    return np.array(ys_full), np.array(ts_full[0])
