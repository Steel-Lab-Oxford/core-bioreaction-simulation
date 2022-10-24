

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from src.bioreaction.model.data_tools import construct_model
from src.bioreaction.model.data_containers import QuantifiedReactions
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from bioreaction.misc.misc import load_json_as_dict


def main():

    config = load_json_as_dict('./scripts/playground_signals/simple_config.json')
    model = construct_model(config)

    qreactions = QuantifiedReactions()
    qreactions.init_properties(model, config)

    # sim_result = basic_de_sim(qreactions.quantities, qreactions.reactions,
    #                           delta_t=config['simulation']['delta_t'], num_steps=config['simulation']['num_steps'])
    # Signal
    from functools import partial
    import diffrax as dfx

    def x(t, total_time, dt, step_signal):  # control signal
        return step_signal * t / (total_time / dt)

    def step_function(t, total_time, dt):
        if t < (total_time / dt / 2):
            return 0
        else:
            return 1

    # exponential decay subject to affine control
    def vector_field(t, y, args, x=None):
        return -y + x(t)

    t0, t1, dt0 = 0, 30, 0.1
    # signal = partial(x, total_time=t1, dt=dt0, step_signal=4)
    signal = partial(step_function, total_time=t1, dt=dt0)

    term = dfx.ODETerm(partial(vector_field, x=signal))
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sim_result = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=1, saveat=saveat)


    pass
