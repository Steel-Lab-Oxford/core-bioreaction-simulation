


import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from functools import partial
import diffrax as dfx

from bioreaction.model.data_tools import construct_model
from bioreaction.model.data_containers import QuantifiedReactions
from bioreaction.misc.misc import load_json_as_dict


def main():

    config = load_json_as_dict('./scripts/playground_signals/simple_config.json')
    
    model = construct_model(config)

    qreactions = QuantifiedReactions()
    qreactions.init_properties(model, config)

    # sim_result = basic_de_sim(qreactions.quantities, qreactions.reactions,
    #                           delta_t=config['simulation']['delta_t'], num_steps=config['simulation']['num_steps'])
    # Signal

    def x(t, total_time, dt, step_signal):  # control signal
        return step_signal * t / (total_time / dt)

    def step_function(t, total_time_dt):
        return jnp.mod(total_time_dt / 2, t+1) / (total_time_dt / 2)

    # exponential decay subject to affine control
    def fbioreaction(t, y, args, x=None):
        # return x(t) #-y + x(t)
        return x(t)

    t0, t1, dt0 = 0, 30, 0.01
    # signal = partial(x, total_time=t1, dt=dt0, step_signal=4)
    signal = partial(step_function, total_time_dt=t1 / dt0)

    term = dfx.ODETerm(partial(fbioreaction, x=signal))
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sim_result = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=1, saveat=saveat)



    pass
