


import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpy as np
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


    def step_function(t, total_time, step_num, dt, target):
        return jnp.floor_divide(t, (total_time - dt) / step_num) * (target/dt) * \
            (1 - jnp.floor_divide(t, (total_time  + dt) / step_num))

    # exponential decay subject to affine control
    def fbioreaction(t, y, args, x=0, signal_onehot=1):
        return -y * signal_onehot + x(t) * (-signal_onehot + 1)


    signal_species = [s for s in model.species if s.name == "RNA1"][0]
    signal_onehot = np.ones(len(model.species))
    signal_onehot[model.species.index(signal_species)] = 0

    t0, t1, dt0 = 0, 60, 0.1
    # signal = partial(x, total_time=t1, dt=dt0, step_signal=4)
    signal = partial(step_function, total_time=t1, step_num=2, dt=dt0, target=100)

    term = dfx.ODETerm(partial(fbioreaction, x=signal, signal_onehot=signal_onehot))
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sim_result = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, 
        y0=qreactions.quantities * signal_onehot, saveat=saveat)




    pass
