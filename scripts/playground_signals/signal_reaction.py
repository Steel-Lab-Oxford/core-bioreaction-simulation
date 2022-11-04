

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

    config = load_json_as_dict(
        './scripts/playground_signals/simple_config.json')
    model = construct_model(config)

    qreactions = QuantifiedReactions()
    qreactions.init_properties(model, config)

    def one_step_de_sim(spec_conc, reactions, delta_t=1):
        concentration_factors_in = jnp.prod(
            jnp.power(spec_conc, (reactions.inputs)), axis=1)
        concentration_factors_out = jnp.prod(
            jnp.power(spec_conc, (reactions.outputs)), axis=1)
        forward_delta = concentration_factors_in * reactions.forward_rates * delta_t
        reverse_delta = concentration_factors_out * reactions.reverse_rates * delta_t
        return spec_conc \
            + forward_delta @ (reactions.outputs) + reverse_delta @ (reactions.inputs) \
            - forward_delta @ (reactions.inputs) - \
            reverse_delta @ (reactions.outputs)

    def step_function(t, total_time, step_num, dt, target):
        return (jnp.floor_divide(t, (total_time - dt) / step_num) -
                jnp.floor_divide(t, (total_time + dt) / step_num)) * (target/dt)

    def bioreaction_sim(t, y, args, reactions, signal, signal_onehot, dt):
        return one_step_de_sim(spec_conc=(y * signal_onehot + signal(t) * (-signal_onehot + 1)),
                               reactions=reactions)

    t0, t1, dt0 = 0, 30, 0.2

    signal_species = [s for s in model.species if s.name ==
                      "RNA1" or s.name == "RNA2"]
    signal_species = []
    signal_onehot = np.ones(len(model.species))
    for s in signal_species:
        signal_onehot[model.species.index(s)] = 0

    signal = partial(step_function, total_time=t1,
                     step_num=2, dt=dt0, target=100)
    term = dfx.ODETerm(partial(bioreaction_sim, reactions=qreactions.reactions, signal=signal,
                               signal_onehot=signal_onehot, dt=dt0))
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)

    def naive(t1, y, reactions):
        for t in range(t1):
            y = one_step_de_sim(spec_conc=(y * signal_onehot + signal(t) * (-signal_onehot + 1)),
                            reactions=reactions)

    naive(t1, qreactions.quantities, qreactions.reactions)

    sim_result = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0,
                                 y0=qreactions.quantities * signal_onehot,
                                 saveat=saveat, max_steps=16**4)

    pass
