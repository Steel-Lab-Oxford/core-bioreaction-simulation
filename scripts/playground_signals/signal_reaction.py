

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from src.bioreaction.model.data_tools import construct_model
from src.bioreaction.model.data_containers import QuantifiedReactions
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from scripts.playground.misc import load_json_as_dict


def main():
    config = load_json_as_dict('./scripts/playground/simple_config.json')
    model = construct_model(config)

    qreactions = QuantifiedReactions()
    qreactions.init_properties(model, config)

    # sim_result = basic_de_sim(qreactions.quantities, qreactions.reactions,
    #                           delta_t=config['simulation']['delta_t'], num_steps=config['simulation']['num_steps'])
    # Signal
    import diffrax as dfx

    def x(t):  # control signal
        return t + 1

    # exponential decay subject to affine control
    def vector_field(t, y, args):
        return -y + x(t)

    term = dfx.ODETerm(vector_field)
    solver = dfx.Tsit5()
    sim_result = dfx.diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=1)

    # Visualise
    # df = pd.DataFrame(data=sim_result[1], columns=[
    #                   str(s.name) for s in model.species])
    # df['time'] = np.arange(config['simulation']['num_steps'] *
    #                        config['simulation']['delta_t'], step=config['simulation']['delta_t'])
    # dfm = df.melt('time', var_name='cols', value_name='vals')
    # sns.lineplot(x='time', y='vals', hue='cols', data=dfm)  # , kind='point')
    # plt.savefig('test.png')

    pass
