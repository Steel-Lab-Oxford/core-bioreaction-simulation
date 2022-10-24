

import logging
from typing import Dict, List

import chex
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from src.bioreaction.model.data_tools import construct_model


from src.bioreaction.model.data_containers import BasicModel, QuantifiedReactions, Reaction, Species
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from scripts.playground.misc import load_json_as_dict


def main():
    logging.info('Activating logger')

    config = load_json_as_dict('./scripts/playground/simple_config.json')
    model = construct_model(config)

    qreactions = QuantifiedReactions()
    qreactions.init_properties(model, config)

    sim_result = basic_de_sim(qreactions.quantities, qreactions.reactions,
                              delta_t=config['simulation']['delta_t'], num_steps=config['simulation']['num_steps'])

    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(data=sim_result[1], columns=[
                      str(s.name) for s in model.species])
    df['time'] = np.arange(config['simulation']['num_steps'] *
                           config['simulation']['delta_t'], step=config['simulation']['delta_t'])
    dfm = df.melt('time', var_name='cols', value_name='vals')
    sns.lineplot(x='time', y='vals', hue='cols', data=dfm)  # , kind='point')
    plt.savefig('test.png')

    # plt.plot(time, sim_result[1])
    # plt.show()

    pass
