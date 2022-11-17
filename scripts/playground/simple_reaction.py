

import logging

import numpy as np
import pandas as pd
from src.bioreaction.model.data_tools import construct_model


from src.bioreaction.model.data_containers import QuantifiedReactions
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from bioreaction.misc.misc import flatten_listlike, load_json_as_dict


def main():
    logging.info('Activating logger')

    config = load_json_as_dict('./scripts/playground/simple_config.json')
    model = construct_model(config)

    qreactions = QuantifiedReactions()
    qreactions.init_properties(model, config)

    sim_result = basic_de_sim(qreactions.quantities, qreactions.reactions,
                              delta_t=config['simulation']['delta_t'], num_steps=config['simulation']['num_steps'])

    # Visualisation
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(data=sim_result[1], columns=[
                      str(s.name) for s in model.species])
    inout_grouping = {}
    output_group = {s.name: 'output' for s in model.species if s in flatten_listlike(r.output for r in model.reactions)}
    inout_grouping.update(output_group)
    input_group = {s.name: 'input' for s in model.species if s in flatten_listlike(r.input for r in model.reactions)}
    inout_grouping.update(input_group)
    both_group = {s.name: 'both' for s in model.species if s in flatten_listlike(r.output for r in model.reactions) and s in flatten_listlike(r.input for r in model.reactions)}
    inout_grouping.update(both_group)
    # df = df[list(both_group.keys())]
    df['time'] = np.arange(config['simulation']['num_steps'] *
                           config['simulation']['delta_t'], step=config['simulation']['delta_t'])
    df.drop(columns=['dRNA1', 'dRNA2', 'dRNA3', 'aRNA1', 'aRNA2', 'aRNA3'], inplace=True)
    dfm = df.melt('time', var_name='species', value_name='amounts')
    # ser = pd.Series(dfm['species'])
    # for k, v in output_group.items():
    #     ser[dfm['species'] == k] = v
    # dfm['in_out'] = ser
    sns.lineplot(x='time', y='amounts', hue='species', data=dfm)  # , kind='point')
    plt.savefig('test.png')

    # plt.plot(time, sim_result[1])
    # plt.show()

    pass
