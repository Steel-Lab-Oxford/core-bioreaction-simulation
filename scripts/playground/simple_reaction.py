

import logging
import sys
from typing import Dict, List

import chex
import numpy as np
import jax


from src.bioreaction.model.data_containers import BasicModel, QuantifiedReactions, Reactant, Reaction, Reactions, Species
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from scripts.playground.misc import flatten_listlike, load_json_as_dict


def main():
    logging.info('Activating logger')

    def combine_species(*species: str):
        return [Species(tuple(species))]

    def make_species(species: List[str], ref_species: Dict[str, Species]):
        return list(ref_species[i] for i in sorted(species))

    def retrieve_species_from_reactions(model: BasicModel):
        return list(set(flatten_listlike([s for s in [r.input + r.output for r in model.reactions]])))

    def construct_model(config: dict):
        model = BasicModel()
        reactions_config = config['reactions']
        inputs = reactions_config.get('inputs')
        outputs = reactions_config.get('outputs')

        if outputs is None:
            outputs = [None] * len(inputs)
        ref_species = {s: Species(s) for s in set(flatten_listlike(inputs))}
        for i, o in zip(inputs, outputs):
            reaction = Reaction()
            reaction.input = make_species(i, ref_species)
            reaction.output = combine_species(
                *i) if o is None else make_species(i, ref_species)
            model.reactions.append(reaction)

        model.species = retrieve_species_from_reactions(model)
        return model

    config = load_json_as_dict('./scripts/playground/simple_config.json')
    model = construct_model(config)


    ##

    def create_reactions_from_model(model: BasicModel, config: dict):
        qreactions = QuantifiedReactions()
        qreactions.init_properties(model, config)
        return qreactions

    qreactions = create_reactions_from_model(model, config)
    sim_result = basic_de_sim(qreactions.quantities, qreactions.reactions, delta_t=0.1, num_steps=2)

    import matplotlib.pyplot as plt

    plt.plot(sim_result)
    plt.show()

    pass
