

import logging
import sys

import chex
import numpy
import jax


from src.bioreaction.model.data_containers import BasicModel, QuantifiedReactions, Reactant, Reaction, Reactions, Species
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from scripts.playground.misc import flatten_listlike, load_json_as_dict


def main():
    logging.info('Activating logger') 

    def create_combined_species(*species: str):
        return tuple(species)

    def construct_model(config: dict):
        model = BasicModel()

        reactions_config = config['reactions']
        input = reactions_config.get('inputs')
        output = reactions_config.get('outputs')
        if output is None:
            output = [None] * len(input)
        for i, o in zip(input, output):
            reaction = Reaction()
            reaction.input = sorted(i)
            reaction.output = create_combined_species(*i) if o is None else tuple(o)
            model.reactions.append(reaction)

        model.species = set(list(set(flatten_listlike([r.input for r in model.reactions])))
        + list(set([r.output for r in model.reactions])) )
        return model

    config = load_json_as_dict('./scripts/playground/simple_config.json')

    model = construct_model(config)

    sys.exit()

    ##

    def create_reactions_from_model(model : BasicModel, config: dict):
        qreactions = QuantifiedReactions()
        qreactions.init_properties(model, config)

        reactions = Reactions()
        reactions.inputs = qreactions.quantities 
        reactions.output_rates = chex.array(config.get('output_rates'))
        return qreactions

    reactions = create_reactions_from_model(model, config)
    sim_result = basic_de_sim(reactions.quantities, reactions)

    logging.info(sim_result)
