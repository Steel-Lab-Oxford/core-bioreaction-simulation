

import logging
import sys
from typing import List

import chex
import numpy
import jax


from src.bioreaction.model.data_containers import BasicModel, QuantifiedReactions, Reactant, Reaction, Reactions, Species
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from scripts.playground.misc import flatten_listlike, load_json_as_dict


def main():
    logging.info('Activating logger')

    def create_combined_species(*species: str):
        return [Species(tuple(species))]

    def make_species(species: List[str]):
        return list(Species(i) for i in sorted(species))

    def construct_model(config: dict):
        model = BasicModel()

        reactions_config = config['reactions']
        input = reactions_config.get('inputs')
        output = reactions_config.get('outputs')
        if output is None:
            output = [None] * len(input)
        for i, o in zip(input, output):
            reaction = Reaction()
            reaction.input = make_species(i)
            reaction.output = create_combined_species(
                *i) if o is None else make_species(i)
            model.reactions.append(reaction)

        species = [s for s in [r.input + r.output for r in model.reactions]]
        species = [s for s in model.reactions.input + model.reactions.output]

        species_names = list(set(
            list(set(flatten_listlike([r.input for r in model.reactions])))
            + list(set([r.output for r in model.reactions]))))
        model.species = list(Species(i) for i in species_names)
        model.species = list(set(
            list(set(flatten_listlike([r.input for r in model.reactions])))
            + list(set([r.output for r in model.reactions]))))
        return model

    config = load_json_as_dict('./scripts/playground/simple_config.json')

    model = construct_model(config)

    sys.exit()

    ##

    def create_reactions_from_model(model: BasicModel, config: dict):
        qreactions = QuantifiedReactions()
        qreactions.init_properties(model, config)

        reactions = Reactions()
        reactions.inputs = qreactions.quantities
        reactions.output_rates = chex.array(config.get('output_rates'))
        return qreactions

    reactions = create_reactions_from_model(model, config)
    sim_result = basic_de_sim(reactions.quantities, reactions)

    logging.info(sim_result)
