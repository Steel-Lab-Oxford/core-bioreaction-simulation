

import logging
import chex


from src.bioreaction.model.data_containers import BasicModel, QuantifiedReactions, Reactant, Reaction, Reactions, Species
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from scripts.playground.misc import load_json_as_dict


def main():

    def create_combined_species(*species: str):
        return (s for s in species)

    def construct_model(input_species_names):
        model = BasicModel()
        for species1 in input_species_names:
            for species2 in input_species_names:
                reaction = Reaction()
                reaction.input = [Species(species1), Species(species2)]
                reaction.output = [create_combined_species(Species(species1), Species(species2))]
                
                model.reactions.append(reaction)

        model.species = set([set(r.input) for r in model.reactions] 
        + [set(r.output) for r in model.reactions] )

    config = load_json_as_dict('./scripts/playground/simple_config.json')

    circuit_size = 3
    circuit_node_names = [f'RNA_{i}' for i in range(circuit_size)]

    model = construct_model(circuit_node_names)

    ##

    def pairup_reactants(model: BasicModel, config: dict):
        reactants = []
        for specie in model.species:
            reactant = Reactant()
            reactant.species = specie
            reactant.quantity = config['starting_concentration']
            reactants.append(reactant)
        return reactants

    def create_reactions_from_model(model : BasicModel, config: dict):
        reactions = QuantifiedReactions()
        reactions.reactants = pairup_reactants(model, config)
        reactions.reactions = model.reactions
        return reactions

    reactions = create_reactions_from_model(model, config)
    sim_result = basic_de_sim(reactions.quantities)

    logging.info(sim_result)
