

import logging
import chex


from src.bioreaction.model.data_containers import BasicModel, QuantifiedReactions, Reactant, Reaction, Reactions, Species
from src.bioreaction.simulation.simfuncs.basic_de import basic_de_sim
from scripts.playground.misc import load_json_as_dict


def main():
    logging.info('Activating logger')

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

        logging.info([set(r.input) for r in model.reactions])
        model.species = set([tuple(set(r.input)) for r in model.reactions] 
        + [tuple(set(r.output)) for r in model.reactions] )
        return model

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
        qreactions = QuantifiedReactions()
        qreactions.reactants = pairup_reactants(model, config)
        qreactions.combine_reactants()
        qreactions.reactions = model.reactions

        reactions = Reactions()
        reactions.inputs = qreactions.quantities 
        reactions.output_rates = config
        return qreactions

    reactions = create_reactions_from_model(model, config)
    sim_result = basic_de_sim(reactions.quantities, reactions)

    logging.info(sim_result)
