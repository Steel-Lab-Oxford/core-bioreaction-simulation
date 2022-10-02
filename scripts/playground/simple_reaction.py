

import chex


from bioreaction.model.data_containers import BasicModel, Reaction, Reactions, Species
from bioreaction.simulation.simfuncs.basic_de import basic_de_sim
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

    starting_concentration = chex.array(config['starting_concentration'])

    Reactions.inputs
    basic_de_sim(starting_concentration)

