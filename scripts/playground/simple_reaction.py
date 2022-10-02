

from bioreaction.bioreactions.data_containers import Reaction, Species


def main():

    def create_combined_species(*species: str):
        return (s for s in species)

    circuit_size = 3
    circuit_node_names = [f'RNA_{i}' for i in range(circuit_size)]

    reactions = Reaction
    for species1 in circuit_node_names:
        for species2 in circuit_node_names:
            reaction = Reaction()
            reaction.input = [species1, species2]
            reaction.output = [create_combined_species(species1, species2)]
            
            reactions

    Species