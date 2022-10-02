

from bioreaction.model.data_containers import BasicModel, Reaction, Species


def main():

    def create_combined_species(*species: str):
        return (s for s in species)

    circuit_size = 3
    circuit_node_names = [f'RNA_{i}' for i in range(circuit_size)]

    model = BasicModel()
    for species1 in circuit_node_names:
        for species2 in circuit_node_names:
            reaction = Reaction()
            reaction.input = [Species(species1), Species(species2)]
            reaction.output = [create_combined_species(Species(species1), Species(species2))]
            
            model.reactions.append(reaction)

    model.species = set([set(r.input) for r in model.reactions] 
    + [set(r.output) for r in model.reactions] )
