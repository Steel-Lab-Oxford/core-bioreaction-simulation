

from typing import Dict, List
from src.bioreaction.model.data_containers import BasicModel, Reaction, Species
from scripts.playground.misc import flatten_listlike


def combine_species(species: List[str], ref_species: Dict[str, Species]):
    return [Species(tuple(make_species(species, ref_species)))]


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
    ref_species = {s: Species(s) for s in set(
        flatten_listlike(inputs + outputs, safe=True)) if s is not None}
    for i, o in zip(inputs, outputs):
        reaction = Reaction()
        reaction.input = make_species(i, ref_species)
        reaction.output = combine_species(
            i, ref_species) if o is None else make_species(i, ref_species)
        model.reactions.append(reaction)

    model.species = retrieve_species_from_reactions(model)
    return model
