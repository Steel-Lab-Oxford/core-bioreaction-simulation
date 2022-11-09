

from typing import Dict, List
from bioreaction.model.data_containers import BasicModel, Reaction, Species
from bioreaction.misc.misc import flatten_listlike, get_unique_flat, per_mol_to_per_molecules
import jax.numpy as jnp


def combine_species(species: List[str], ref_species: Dict[str, Species]):
    return [Species(tuple(make_species(species, ref_species)))]


def pairup_combination(samples: list, astype=list):
    combination = []
    for si in samples:
        for sj in samples:
            combination.append(tuple(sorted([si, sj])))
    return sorted([astype(s) for s in (set(combination))])


def make_species(species: List[str], ref_species: Dict[str, Species]):
    return list(ref_species[i] for i in sorted(species))


def retrieve_species_from_reactions(model: BasicModel):
    return get_unique_flat([r.input + r.output for r in model.reactions])


def construct_model(config: dict):
    JNP_DTYPE = jnp.float32

    model = BasicModel()
    reactions_config = config['reactions']
    inputs = reactions_config.get('inputs')
    outputs = reactions_config.get('outputs')
    forward_rates = jnp.array(
        per_mol_to_per_molecules(config.get('forward_rates')), dtype=JNP_DTYPE)
    reverse_rates = jnp.array(
        per_mol_to_per_molecules(config.get('reverse_rates')), dtype=JNP_DTYPE)

    outputs = [None] * len(inputs) if outputs is None else outputs
    ref_species = {s: Species(s) for s in set(
        flatten_listlike(inputs + outputs, safe=True)) if s is not None}
    for idx, (i, o) in enumerate(zip(inputs, outputs)):
        reaction = Reaction()
        reaction.input = make_species(i, ref_species)
        reaction.output = combine_species(
            i, ref_species) if o is None else make_species(o, ref_species)
        reaction.forward_rate = forward_rates[idx]
        reaction.reverse_rate = reverse_rates[idx]
        model.reactions.append(reaction)

    model.species = retrieve_species_from_reactions(model)
    return model


def construct_model_fromnames(sample_names):

    model = BasicModel()
    inputs = sample_names + pairup_combination(sample_names) + [[]] * len(sample_names)
    outputs = [[]] * len(sample_names) + pairup_combination(sample_names, astype=str) + sample_names
    ref_species = {s: Species(s) for s in set(
        flatten_listlike(inputs + outputs, safe=True)) if s is not None}
    for idx, (i, o) in enumerate(zip(inputs, outputs)):
        reaction = Reaction()
        reaction.input = make_species(i, ref_species)
        reaction.output = combine_species(
            i, ref_species) if o is None else make_species(o, ref_species)
        reaction.forward_rate = None
        reaction.reverse_rate = None
        model.reactions.append(reaction)

    model.species = retrieve_species_from_reactions(model)
    return model
