import logging
from typing import Any, List, Tuple
from jax import numpy as jnp
import numpy as np
import chex

from scripts.playground.misc import flatten_listlike


JNP_DTYPE = jnp.float32


class Unit():
    """
    Handling units
    """
    def __init__(self) -> None:
        self.name : str


class Species():
    """
    A chemical, protein, or other item which we desire to model
    the amount of over time.
    """
    def __init__(self, name: str) -> None:
        self.name : str = name
        #self.identifier : str
        #self.data : Data


class Reaction():
    """
    Some process which converts species into other 
    species. Mostly a symbolic class.
    """
    def __init__(self) -> None:
        self.input : List[Species]
        self.output : List[Species]
        self.base_rate : float
        #self.environmental: List[Tuple[int,Extrinsics]]


class Reactant():
    """
    Translate between species and reaction
    """
    def __init__(self) -> None:
        self.species : Species
        self.quantity : Any
        self.units : Unit


@chex.dataclass
class Reactions():

    def __init__(self) -> None:
        # Inputs, one hot, each row being a different reaction
        #  each column is a species
        self.inputs : chex.ArrayDevice
        # Outputs * rates. Each row is a different reaction
        #  each column is a species
        self.output_rates : chex.ArrayDevice

class Extrinsics():
    """
    Other factors which we desire to model, which are relevent to our
    process.
    """
    def __init__(self) -> None:
        pass


class BasicModel():
    """
    A class representing a collection of species, reactions, and other facts.
    This should represent the abstract notion of some mathematic model of a system.
    """
    def __init__(self) -> None:

        self.species : List[Species] = []
        self.reactions : List[Reaction] = []


class QuantifiedReactions():
    """
    Translation from a symbolic-style Reaction to a form that includes 
    numbers for simulation.

    Might be mergable with BasicModel
    """
    def __init__(self) -> None:
        self.reactions : Reactions
        self.reactants : List[Reactant]
        self.quantities : chex.ArrayDevice
        self.rates : chex.ArrayDevice

    def init_properties(self, model: BasicModel, config):
        self.reactants = self.pairup_reactants(model, config)
        self.quantities = self.combine_reactants(self.reactants)

        reactions = Reactions()
        input_species = [s for s in model.species if s in flatten_listlike([r.input for r in model.reactions])]
        # onehot_indices = []
        # for i, r in enumerate(model.reactions):
        #     for s in r.input:
        #         onehot_indices.append([i, input_species.index(s)])
        # inputs[np.array(onehot_indices)[:, 0], np.array(onehot_indices)[:, 1]] = 1
        inputs = np.zeros((len(model.reactions), len(input_species)))
        for i, r in enumerate(model.reactions):
            for s in r.input:
                inputs[i, input_species.index(s)] += 1
        reactions.inputs = jnp.array(inputs, dtype=JNP_DTYPE)
        reactions.output_rates = jnp.array(config.get('output_rates'), dtype=JNP_DTYPE)
        self.reactions = reactions

    @staticmethod
    def combine_reactants(reactants: List[Reactant]):
        quantities = jnp.array([r.quantity for r in reactants], dtype=JNP_DTYPE)
        logging.warning(f'\nNot implemented: array returned as numpy instead of chex')
        return quantities

    def set_rates(self):
        pass

    def pairup_reactants(self, model: BasicModel, config: dict):
        reactants = []
        for specie, starting_concentration in zip(model.species, config['starting_concentration']):
            reactant = Reactant()
            reactant.species = specie
            reactant.quantity = starting_concentration
            reactants.append(reactant)
        return reactants
        