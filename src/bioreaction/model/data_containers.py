from copy import deepcopy
from typing import Any, List
from jax import numpy as jnp
import numpy as np
import chex

from bioreaction.misc.misc import flatten_listlike, get_unique_flat


JNP_DTYPE = jnp.float32


class Unit():
    """
    Handling units
    """

    def __init__(self) -> None:
        self.name: str


class Species():
    """
    A chemical, protein, or other item which we desire to model
    the amount of over time.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        #self.identifier : str
        #self.data : Data

    def __lt__(self, other):
        if type(other.name) == tuple and type(self.name) == tuple:
            return self.name[0] < other.name[0]
        elif type(other.name) == tuple:
            return self.name > ''
        elif type(self.name) == tuple:
            return other.name > ''
        return self.name < other.name

    def __repr__(self) -> str:
        return 'Species: ' + self.name


class Reaction():
    """
    Some process which converts species into other 
    species. Mostly a symbolic class.
    """

    def __init__(self) -> None:
        self.input: List[Species]
        self.output: List[Species]
        self.base_rate: float
        #self.environmental: List[Tuple[int,Extrinsics]]


class Reactant():
    """
    Translate between species and reaction
    """

    def __init__(self) -> None:
        self.species: Species
        self.quantity: Any
        self.units: Unit


@chex.dataclass
class Reactions:
    # Input amounts, n-hot, each row being a different reaction
    #  each column is a species
    col_labels: list
    inputs: chex.ArrayDevice
    inputs_onehot: chex.ArrayDevice
    # Output amounts * rates. Each row is a different reaction
    #  each column is a species
    output_rates: chex.ArrayDevice



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

        self.species: List[Species] = []
        self.reactions: List[Reaction] = []


class QuantifiedReactions():
    """
    Translation from a symbolic-style Reaction to a form that includes 
    numbers for simulation.

    Might be mergable with BasicModel
    """

    def __init__(self) -> None:
        self.reactions: Reactions
        self.reactants: List[Reactant]
        self.quantities: chex.ArrayDevice
        self.rates: chex.ArrayDevice

    def init_properties(self, model: BasicModel, config):
        self.reactants = self.init_reactants(model, config)
        self.quantities = jnp.array(
            [r.quantity for r in self.reactants], dtype=JNP_DTYPE)
        self.reactions = self.init_reactions(model, config)

    def init_reactions(self, model: BasicModel, config: dict):
        species = get_unique_flat(
            [r.input + r.output for r in model.reactions])
        inputs = np.zeros((len(model.reactions), len(species)))
        for i, r in enumerate(model.reactions):
            for s in r.input:
                inputs[i, species.index(s)] += 1
        col_labels = list([s.name for s in get_unique_flat([r.input + r.output for r in model.reactions])])
        inputs_onehot = deepcopy(inputs)
        inputs_onehot[inputs_onehot > 0] = 1
        inputs_onehot = jnp.array(inputs_onehot, dtype=JNP_DTYPE)
        inputs = jnp.array(inputs, dtype=JNP_DTYPE)
        output_rates = jnp.array(
            config.get('output_rates'), dtype=JNP_DTYPE)
        reactions = Reactions(col_labels=col_labels, inputs=inputs, inputs_onehot=inputs_onehot, output_rates=output_rates)
        return reactions

    def init_reactants(self, model: BasicModel, config: dict):
        reactants = []
        input_species = get_unique_flat([r.input for r in model.reactions])

        for specie in model.species:
            reactant = Reactant()
            reactant.species = specie
            if specie in input_species:
                reactant.quantity = config['starting_concentration'][specie.name]
            else:
                reactant.quantity = 0
            reactants.append(reactant)
        return reactants
