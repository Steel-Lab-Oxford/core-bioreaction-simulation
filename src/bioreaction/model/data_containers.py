from copy import deepcopy
from typing import Any, List, Callable
from jax import numpy as jnp
import numpy as np
import chex
from dataclasses import dataclass, field
from ..misc.misc import get_unique_flat, per_mol_to_per_molecules

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

    def __init__(self, name: str, physical_data: dict = {}) -> None:
        self.name: str = name
        self.physical_data = physical_data
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
        return 'Species: ' + str(self.name)


@dataclass
class Reaction():
    """
    Some process which converts species into other 
    species. Mostly a symbolic class.
    """
    input: List[Species]
    output: List[Species]
    forward_rate: float
    reverse_rate: float = 0.0


class Reactant():
    """
    Translate between species and reaction
    """

    def __init__(self) -> None:
        self.species: Species
        self.quantity: Any
        self.units: Unit


class Extrinsics():
    """
    Other factors which we desire to model, which are relevent to our
    process.
    """

    def __init__(self) -> None:
        pass


@dataclass
class OtherFactor():
    """
    Represents some non-species quantity,
    such as the output of controller,
    or abundence of some cellualar resources.
    Probably a real number? 🙂 idk lol
    """
    name: str


@dataclass
class OUProcess():
    """
    Some other factor follows an OU process. 
    Simple
    """
    target: OtherFactor
    restoring_rate: float
    noise_scale: float

    @classmethod
    def scale_std_init(self, target: OtherFactor, time_scale: float, y_scale: float):
        """
        Define an OU process by the time scale, and the
        y scale. More natural and understandable. 
        """
        r_rate = 1.0/time_scale
        sigma = y_scale * np.sqrt(2 * r_rate)
        return OUProcess(target=target, restoring_rate=r_rate, noise_scale=sigma)


@dataclass
class ExtraReactionEffect():
    """
    Some other factor affects the rate of some reaction. 
    So the rate of reaction will be multiplied by exp(k*other_factor)
    """
    factor: OtherFactor
    target_reaction: Reaction
    forward_strength: float
    backward_stength: float = 0.0


@dataclass
class Impulse():
    """
    Represents a fixed amount of the target appearing 
    at a certain point in time. 
    """
    target: Species
    delta_target: float
    time: float
    impulse_width: float


@dataclass
class ControlledFactor():
    """
    Represents a fixed amount of the target appearing 
    at a certain in timepoint . 
    """
    targets: List[OtherFactor]
    observations: List[Species]
    control_function: Callable[[chex.ArrayDevice], chex.ArrayDevice]
    # how often you observe the system
    observation_period: float
    # how many observations before effect happen
    observation_delay: int
    # We have a tanh backend, with these effects in mind.
    # Hence we effectively put a tanh on whatever the network outputs.
    # The reason for a list is because if we have a neural network
    # controlling lots of things at once, with different scales.
    output_max: List[float]
    output_min: List[float]
    output_sensitivity: List[float]


@dataclass
class MedModel():
    """
    A slightly better, more complete model.
    Allows for more good stuff
    """
    species: List[Species]
    reactions: List[Reaction]
    other_factors: List[OtherFactor] = field(default_factory=list)
    reaction_extrinsics: List[ExtraReactionEffect] = field(
        default_factory=list)
    ou_effects: List[OUProcess] = field(default_factory=list)
    impulses: List[Impulse] = field(default_factory=list)
    controllers: List[ControlledFactor] = field(default_factory=list)


class BasicModel():
    """
    A class representing a collection of species, reactions, and other facts.
    This should represent the abstract notion of some mathematic model of a system.
    """

    def __init__(self, species: List[Species], reactions: List[Reaction]) -> None:
        self.species = species
        self.reactions = reactions


@chex.dataclass
class Reactions:
    # Input and output amounts, n-hot, each row being a different reaction
    #  each column is a species
    col_labels: list
    # n-hot inputs and outputs (again row: reaction, column: species)
    inputs: chex.ArrayDevice
    outputs: chex.ArrayDevice
    # inputs_onehot: chex.ArrayDevice
    # outputs_onehot: chex.ArrayDevice
    # Forward and reverse rates for each reaction
    forward_rates: chex.ArrayDevice
    reverse_rates: chex.ArrayDevice


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

    def init_properties(self, model: BasicModel, config):
        self.reactants = self.init_reactants(model, config)
        self.quantities = jnp.array(
            [r.quantity for r in self.reactants], dtype=JNP_DTYPE)
        self.reactions = self.init_reactions(model)

    def init_reactions(self, model: BasicModel):

        def make_onehot(matrix):
            onehot = deepcopy(matrix)
            onehot[onehot > 0] = 1
            return jnp.array(onehot, dtype=JNP_DTYPE)

        inputs = np.zeros((len(model.reactions), len(model.species)))
        outputs = np.zeros((len(model.reactions), len(model.species)))
        for i, r in enumerate(model.reactions):
            for inp in r.input:
                inputs[i, model.species.index(inp)] += 1
            for inp in r.output:
                outputs[i, model.species.index(inp)] += 1
        forward_rates = jnp.array([r.forward_rate for r in model.reactions])
        reverse_rates = jnp.array([r.reverse_rate for r in model.reactions])

        reactions = Reactions(col_labels=list([s.name for s in model.species]),
                              inputs=jnp.array(inputs, dtype=JNP_DTYPE),
                              outputs=jnp.array(outputs, dtype=JNP_DTYPE),
                              forward_rates=forward_rates, reverse_rates=reverse_rates)
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
