import bioreaction
import chex
import jax.numpy as jnp
import numpy as np
import jax
from typing import Any, List

@chex.dataclass
class Reactions:
    # n-hot inputs and outputs (again row: reaction, column: species)
    inputs: chex.ArrayDevice
    outputs: chex.ArrayDevice
    # inputs_onehot: chex.ArrayDevice
    # outputs_onehot: chex.ArrayDevice
    # Forward and reverse rates for each reaction
    forward_rates: chex.ArrayDevice
    reverse_rates: chex.ArrayDevice



@chex.dataclass
class Impulse:
    target: int
    scale: float
    time_spread: float
    time: float

@chex.dataclass
class Impulse:
    target: int
    scale: float
    time_spread: float
    time: float


@chex.dataclass
class MedSimModel:
    reactions: Reactions
    other_factor_reaction_effects: chex.ArrayDevice
    impulses: List[Impulse]




    # species : List[Species]
    # reactions: List[Reaction]
    # other_factors: List[OtherFactor]
    # reaction_extrinsics: List[ExtraReactionEffect]
    # ou_effects: List[OUProcess]
    # impuluses: List[Impulse]
    # controllers: List[ControlledFactor]


@chex.dataclass
class BasicSimState:
    # This just be the state lol
    concentrations: chex.ArrayDevice
    other_factors: chex.ArrayDevice


@chex.dataclass
class BasicSimParams:
    delta_t: float
    t_start: float
    t_end: float


def get_reactions(input_model : bioreaction.data_containers.MedModel) -> Reactions:
    sp_num = len(input_model.species)
    react_num = len(input_model.reactions)
    sp_list = input_model.species

    inputs, outputs = [np.zeros((react_num, sp_num)) for i in range(2)]
    forward_rates, reverse_rates = [np.zeros(react_num) for i in range(2)]

    for react_ind, reacty in enumerate(input_model.reactions):
        for sp in reacty.input:
            inputs[react_ind, sp_list.index(sp)] += 1
        for sp in reacty.output:
            outputs[react_ind, sp_list.index(sp)] += 1
        forward_rates[react_ind] = reacty.forward_rate
        reverse_rates[react_ind] = reacty.reverse_rate

    return Reactions(inputs = jnp.array(inputs), outputs = jnp.array(outputs), 
                    forward_rates = jnp.array(forward_rates), reverse_rates = jnp.array(reverse_rates))



