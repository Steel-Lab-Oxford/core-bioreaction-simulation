import bioreaction
import chex
import jax.numpy as jnp
import numpy as np
import jax
from typing import Any, List, Callable

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

@


@chex.dataclass
class MedSimModel:
    reactions: Reactions
    other_factor_reaction_effects: chex.ArrayDevice
    species_impulses: Callable[[float], chex.ArrayDevice]




    # species : List[Species]
    # reactions: List[Reaction]
    # other_factors: List[OtherFactor]
    # reaction_extrinsics: List[ExtraReactionEffect]
    # ou_effects: List[OUProcess]
    # impuluses: List[Impulse]
    # controllers: List[ControlledFactor]


@chex.dataclass
class MedSimState:
    # This just be the state lol
    concentrations: chex.ArrayDevice
    other_factors: chex.ArrayDevice
	stored_control: chex.ArrayDevice
	time: float


@chex.dataclass
class BasicSimParams:
    delta_t: float
    t_start: float
    t_end: float
	poisson_sim_reactions: chex.ArrayDevice
	brownian_sim_reaction: chex.ArrayDevice
	#Rest are going to be modelled continuously 


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
					
def get_base_reaction_rates(spec_conc: chex.ArrayDevice, reactions: Reactions):
    concentration_factors_in = jnp.prod(
        jnp.power(spec_conc, (reactions.inputs)), axis=1)
    concentration_factors_out = jnp.prod(
        jnp.power(spec_conc, (reactions.outputs)), axis=1)
    forward_delta = concentration_factors_in * reactions.forward_rates
    reverse_delta = concentration_factors_out * reactions.reverse_rates
    return (forward_delta - reverse_delta) @ (reactions.outputs - reactions.inputs)



