
import chex
import jax.numpy as jnp
import numpy as np
import jax

from ..model import data_containers
from ..simulation.simfuncs.basic_de import one_step_scan_wrapper

@chex.dataclass
class BasicSimModel:
    # n-hot inputs and outputs (again row: reaction, column: species)
    inputs: chex.ArrayDevice
    outputs: chex.ArrayDevice
    # inputs_onehot: chex.ArrayDevice
    # outputs_onehot: chex.ArrayDevice
    # Forward and reverse rates for each reaction
    forward_rates: chex.ArrayDevice
    reverse_rates: chex.ArrayDevice

@chex.dataclass
class BasicSimState:
    # This just be the state lol
    concentrations: chex.ArrayDevice

@chex.dataclass
class BasicSimParams:
    delta_t: float
    total_time: float


def convert_model(input_model : data_containers.BasicModel) -> BasicSimModel:
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

    return BasicSimModel(inputs = jnp.array(inputs), outputs = jnp.array(outputs), 
                    forward_rates = jnp.array(forward_rates), reverse_rates = jnp.array(reverse_rates))


def basic_de_sim(starting_state: BasicSimState, model: BasicSimModel, params: BasicSimParams):
    def to_scan(carry, inp):
        step_output = one_step_scan_wrapper(carry, model, params.delta_t)
        return step_output, step_output
    return jax.lax.scan(to_scan, starting_state.concentrations, None, length= params.total_time // params.delta_t)
