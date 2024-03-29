import jax
import jax.numpy as jnp
import chex
from ...model.data_containers import Reactions
from ...misc.misc import invert_onehot


def one_step_de_sim(spec_conc, reactions: Reactions):
    concentration_factors_in = jnp.prod(
        jnp.power(spec_conc, (reactions.inputs)), axis=1)
    concentration_factors_out = jnp.prod(
        jnp.power(spec_conc, (reactions.outputs)), axis=1)
    forward_delta = concentration_factors_in * reactions.forward_rates
    reverse_delta = concentration_factors_out * reactions.reverse_rates
    return (forward_delta - reverse_delta) @ (reactions.outputs - reactions.inputs)


def one_step_de_sim_expanded(spec_conc, inputs, outputs, forward_rates, reverse_rates):
    concentration_factors_in = jnp.prod(
        jnp.power(spec_conc, (inputs)), axis=1)
    concentration_factors_out = jnp.prod(
        jnp.power(spec_conc, (outputs)), axis=1)
    forward_delta = concentration_factors_in * forward_rates
    reverse_delta = concentration_factors_out * reverse_rates
    return (forward_delta - reverse_delta) @ (outputs - inputs)


def one_step_scan_wrapper(spec_conc: chex.ArrayDevice, reactions: Reactions, delta_t: float):
    return spec_conc + one_step_de_sim(spec_conc, reactions) * delta_t


def basic_de_sim(starting_concentration: chex.ArrayDevice, reactions: Reactions, delta_t: float, num_steps: int):
    def to_scan(carry, inp):
        step_output = one_step_scan_wrapper(carry, reactions, delta_t)
        return step_output, step_output
    return jax.lax.scan(to_scan, starting_concentration, None, length=num_steps)


# ODE Terms
def bioreaction_sim(t, y, args, reactions: Reactions, signal, signal_onehot: jnp.ndarray):
    return one_step_de_sim(spec_conc=y,
                           reactions=reactions) + signal(t) * signal_onehot


# ODE Terms
def bioreaction_sim_expanded(t, y,
                             args,
                             inputs, outputs,
                            #  signal, signal_onehot: jnp.ndarray,
                             forward_rates=None, reverse_rates=None):
    return one_step_de_sim_expanded(
        spec_conc=y, inputs=inputs,
        outputs=outputs,
        forward_rates=forward_rates,
        reverse_rates=reverse_rates) #+ signal(t) * signal_onehot
