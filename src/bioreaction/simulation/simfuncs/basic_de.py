from bioreaction.model.data_containers import Reactions
import jax
import jax.numpy as jnp
import chex


def one_step_de_sim(spec_conc: chex.ArrayDevice, reactions: Reactions, delta_t: float):
    concentration_factors_in = jnp.prod(
        jnp.power(spec_conc, (reactions.inputs)), axis=1)
    concentration_factors_out = jnp.prod(
        jnp.power(spec_conc, (reactions.outputs)), axis=1)
    forward_delta = concentration_factors_in * reactions.forward_rates
    reverse_delta = concentration_factors_out * reactions.reverse_rates
    return spec_conc \
        + ((forward_delta - reverse_delta) @ (reactions.outputs - reactions.inputs) 
        )  * delta_t


def basic_de_sim(starting_concentration: chex.ArrayDevice, reactions: Reactions, delta_t: float, num_steps: int):
    def to_scan(carry, inp):
        step_output = one_step_de_sim(carry, reactions, delta_t)
        return step_output, step_output
    return jax.lax.scan(to_scan, starting_concentration, None, length=num_steps)
