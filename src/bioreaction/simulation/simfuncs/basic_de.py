import jax
import jax.numpy as jnp
import chex

@chex.dataclass
class Reactions:
    # Inputs, one hot, each row being a different reaction
    #  each column is a species
    inputs : chex.ArrayDevice
    # Outputs * rates. Each row is a different reaction
    #  each column is a species
    output_rates : chex.ArrayDevice


def one_step_de_sim(spec_conc: chex.ArrayDevice, reactions: Reactions, delta_t: float):
    concentration_factors = jnp.prod(jnp.power(spec_conc, reactions.inputs), axis=1)
    implied_delta = reactions.output_rates @ concentration_factors * delta_t
    return spec_conc + implied_delta


def basic_de_sim(starting_concentration: chex.ArrayDevice, reactions: Reactions, delta_t: float, num_steps : int):
    def to_scan(carry, inp):
        step_output = one_step_de_sim(carry, reactions, delta_t)
        return step_output, step_output
    return jax.lax.scan(to_scan, starting_concentration, None, length = num_steps)[1]

