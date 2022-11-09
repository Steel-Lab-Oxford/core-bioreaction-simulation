import jax.numpy as jnp 


def step_function(t, t0, t1, dt, target):
    return (jnp.sin(t) + 1)* jnp.where((t0 < t) & (t < t1), target/dt, 0)
