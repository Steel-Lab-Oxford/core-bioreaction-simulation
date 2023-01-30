
import chex
import jax.numpy as jnp
import numpy as np
import jax
from typing import Any, List, Callable, Tuple
from jaxtyping import Array, PyTree
import diffrax as dfx
import jax.random as jr
import equinox as eqx
import jax.tree_util as jtu
from ..model import data_containers


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
class OFModel:
    restoring_rates: chex.ArrayDevice
    sigmas: chex.ArrayDevice
    """
    Just defines an OU process atm lol
    """


@chex.dataclass
class OFEffects:
    forward: chex.ArrayDevice
    backward: chex.ArrayDevice
    """
    Forward and backward effect on reactions
    """


@chex.dataclass
class MedSimInternelState:
    concentrations: chex.ArrayDevice
    other_factors: chex.ArrayDevice


@chex.dataclass
class MedSimInternalImpulse:
    time: float
    impulse_width: float
    delta_species: MedSimInternelState


@chex.dataclass
class MedSimInternalModel:
    reactions: Reactions
    other_factor_noise_model: OFModel
    of_reaction_effects: OFEffects
    species_impulses: MedSimInternalImpulse
    # controllers: List[ControlledFactor]


@chex.dataclass
class MedSimState:
    # This just be the state lol
    int_state: MedSimInternelState
    stored_control: chex.ArrayDevice
    time: float


@chex.dataclass
class MedSimParams:
    delta_t: float
    t_start: float
    t_end: float
    poisson_sim_reactions: chex.ArrayDevice
    brownian_sim_reaction: chex.ArrayDevice
    # Rest are going to be modelled continuously


def get_OF(input_model: data_containers.MedModel) -> Tuple[OFModel, OFEffects]:
    """
    gives the ou stuff
    OF: Other Factors
    """
    of_num = len(input_model.other_factors)
    react_num = len(input_model.reactions)
    of_list = input_model.other_factors
    react_list = input_model.reactions

    react_effects_f = np.zeros((react_num, of_num))
    react_effects_b = np.zeros((react_num, of_num))
    rest_rates = np.zeros(of_num)
    sigmas = np.zeros(of_num)

    for x in input_model.ou_effects:
        rest_rates[of_list.index(x.target)] += x.restoring_rate
        sigmas[of_list.index(x.target)] += x.noise_scale

    sigmas = np.sqrt(sigmas)

    moddy = OFModel(restoring_rates=rest_rates, sigmas=sigmas)

    for x in input_model.reaction_extrinsics:
        react_i = react_list.index(x.target_reaction)
        factor_i = of_list.index(x.factor)
        react_effects_f[react_i, factor_i] += x.forward_strength
        react_effects_b[react_i, factor_i] += x.backward_stength

    effecty = OFEffects(forward=react_effects_f, backward=react_effects_b)

    return (moddy, effecty)


def get_int_impulse(input_model: data_containers.MedModel) -> MedSimInternalImpulse:
    sp_num = len(input_model.species)
    of_num = len(input_model.other_factors)
    imp_number = len(input_model.impulses)

    imp_times = jnp.array([x.time for x in input_model.impulses])
    time_widths = jnp.array(
        [x.impulse_width + 1e-10 for x in input_model.impulses])

    con_matrix = np.zeros((imp_number, sp_num))
    for i, x in enumerate(input_model.impulses):
        con_matrix[i, input_model.species.index(x.target)] = x.delta_target

    con_matrix = jnp.array(con_matrix)

    delta_sp = MedSimInternelState(concentrations=con_matrix, other_factors=0)

    return MedSimInternalImpulse(time=imp_times, impulse_width=time_widths, delta_species=delta_sp)


def get_reactions(input_model: data_containers.MedModel) -> Reactions:
    """
    Gives the reaction object thingy
    """
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

    return Reactions(inputs=jnp.array(inputs), outputs=jnp.array(outputs),
                     forward_rates=jnp.array(forward_rates), reverse_rates=jnp.array(reverse_rates))


def get_int_med_model(input_model: data_containers.MedModel) -> MedSimInternalModel:
    reacts = get_reactions(input_model)
    of_mod, of_effects = get_OF(input_model)
    int_imp = get_int_impulse(input_model)
    return MedSimInternalModel(reactions=reacts, other_factor_noise_model=of_mod,
                               of_reaction_effects=of_effects, species_impulses=int_imp)


def get_base_reaction_rates(spec_conc: chex.ArrayDevice, reactions: Reactions):
    concentration_factors_in = jnp.prod(
        jnp.power(spec_conc, (reactions.inputs)), axis=1)
    concentration_factors_out = jnp.prod(
        jnp.power(spec_conc, (reactions.outputs)), axis=1)
    forward_delta = concentration_factors_in * reactions.forward_rates
    reverse_delta = concentration_factors_out * reactions.reverse_rates
    return (forward_delta - reverse_delta)


def get_total_reaction_rates(state: MedSimInternelState, model: MedSimInternalModel) -> chex.ArrayDevice:
    base_rate = get_base_reaction_rates(state.concentrations, model.reactions)
    extra_rate = jnp.exp(
        model.of_reaction_effects.forward @ state.other_factors)
    return base_rate * extra_rate


class BasicPoisson(dfx.AbstractPath):
    """
    I give up, this is for reactions only lol
    """
    key: "jr.PRNGKey"
    transf_func: Callable[[PyTree[Array]], Array] = lambda x: x
    out_reshaper: Callable[[Array], PyTree[Array]] = lambda x: x

    @property
    def t0(self):
        return None

    @property
    def t1(self):
        return None

    def evaluate(self, t0, t1=None, left=True, *, y):
        t0_ = dfx.misc.force_bitcast_convert_type(t0, jnp.int32)
        t1_ = dfx.misc.force_bitcast_convert_type(t1, jnp.int32)
        new_key = jr.fold_in(self.key, t0_)
        new_key = jr.fold_in(new_key, t1_)
        return jr.poisson(new_key, self.transf_func(y(t0))*(t1-t0))


_sentinel = object()


class SolutionDependentControl(dfx.AbstractPath):
    control: dfx.AbstractPath
    y: Callable[[float], PyTree[Array]] = _sentinel

    @property
    def t0(self):
        return self.control.t0

    @property
    def t1(self):
        return self.contr0l.t1

    def evaluate(self, t0, t1=None, left=True):
        return self.control.evaluate(t0, t1, left, y=self.y)


class TauLeapingSolutionDependentSolver(dfx.AbstractItoSolver):
    solver: dfx.AbstractSolver

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):
        return self.solver.interpolation_cls

    def order(self, terms):
        return self.solver.order(terms)

    def strong_order(self, terms):
        return self.solver.strong_order(terms)

    def error_order(self, terms):
        return self.solver.error_order(terms)

    def init(self, terms, t0, t1, y0, args):
        return self.solver.init(terms, t0, t1, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        def is_control(x): return isinstance(x, SolutionDependentControl)

        def _bind_state(x):
            if is_control(x):
                assert x.y is _sentinel
                def sol_estimate(_): return y0  # tau-leaping approximation
                return eqx.tree_at(lambda control: control.y, x, sol_estimate)
            else:
                return x

        terms = jtu.tree_map(_bind_state, terms, is_leaf=is_control)

        new_y = jax.tree_util.tree_map(jax.nn.relu, y0)
        return self.solver.step(terms, t0, t1, new_y, args, solver_state, made_jump)

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)


def get_f_b_rates(state: MedSimInternelState, model: MedSimInternalModel):
    concentration_factors_in = jnp.prod(
        jnp.power(state.concentrations, (model.reactions.inputs)), axis=1)
    concentration_factors_out = jnp.prod(
        jnp.power(state.concentrations, (model.reactions.outputs)), axis=1)
    forward_base = concentration_factors_in * model.reactions.forward_rates
    reverse_base = concentration_factors_out * model.reactions.reverse_rates
    forward_extra = jnp.exp(
        model.of_reaction_effects.forward @ state.other_factors)
    reverse_extra = jnp.exp(
        model.of_reaction_effects.backward @ state.other_factors)
    return forward_base*forward_extra, reverse_base * reverse_extra


def get_dt_func(model: MedSimInternalModel, params: MedSimParams):
    def dt_term(t, y: MedSimInternelState, args):
        f_rate, b_rate = get_f_b_rates(y, model)
        masked_rates = (f_rate - b_rate) * (1 - params.poisson_sim_reactions)
        net_rate = masked_rates @ (model.reactions.outputs -
                                   model.reactions.inputs)
        other_restore = -1.0 * model.other_factor_noise_model.restoring_rates * y.other_factors

        return MedSimInternelState(concentrations=net_rate, other_factors=other_restore)
    return dt_term


def get_brown_noise_func(model: MedSimInternalModel, params: MedSimParams):
    def brown_noise_term(t, y: MedSimInternelState, args):
        f_rate, b_rate = get_f_b_rates(y, model)
        has_noise_mask = (1 - params.poisson_sim_reactions) * \
            params.brownian_sim_reaction
        noise_field = jnp.sqrt(f_rate + b_rate) * has_noise_mask * \
            (model.reactions.outputs - model.reactions.inputs).T

        other_noise_field = jnp.diag(model.other_factor_noise_model.sigmas)
        return MedSimInternelState(concentrations=noise_field, other_factors=other_noise_field)
    return brown_noise_term


def get_brown_noise_tree(key, model: MedSimInternalModel, params: MedSimParams):
    N_reacts = model.reactions.forward_rates.shape
    N_o_facts = model.other_factor_noise_model.restoring_rates.shape

    conc_shape = jax.ShapeDtypeStruct(N_reacts, jnp.float32)
    other_shape = jax.ShapeDtypeStruct(N_o_facts, jnp.float32)

    noise_shape = MedSimInternelState(
        concentrations=conc_shape, other_factors=other_shape)
    return dfx.VirtualBrownianTree(params.t_start, params.t_end, tol=params.delta_t * 0.1, shape=noise_shape, key=key)


class ReactionPoisson(dfx.AbstractPath):
    """
    Just getting something to work lol
    """
    key: "jr.PRNGKey"
    model: MedSimInternalModel
    params: MedSimParams

    @property
    def t0(self):
        return None

    @property
    def t1(self):
        return None

    def evaluate(self, t0, t1=None, left=True, *, y):
        t0_ = dfx.misc.force_bitcast_convert_type(t0, jnp.int32)
        t1_ = dfx.misc.force_bitcast_convert_type(t1, jnp.int32)
        new_key = jr.fold_in(self.key, t0_)
        new_key2 = jr.fold_in(new_key, t1_)
        forw_rate, back_rate = get_f_b_rates(y(t0), self.model)
        dt = t1-t0
        k1, k2 = jr.split(new_key2)
        forward_react_number = jr.poisson(k1, forw_rate*dt)
        backward_react_number = jr.poisson(k2, back_rate*dt)
        delta_react = forward_react_number - backward_react_number
        return MedSimInternelState(concentrations=delta_react, other_factors=0.0)


def get_poisson_func(model: MedSimInternalModel, params: MedSimParams):
    react_delta = model.reactions.outputs - model.reactions.inputs
    has_mask = params.poisson_sim_reactions
    masked_react_mat = has_mask * react_delta.T
    other_factor_stuff = jnp.zeros(
        model.other_factor_noise_model.restoring_rates.shape)
    return lambda t, y, args: MedSimInternelState(concentrations=masked_react_mat, other_factors=other_factor_stuff)


def eval_impulse(time: float, impulses: MedSimInternalImpulse) -> MedSimInternelState:
    return (1.0 + jnp.tanh((time - impulses.time)/impulses.impulse_width)) @ impulses.delta_species.concentrations / 2


class DumbControlPath(dfx.AbstractPath):
    model: MedSimInternalModel
    params: MedSimParams

    @property
    def t0(self):
        return None

    @property
    def t1(self):
        return None

    def evaluate(self, t0, t1, left: bool = True) -> PyTree:
        t1_val = eval_impulse(t1, self.model.species_impulses)
        t0_val = eval_impulse(t0, self.model.species_impulses)
        of_number = self.model.other_factor_noise_model.restoring_rates.shape
        delta = MedSimInternelState(
            concentrations=t1_val - t0_val, other_factors=jnp.zeros(of_number))
        return delta


# @chex.dataclass
# class MedSimInternalImpulse:
#     time: float
#     impulse_width: float
#     delta_species: MedSimInternelState

#     return lambda t : MedSimInternelState(concentrations =   (1.0 + jnp.tanh((t - imp_times)/time_widths  ) ) @ con_matrix / 2.0
#                                            , other_factors = np.zeros(of_num) )

def get_impulse_term(model: MedSimInternalModel, params: MedSimParams):
    def func(t, y, args):
        conc_part = jnp.ones(model.reactions.inputs.shape[1])
        of_part = jnp.zeros(
            model.other_factor_noise_model.restoring_rates.shape)
        return MedSimInternelState(concentrations=conc_part, other_factors=of_part)
    control = DumbControlPath(model, params)
    return dfx.WeaklyDiagonalControlTerm(func, control)


def simulate_chunk(key: jr.PRNGKey, init_state: MedSimInternelState, model: MedSimInternalModel, params: MedSimParams) -> chex.ArrayDevice:
    k1, k2, k3 = jr.split(key, 3)

    dt_func = get_dt_func(model, params)
    time_term = dfx.ODETerm(dt_func)

    brown_func = get_brown_noise_func(model, params)
    brown_tree = get_brown_noise_tree(k1, model, params)
    brown_term = dfx.ControlTerm(brown_func, brown_tree)

    poiss_path = ReactionPoisson(k2, model, params)
    poiss_func = get_poisson_func(model, params)
    poiss_term = dfx.ControlTerm(
        poiss_func, SolutionDependentControl(poiss_path))

    impulse_term = get_impulse_term(model, params)

    terms = dfx.MultiTerm(poiss_term, time_term, brown_term, impulse_term)

    solver = TauLeapingSolutionDependentSolver(dfx.Euler())

    saveat = dfx.SaveAt(ts=jnp.linspace(params.t_start, params.t_end, 100))
    return dfx.diffeqsolve(terms, solver, t0=params.t_start, t1=params.t_end, dt0=params.delta_t, y0=init_state, saveat=saveat)


def debug_simulate_chunk(key: jr.PRNGKey, init_state: MedSimInternelState, model: MedSimInternalModel, params: MedSimParams) -> chex.ArrayDevice:
    k1, k2, k3 = jr.split(key, 3)

    dt_func = get_dt_func(model, params)
    time_term = dfx.ODETerm(dt_func)

    brown_func = get_brown_noise_func(model, params)
    brown_tree = get_brown_noise_tree(k1, model, params)
    brown_term = dfx.ControlTerm(brown_func, brown_tree)

    poiss_path = ReactionPoisson(k2, model, params)
    poiss_func = get_poisson_func(model, params)
    poiss_term = dfx.ControlTerm(
        poiss_func, SolutionDependentControl(poiss_path))

    impulse_term = get_impulse_term(model, params)

    terms = dfx.MultiTerm(poiss_term, time_term, brown_term, impulse_term)

    solver = TauLeapingSolutionDependentSolver(dfx.Euler())

    saveat = dfx.SaveAt(ts=jnp.linspace(params.t_start, params.t_end, 100))
    return {"t_f": dt_func, "t_t": time_term, "b_f": brown_func,
            "b_p": brown_tree, "b_t": brown_term, "p_p": poiss_path, "p_f": poiss_func, "p_t": poiss_term, "i_t": impulse_term}


def very_basic_de(init_state: MedSimInternelState):
    def dumb_term(t, y, args): return init_state
    terms = dfx.ODETerm(dumb_term)
    solver = dfx.Euler()

    # Bad times :)))
    saveat = dfx.SaveAt(ts=jnp.linspace(0, 1, 100))
    return dfx.diffeqsolve(terms, solver, t0=0, t1=1, dt0=0.01, y0=init_state, saveat=saveat)
