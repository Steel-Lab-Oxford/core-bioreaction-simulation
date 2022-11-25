import bioreaction
import chex
import jax.numpy as jnp
import numpy as np
import jax
from typing import Any, List, Callable
from jaxtyping import Array, PyTree
import diffrax as dfx
import jax.random as jr
import equinox as eqx
import jax.tree_util as jtu

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
class MedSimInternalModel:
    reactions: Reactions
    other_factor_noise_model: OFModel
    of_reaction_effects: OFEffects
    species_impulses: Callable[[float], MedSimInternelState]

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
	#Rest are going to be modelled continuously 

def get_OF(input_model : bioreaction.data_containers.MedModel) -> tuple[OFModel, OFEffects]:
    """
    gives the ou stuff
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

    moddy = OFModel(restoring_rates = rest_rates, sigmas = sigmas)

    for x in input_model.reaction_extrinsics:
        react_i = react_list.index(x.target_reaction)
        factor_i = of_list.index(x.factor)
        react_effects_f[react_i, factor_i] += x.forward_strength
        react_effects_b[react_i, factor_i] += x.backward_stength

    effecty = OFEffects(forward = react_effects_f, backward = react_effects_b)

    return (moddy, effecty)

def get_impulse_func(input_model: bioreaction.data_containers.MedModel) ->  Callable[[float], MedSimInternelState]:
    sp_num = len(input_model.species)
    of_num = len(input_model.other_factors)
    imp_number = len(input_model.impuluses)

    input_model.impuluses

    imp_times = jnp.array([x.time for x in input_model.impuluses])
    time_widths = jnp.array([x.impulse_width + 1e-10 for x in input_model.impuluses])

    con_matrix = np.zeros((imp_number, sp_num))
    for i, x in enumerate(input_model.impuluses):
        con_matrix[i, input_model.species.index(x.target)] = x.delta_target

    con_matrix = jnp.array(con_matrix)

    

    return lambda t : MedSimInternelState(concentrations =   (1.0 + jnp.tanh((t - imp_times)/time_widths  ) ) @ con_matrix / 2.0
                                           , other_factors = np.zeros(of_num) )




def get_reactions(input_model : bioreaction.data_containers.MedModel) -> Reactions:
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

    return Reactions(inputs = jnp.array(inputs), outputs = jnp.array(outputs), 
                    forward_rates = jnp.array(forward_rates), reverse_rates = jnp.array(reverse_rates))

def get_int_med_model(input_model : bioreaction.data_containers.MedModel) -> MedSimInternalModel:
    reacts = get_reactions(input_model)
    of_mod, of_effects = get_OF(input_model)
    imp_func = get_impulse_func(input_model)
    return MedSimInternalModel(reactions = reacts, other_factor_noise_model = of_mod, of_reaction_effects = of_effects, species_impulses = imp_func)

def get_base_reaction_rates(spec_conc: chex.ArrayDevice, reactions: Reactions):
    concentration_factors_in = jnp.prod(
        jnp.power(spec_conc, (reactions.inputs)), axis=1)
    concentration_factors_out = jnp.prod(
        jnp.power(spec_conc, (reactions.outputs)), axis=1)
    forward_delta = concentration_factors_in * reactions.forward_rates
    reverse_delta = concentration_factors_out * reactions.reverse_rates
    return (forward_delta - reverse_delta) 

def get_total_reaction_rates(state: MedSimInternelState, model : MedSimInternalModel) -> chex.ArrayDevice:
    base_rate = get_base_reaction_rates(state.concentrations, model.reactions)
    extra_rate = jnp.exp(model.other_factor_reaction_effects @ state.other_factors)
    return base_rate * extra_rate 


class BasicPoisson(dfx.AbstractPath):
    """
    Just getting something to work lol
    """
    key: "jr.PRNGKey"
    transf_func: Callable[[PyTree[Array]], PyTree[Array]] = lambda x : x

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
        return jr.poisson(new_key, self.transf_func(y(t0))*(t1-t0)   )

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

class TauLeapingSolutionDependentSolver(dfx.AbstractSolver):
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
        is_control = lambda x: isinstance(x, SolutionDependentControl)
        
        def _bind_state(x):
            if is_control(x):
                assert x.y is _sentinel
                sol_estimate = lambda _ : y0  # tau-leaping approximation
                return eqx.tree_at(lambda control: control.y, x, sol_estimate)
            else:
                return x

        terms = jtu.tree_map(_bind_state, terms, is_leaf=is_control)
        return self.solver.step(terms, t0, t1, y0, args, solver_state, made_jump)

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)


def simulate_chunk(key : jr.PRNGKey, init_state: MedSimInternelState, model: MedSimInternalModel, params: MedSimParams) -> chex.ArrayDevice:
    k1, k2, k3 = jr.split(key, 3)
    
    def dt_term(t, y: MedSimInternelState, args):
        total_rate = get_total_reaction_rates(y, model)
        masked_rates = total_rate * (1 - params.poisson_sim_reactions)
        implied_rate =  masked_rates @ (model.reactions.outputs - model.reactions.inputs)
        
        other_restore = -1.0 * model.other_factor_noise_model.restoring_rates * y.other_factors
        return MedSimInternelState(concentrations = implied_rate, other_factors = other_restore)

    def brown_noise_term(t, y:MedSimInternelState, args):
        total_rate = get_total_reaction_rates(y, model)
        masked_rates = total_rate * (1 - params.poisson_sim_reactions) * (params.brownian_sim_reaction)
        spec_noise_scale = jnp.sqrt(masked_rates) @ (model.reactions.outputs - model.reactions.inputs)
        other_noise =  model.other_factor_noise_model.sigmas
        return MedSimInternelState(concentrations = spec_noise_scale, other_factors = other_noise)
    
    def poiss_trans_func(y: MedSimInternelState) -> chex.ArrayDevice:
        total_rate = get_total_reaction_rates(y, model)
        return total_rate * (params.poisson_sim_reactions)

    poiss_out_part = lambda t,y,args : (model.reactions.outputs - model.reactions.inputs)

    poiss_term = dfx.ControlTerm(poiss_out_part, SolutionDependentControl(BasicPoisson(k2, poiss_trans_func)))
    time_term = dfx.ODETerm(dt_term)

    brown_noise = dfx.UnsafeBrownianPath((1,), k1)

    brown_term = dfx.ControlTerm(brown_noise_term, brown_noise)
    #brown_noise = dfx.VirtualBrownianTree(params.t_start, params.t_end, tol = params.delta_t * 1e-2, shape = (), key = k1)

    #terms = dfx.MultiTerm(poiss_term, time_term, brown_term)
    #solver = TauLeapingSolutionDependentSolver(dfx.Euler())

    dumb_term = lambda t,y,args : init_state
    terms = dfx.ODETerm(dumb_term)
    solver = dfx.Euler()

    #Bad times :)))
    saveat = dfx.SaveAt(ts = jnp.linspace(params.t_start, params.t_end, 100))
    return dfx.diffeqsolve(terms, solver, t0 = params.t_start, t1 = params.t_end, dt0 = params.delta_t, y0=init_state, saveat=saveat)
    


    
    
