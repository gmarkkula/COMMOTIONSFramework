# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:47:55 2021

@author: tragma
"""
# assuming this file is in a subfolder to the COMMOTIONS framework root, so 
# add parent directory to Python path
import os 
import sys
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR, __ = os.path.split(THIS_FILE_DIR)
if not PARENT_DIR in sys.path:
    sys.path.append(PARENT_DIR)

# other imports
import os
import math
import copy
from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import commotions
import parameter_search
from sc_scenario import CtrlType, OptionalAssumption
import sc_scenario
import sc_scenario_helper

# expecting a results subfolder in the folder where this file is located
SCPAPER_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
FIT_RESULTS_FOLDER = SCPAPER_PATH + 'results/'

# scenario basics
PED_NAME = 'P'
VEH_NAME = 'V'
AGENT_NAMES = (PED_NAME, VEH_NAME)
i_PED_AGENT = 0
i_VEH_AGENT = 1
AGENT_CTRL_TYPES = (CtrlType.SPEED, CtrlType.ACCELERATION)
AGENT_WIDTHS = (0.8, 1.8)
AGENT_LENGTHS = (0.8, 4.2)
AGENT_FREE_SPEEDS = np.array([1.3, 50 / 3.6]) # m/s 
AGENT_GOALS = np.array([[0, 5], [-50, 0]]) # m
COLLISION_MARGIN = 1 # m
TIME_STEP = 0.1 # s
END_TIME = 8 # s
V_NY_REL = -1.5
AGENT_COLL_DISTS = []
for i_ag in range(2):
    AGENT_COLL_DISTS.append(sc_scenario_helper.get_agent_coll_dist(
        AGENT_LENGTHS[i_ag], AGENT_WIDTHS[1-i_ag]))
    

# model parameter values kept fixed in these fits
# - fixed value function gains, affordance-based
DEFAULT_PARAMS_K_VA = {}
for ctrl_type in CtrlType:
    DEFAULT_PARAMS_K_VA[ctrl_type] = commotions.Parameters()
    sc_scenario_helper.set_val_gains_for_free_speed(
        DEFAULT_PARAMS_K_VA[ctrl_type], 
        AGENT_FREE_SPEEDS[AGENT_CTRL_TYPES.index(ctrl_type)])
    DEFAULT_PARAMS_K_VA[ctrl_type]._da = 0.5
# - fixed value function gains, non-affordance-based
DEFAULT_PARAMS_K_NVA = {}
for ctrl_type in CtrlType:
    DEFAULT_PARAMS_K_NVA[ctrl_type] = commotions.Parameters()
    sc_scenario_helper.set_val_gains_for_free_speed(
        DEFAULT_PARAMS_K_NVA[ctrl_type], 
        AGENT_FREE_SPEEDS[AGENT_CTRL_TYPES.index(ctrl_type)])
    DEFAULT_PARAMS_K_NVA[ctrl_type]._e = 0
    # these two are only really applicable to acceleration-controlling agents, 
    # but there is no harm in adding them here for both agents
    DEFAULT_PARAMS_K_NVA[ctrl_type]._da = 0.01
    DEFAULT_PARAMS_K_NVA[ctrl_type]._sg = 0
def get_default_params_k(model_str):
    if 'oVA' in model_str:
        return DEFAULT_PARAMS_K_VA
    else:
        return DEFAULT_PARAMS_K_NVA
# - other fixed parameters
DEFAULT_PARAMS = commotions.Parameters()
DEFAULT_PARAMS.H_e = 1.5
DEFAULT_PARAMS.sigma_xdot = 0.1
DEFAULT_PARAMS.T_P = 0.5
DEFAULT_PARAMS.T_s = 1
DEFAULT_PARAMS.D_s = 1
DEFAULT_PARAMS.thetaDot_0 = 0.001
DEFAULT_PARAMS.beta_O = 1 
DEFAULT_PARAMS.T_O1 = 0.05 
#DEFAULT_PARAMS.DeltaV_th_rel = 0.001 
DEFAULT_PARAMS.DeltaT = 0.5 
DEFAULT_PARAMS.V_0_rel = 4 
DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.5, 0, 0.5, 1]) 
#DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]) 


@dataclass
class ModelWithParams:
    model: str
    param_names: list
    params_array: np.ndarray
        

# deterministic fitting
class SCPaperScenario:
    
    def get_full_metric_name(self, short_metric_name):
        return self.name + '_' + short_metric_name
    
    def get_dists_and_accs(self, i_variation=None, force_calc=False):
        if self.n_variations == 1 and not force_calc:
            # should already have been calculated in call from the constructor
            return self.initial_cp_distances, self.const_accs
        # we need to do the calculations
        # - get the initial TTCAs for the two agents, in this scenario variation
        initial_ttcas = []
        for i_agent in range(2):
            if i_agent == self.i_varied_agent:
                if i_variation == None:
                    raise Exception('Need to specify scenario variation.')
                initial_ttcas.append(self.initial_ttcas[i_agent][i_variation])
            else:
                initial_ttcas.append(self.initial_ttcas[i_agent])
        # - get the corresponding initial distances
        initial_cp_distances = (np.array(initial_ttcas) * AGENT_FREE_SPEEDS 
                                + np.array(AGENT_COLL_DISTS))
        # - get any constant accelerations
        const_accs = [None, None]
        if self.ped_start_standing:
            initial_cp_distances[i_PED_AGENT] = (
                AGENT_COLL_DISTS[i_PED_AGENT] + self.ped_standing_margin)
        if self.ped_const_speed:
            const_accs[i_PED_AGENT] = 0
        if self.veh_const_speed:
            const_accs[i_VEH_AGENT] = 0
        elif self.veh_yielding:
            stop_dist = (initial_cp_distances[i_VEH_AGENT] 
                         - AGENT_COLL_DISTS[i_VEH_AGENT] - self.veh_yielding_margin)
            const_accs[i_VEH_AGENT] = (
                -self.initial_speeds[i_VEH_AGENT] ** 2 / (2 * stop_dist))
        return initial_cp_distances, const_accs
        
    
    def __init__(self, name, initial_ttcas, ped_prio=False,
                 ped_start_standing=False, ped_standing_margin=COLLISION_MARGIN,
                 ped_const_speed=False, veh_const_speed=False, 
                 veh_yielding=False, veh_yielding_margin=COLLISION_MARGIN,
                 stop_criteria = (), metric_names = None):
        """ Construct a scenario. 
            
            initial_ttcas should be a tuple defining the initial times to
            the conflict space (conflict area) for the two agents. Each element
            in the tuple can either be a single numerical value, or math.nan
            for the pedestrian if ped_start_standing=True. For one of the agents,
            instead of a numerical value, a tuple of numerical values can be
            provided; if so these define different variations to the scenario.  
            
        """
        # store scenario info
        self.name = name
        self.ped_prio = ped_prio
        self.ped_start_standing = ped_start_standing
        self.ped_standing_margin = ped_standing_margin
        self.ped_const_speed = ped_const_speed
        self.veh_const_speed = veh_const_speed
        self.veh_yielding = veh_yielding
        self.veh_yielding_margin = veh_yielding_margin
        self.stop_criteria = stop_criteria
        # figure out if there are any kinematic variations to consider
        self.initial_ttcas = initial_ttcas
        self.n_variations = 1
        self.i_varied_agent = None
        for i_agent in range(2):
            if type(initial_ttcas[i_agent]) == tuple:
                if not (self.i_varied_agent == None):
                    raise Exception('Kinematic variations for more than one agent not supported.')
                self.i_varied_agent = i_agent
                self.n_variations = len(initial_ttcas[i_agent])  
        # get and store initial speeds
        self.initial_speeds = np.copy(AGENT_FREE_SPEEDS)
        if ped_start_standing:
            self.initial_speeds[i_PED_AGENT] = 0       
        # set initial distances and constant accelerations here only if 
        # scenario has just a single variation
        if self.n_variations == 1:
            self.initial_cp_distances, self.const_accs = \
                self.get_dists_and_accs(i_variation=1, force_calc=True)
        # store metric info
        self.short_metric_names = metric_names
        self.full_metric_names = []
        for short_metric_name in self.short_metric_names:
            self.full_metric_names.append(
                self.get_full_metric_name(short_metric_name))
        
    
# scenarios
# - just defining some shorter names for the simulation stopping criteria
HALFWAY_STOP = (sc_scenario.SimStopCriterion.ACTIVE_AGENT_HALFWAY_TO_CS,)
MOVED_STOP = (sc_scenario.SimStopCriterion.BOTH_AGENTS_HAVE_MOVED,)
EXITED_STOP = (sc_scenario.SimStopCriterion.BOTH_AGENTS_EXITED_CS,)
# - one-agent scenarios
N_ONE_AG_SCEN_VARIATIONS = 3
ONE_AG_SCENARIOS = {}
ONE_AG_SCENARIOS['VehPrioAssert'] = SCPaperScenario('VehPrioAssert', 
                                                   initial_ttcas=(math.nan, (1.5, 2, 2.5)),  
                                                   ped_start_standing=True, 
                                                   ped_const_speed=True,
                                                   stop_criteria = HALFWAY_STOP,
                                                   metric_names = ('veh_av_speed',))
ONE_AG_SCENARIOS['VehShortStop'] = SCPaperScenario('VehShortStop', 
                                                  initial_ttcas=(math.nan, (3.5, 4, 4.5)),
                                                  ped_prio = True,
                                                  ped_start_standing=True, 
                                                  ped_const_speed=True,
                                                  stop_criteria = HALFWAY_STOP,
                                                  metric_names = ('veh_av_surpl_dec',))
ONE_AG_SCENARIOS['PedHesitateVehConst'] = SCPaperScenario('PedHesitateVehConst', 
                                                          initial_ttcas=(3, (7.5, 8, 8.5)), 
                                                          veh_const_speed=True,
                                                          stop_criteria = HALFWAY_STOP,
                                                          metric_names = ('ped_av_speed',))
ONE_AG_SCENARIOS['PedHesitateVehYield'] = SCPaperScenario('PedHesitateVehYield', 
                                                         initial_ttcas=(3, (2.5, 3, 3.5)), 
                                                         ped_prio=True,
                                                         veh_yielding=True,
                                                         stop_criteria = HALFWAY_STOP,
                                                         metric_names = ('ped_av_speed',))
ONE_AG_SCENARIOS['PedCrossVehYield'] = SCPaperScenario('PedCrossVehYield', 
                                                       initial_ttcas=(math.nan, (1.5, 2, 2.5)), 
                                                       ped_prio=True,
                                                       ped_start_standing=True, 
                                                       veh_yielding=True,
                                                       stop_criteria = HALFWAY_STOP,
                                                       metric_names = ('veh_speed_at_ped_start',))
# - two-agent scenarios
TWO_AG_METRIC_NAMES = ('collision', 'ped_exit_time', 'veh_exit_time')
TWO_AG_SCENARIOS = {}
TWO_AG_SCENARIOS['Encounter'] = SCPaperScenario('Encounter', 
                                                initial_ttcas=(3, 3), 
                                                metric_names = TWO_AG_METRIC_NAMES)
TWO_AG_SCENARIOS['EncounterPedPrio'] = SCPaperScenario('EncounterPedPrio', 
                                                       initial_ttcas=(3, 3), 
                                                       ped_prio=True,
                                                       metric_names = TWO_AG_METRIC_NAMES)
TWO_AG_SCENARIOS['PedLead'] = SCPaperScenario('PedLead', 
                                              initial_ttcas=(3, 8), 
                                              metric_names = TWO_AG_METRIC_NAMES)


DET_FIT_FILE_NAME_FMT = 'DetFit_%s.pkl'
METRIC_FCN_PREFIX = 'metric_'


def get_halfway_to_cs_sample(sim, i_agent):
    agent = sim.agents[i_agent]
    halfway_dist = sc_scenario_helper.get_agent_halfway_to_CS_CP_dist(agent)
    beyond_halfway_samples = np.nonzero(sim.agents[i_agent].signed_CP_dists
                                        <= halfway_dist)[0]
    if len(beyond_halfway_samples) == 0:
        return math.nan
    else:
        return beyond_halfway_samples[0]

def metric_agent_av_speed(sim, i_agent):
    idx_halfway = get_halfway_to_cs_sample(sim, i_agent)
    if math.isnan(idx_halfway):
        return math.nan
    else:
        return np.mean(sim.agents[i_agent].trajectory.long_speed[:idx_halfway])

def metric_ped_av_speed(sim):
    return metric_agent_av_speed(sim, i_PED_AGENT)

def metric_veh_av_speed(sim):
    return metric_agent_av_speed(sim, i_VEH_AGENT)

def metric_veh_av_surpl_dec(sim):
    idx_RT = np.nonzero(sim.time_stamps > DEFAULT_PARAMS.DeltaT)[0][0]
    idx_halfway = get_halfway_to_cs_sample(sim, i_VEH_AGENT)
    if math.isnan(idx_halfway):
        return math.nan
    else:
        veh_agent = sim.agents[i_VEH_AGENT]
        # get the decelerations needed to stop
        stop_dists = (veh_agent.signed_CP_dists[idx_RT:idx_halfway] 
                      - veh_agent.coll_dist - COLLISION_MARGIN)
        stop_decs = (veh_agent.trajectory.long_speed[idx_RT:idx_halfway] ** 2 
                     / (2 * stop_dists))
        # compare to actual decelerations
        actual_decs = -veh_agent.trajectory.long_acc[idx_RT:idx_halfway]
        return np.mean(actual_decs - stop_decs)

def metric_veh_speed_at_ped_start(sim):
    idx_halfway = get_halfway_to_cs_sample(sim, i_PED_AGENT)
    if math.isnan(idx_halfway):
        return math.nan
    else:
        return sim.agents[i_VEH_AGENT].trajectory.long_speed[idx_halfway]

def metric_collision(sim):
    raise NotImplementedError

def metric_agent_exit_time(sim, i_agent):
    raise NotImplementedError

def metric_ped_exit_time(sim):
    return metric_agent_exit_time(sim, i_PED_AGENT)

def metric_veh_exit_time(sim):
    return metric_agent_exit_time(sim, i_VEH_AGENT)


def get_metrics_for_scenario(scenario, sim, verbose=False, report_prefix=''):
    metrics = {}
    for short_metric_name in scenario.short_metric_names:
        metric_value = globals()[
            METRIC_FCN_PREFIX + short_metric_name](sim)
        full_metric_name = scenario.get_full_metric_name(short_metric_name)
        metrics[full_metric_name] = metric_value
        if verbose:
            print(report_prefix + 
                  f'Metric {full_metric_name} = {metrics[full_metric_name]}')
    return metrics


def simulate_scenario(scenario, optional_assumptions, params, params_k, 
                      i_variation=None, snapshots=(None, None)):
    # prepare the simulation
    # - get initial distances and constant accelerations for this scenario variation
    if scenario.n_variations > 1 and i_variation == None:
        raise Exception('Need to specify scenario variation to run.')
    initial_cp_dists, const_accs = scenario.get_dists_and_accs(i_variation)
    # - initial position
    initial_pos = np.array([[0, -initial_cp_dists[i_PED_AGENT]],
                           [initial_cp_dists[i_VEH_AGENT], 0]])
    # - pedestrian priority?
    if scenario.ped_prio:
        params.V_ny_rel = V_NY_REL
    else:
        params.V_ny_rel = 0
    # - set up the SCSimulation object
    sc_simulation = sc_scenario.SCSimulation(
        ctrl_types=AGENT_CTRL_TYPES, 
        widths=AGENT_WIDTHS, lengths=AGENT_LENGTHS, 
        goal_positions=AGENT_GOALS, initial_positions=initial_pos, 
        initial_speeds=scenario.initial_speeds, 
        const_accs=const_accs,
        start_time=0, end_time=END_TIME, time_step=TIME_STEP, 
        optional_assumptions=optional_assumptions, 
        params=params, params_k=params_k, 
        agent_names=AGENT_NAMES, snapshot_times=snapshots,
        stop_criteria=scenario.stop_criteria)
    
    # run the simulation
    sc_simulation.run()
    
    # return the simulation object
    return sc_simulation


def get_metrics_for_params(scenarios, optional_assumptions, params, params_k,
                           i_scenario_variation=None, verbosity=0, report_prefix=''):
    # loop through the scenarios, simulate them, and calculate metrics
    metrics = {}
    for scenario in scenarios.values():
        
        if verbosity >= 1:
            print(report_prefix + f'Simulating scenario "{scenario.name}"...')
        
        # run this scenario with the specified assumptions and parameterisation
        sc_simulation = simulate_scenario(scenario, optional_assumptions, 
                                          params, params_k, i_scenario_variation)
        
        # calculate metric(s) for this scenario
        scenario_metrics = get_metrics_for_scenario(scenario, sc_simulation,
                                                    verbosity >= 2,
                                                    report_prefix + '-> ')
        metrics.update(scenario_metrics)
        
        # plot simulation results?
        # (this will only work nicely if run with %matplotlib inline)
        if verbosity >= 3:
            sc_simulation.do_plots(trajs=False, action_val_ests = False, 
                                   surplus_action_vals = False,
                                   kinem_states = True, beh_accs = False, 
                                   beh_probs = True, action_vals = False, 
                                   sensory_prob_dens = False, 
                                   beh_activs = False)
            print('Showing plots, press [Enter] to continue...')
            input()
            
    # return the dict of metrics
    return metrics


def get_metrics_for_params_parallel(i_parameterisation, i_repetition, 
                                    scenarios, optional_assumptions, params, params_k,
                                    i_scenario_variation=None, verbosity=0, report_prefix=''):
    metrics = get_metrics_for_params(
        scenarios, optional_assumptions, params, params_k,
        i_scenario_variation, verbosity, report_prefix)
    return(i_parameterisation, i_repetition, metrics)


# class for searching/testing parameterisations of sc_scenario.SCSimulation
class SCPaperParameterSearch(parameter_search.ParameterSearch):
    
    def set_params(self, params_dict):
        """
        Set self.params and self.params_k to reflect a given model 
        parameterisation.

        Parameters
        ----------
        params_dict : dict
            Parameter values, with keys from self.param_names

        Returns
        -------
        None.

        """    
        # loop through the provided free parameter values and assign them
        # to the correct attributes of the local parameter objects
        for param_name, param_value in params_dict.items():
            if param_name[0:2] == 'k_':
                # value function gain parameter - set across both control types
                # (redundant, but doesn't matter)
                param_attr = param_name[1:]
                for ctrl_type in CtrlType:
                    setattr(self.params_k[ctrl_type], param_attr, param_value)
            else:
                # other parameter
                setattr(self.params, param_name, param_value)
    
    
    def simulate_scenario(self, scenario, i_variation='all', snapshots=(None, None)):
        """
        Run a given scenario for the model parameterisation currently
        specified by self.params and self.params_k.

        Parameters
        ----------
        scenario : SCPaperScenario
            The scenario to be simulated.

        Returns
        -------
        The resulting sc_scenario.SCSimulation object.

        """        
        if self.n_scenario_variations == 1:
            return simulate_scenario(scenario, self.optional_assumptions, 
                                     self.params, self.params_k, 
                                     snapshots=snapshots)
        else:
            if i_variation == 'all':
                sims = []
                for i_var in range(self.n_scenario_variations):
                    sims.append(simulate_scenario(scenario, 
                                                  self.optional_assumptions, 
                                                  self.params, self.params_k, 
                                                  i_variation=i_var, 
                                                  snapshots=snapshots))
                return sims
            else:
                return simulate_scenario(scenario, self.optional_assumptions, 
                                         self.params, self.params_k, 
                                         i_variation=i_variation, 
                                         snapshots=snapshots)
    
    def get_metrics_for_params(self, params_dict, i_parameterisation, 
                               i_repetition):
        
        self.verbosity_push()
        if self.n_repetitions > 1:
            rep_str = f'rep. #{i_repetition+1}/{self.n_repetitions} for '
        else:
            rep_str = ''
        self.report(f'Simulating {rep_str}params'
                    f' #{i_parameterisation+1}/{self.n_parameterisations}:'
                    f' {params_dict}...')
        
        # set the model parameters as specified and copy objects for parallelisation
        self.set_params(params_dict)
        params = copy.deepcopy(self.params)
        params_k = copy.deepcopy(self.params_k)
        
        # specify which of any scenario variations we are running
        if self.n_scenario_variations > 1:
            i_scenario_variation = i_repetition
        else:
            i_scenario_variation = None
        
        # get and return the dict of metrics
        self.verbosity_push()
        verbosity = self.max_verbosity_depth - self.curr_verbosity_depth + 1
        if self.parallel:
            self.pool.apply_async(get_metrics_for_params_parallel, 
                                  args=(i_parameterisation, i_repetition,
                                        self.scenarios, self.optional_assumptions, 
                                        params, params_k),
                                  kwds={'i_scenario_variation': i_scenario_variation,
                                        'verbosity': verbosity,
                                        'report_prefix': self.get_report_prefix()},
                                  callback = self.receive_metrics_for_params)
            self.verbosity_pop(2)
        else:
            metrics = get_metrics_for_params(
                self.scenarios, self.optional_assumptions, 
                self.params, self.params_k,
                i_scenario_variation=i_scenario_variation,
                verbosity=verbosity,
                report_prefix=self.get_report_prefix())
            self.verbosity_pop(2)
            return metrics
            
    
    def __init__(self, name, scenarios, optional_assumptions, 
                 default_params, default_params_k, param_arrays, 
                 n_repetitions=1, parallel=False, n_workers=mp.cpu_count()-1,
                 verbosity=0):
        """
        Construct and run a parameter search for the SCPaper project.

        Parameters
        ----------
        name : str
            A name for the parameter search.
        scenarios : dict with values of SCPaperScenario type
            The scenarios to simulate for each parameterisation. If these have
            kinematical variations, the number of variations must be the same 
            for all scenarios, and must equal the n_repetitions parameter.
        optional_assumptions : dict with keys of sc_scenario.OptionalAssumption 
                               and bool values.
            The optional assumptions for the model to be searched.
        default_params : commotions.Parameters
            Default parameter values, except value function gains.
        default_params_k : dict of commotions.Parameters with 
                           sc_scenario_helper.CtrlType as keys
            Deafult value function gain parameters.
        param_arrays : dict with str keys and iterable values
            Definition of the parameter grid to search. Only those parameters 
            relevant given optional_assumptions will be included in the grid.
        n_repetitions : int, optional
            The number of times to test each parameterisation. If the scenarios
            have kinematical variations, this number must equal the number of
            variations, which has to be the same across scenarios. The default is 1.
        verbosity : int, optional
            Verbosity - see parameter_search.ParameterSearch. The default is 0.

        Returns
        -------
        None.

        """
        # go through the list of scenarios to (1) check whether there are 
        # kinematic variations, and if so that the number of variations is
        # shared across scenarios, equal to n_repetitions, and (2) build the 
        # list of metrics to calculate in this search, from the scenario 
        # metric information
        self.scenarios = scenarios
        metric_names = []
        n_ind_scenario_variations = np.full(len(scenarios), -1)
        for i_scenario, scenario in enumerate(self.scenarios.values()):
            n_ind_scenario_variations[i_scenario] = scenario.n_variations
            for metric_name in scenario.full_metric_names:
                metric_names.append(metric_name)
        self.n_scenario_variations = np.amax(n_ind_scenario_variations)
        if self.n_scenario_variations > 1:
            if not np.all(n_ind_scenario_variations == n_repetitions):
                raise Exception('If there are scenario variations, the number has' 
                                'to be equal across scenarios, and equal to n_repetitions.')
        # make local copies of the default params objects, to use for
        # parameterising simulations during the fitting
        self.optional_assumptions = copy.copy(optional_assumptions)
        self.params = copy.copy(default_params)
        self.params_k = copy.deepcopy(default_params_k)
        # parse the optional assumptions and get the list of free parameter
        # names for this fit, and build the corresponding list of parameter 
        # value arrays
        free_param_names = []
        free_param_arrays = {}
        def consider_adding_free_param(param_name):
            if param_name in param_arrays:
                free_param_names.append(param_name)
                free_param_arrays[param_name] = param_arrays[param_name]
        # action/prediction parameters
        consider_adding_free_param('T_P')
        consider_adding_free_param('DeltaT')
        # value function parameters
        consider_adding_free_param('V_0_rel')
        consider_adding_free_param('V_ny')
        consider_adding_free_param('k_g')
        consider_adding_free_param('k_dv')
        if optional_assumptions[OptionalAssumption.oVA]:
            # affordance-based value functions
            consider_adding_free_param('T_delta')
            if optional_assumptions[OptionalAssumption.oVAl]:
                consider_adding_free_param('thetaDot_0')
                consider_adding_free_param('thetaDot_1')
        else:
            # non-affordance-based value functions
            consider_adding_free_param('k_c')
            consider_adding_free_param('k_e')
            consider_adding_free_param('k_sc')
            consider_adding_free_param('k_sg')
        # evidence accumulation
        if optional_assumptions[OptionalAssumption.oEA]:
            consider_adding_free_param('T')
            consider_adding_free_param('DeltaV_th_rel')
        # value-based behaviour estimation
        if optional_assumptions[OptionalAssumption.oBEv]:
            consider_adding_free_param('beta_V')
        # observation-based behaviour estimation
        if optional_assumptions[OptionalAssumption.oBEo]:
            consider_adding_free_param('beta_O')
            consider_adding_free_param('T_O1')
            consider_adding_free_param('T_Of')
            consider_adding_free_param('sigma_O')
        # check for unsupported assumptions
        unsupported_assumptions = (OptionalAssumption.oAN,)
        for unsupp in unsupported_assumptions:
            if optional_assumptions[unsupp]:
                raise Exception(f'Found unsupported assumption: {unsupp}')
        # call inherited constructor
        super().__init__(tuple(free_param_names), tuple(metric_names), 
                         name=name, n_repetitions=n_repetitions, 
                         parallel=parallel, n_workers=n_workers,
                         verbosity=verbosity)
        # run the grid search
        self.search_grid(free_param_arrays)
        # save the results
        self.save(FIT_RESULTS_FOLDER + (DET_FIT_FILE_NAME_FMT % name))
        
        
# unit testing
if __name__ == "__main__":
        
    plt.close('all')
    
    MODEL = ''
    
    PARAM_ARRAYS = {}
    PARAM_ARRAYS['k_c'] = (0.2, 2)
    PARAM_ARRAYS['k_sc'] = (0.2,)

    OPTIONAL_ASSUMPTIONS = sc_scenario.get_assumptions_dict_from_string(MODEL)
    test_fit = SCPaperParameterSearch('test', ONE_AG_SCENARIOS,
                                      OPTIONAL_ASSUMPTIONS, 
                                      DEFAULT_PARAMS, 
                                      get_default_params_k(MODEL), 
                                      PARAM_ARRAYS, 
                                      n_repetitions=N_ONE_AG_SCEN_VARIATIONS,
                                      parallel=True,
                                      verbosity=3)
    
    