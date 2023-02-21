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
import pickle
from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from statsmodels.distributions.empirical_distribution import ECDF
import commotions
import parameter_search
from sc_scenario import CtrlType, OptionalAssumption
import sc_scenario
import sc_scenario_helper
import sc_scenario_perception
import sc_plot

# expecting a results subfolder in the folder where this file is located
SCPAPER_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
FIT_RESULTS_FOLDER = SCPAPER_PATH + 'results/'
DATA_FOLDER = SCPAPER_PATH + 'data/'

# file names
HIKER_DATA_FILE_NAME = 'HIKERData.pkl'
DET_FIT_FILE_NAME_FMT = 'DetFit_%s.pkl'
ALT_SHORTSTOP_FIT_FILE_NAME_FMT = 'AltShortStopFit_%s.pkl'
PROB_FIT_FILE_NAME_FMT = 'ProbFit_%s.pkl'
HESITATION_CHECK_FNAME = 'HesitationChecks.pkl'
COMB_FIT_FILE_NAME_FMT = 'CombFit_%s.pkl'
HIKER_FIT_FILE_NAME_FMT = 'HIKERFit_%s.pkl'
RETAINED_DET_FNAME ='RetainedDetModels.pkl'
RETAINED_PROB_FNAME ='RetainedProbModels.pkl'
RETAINED_COMB_FNAME ='RetainedCombModels.pkl'
MODEL_CIT_FNAME_FMT = 'HIKER_CITs_%s.pkl'
EXCL_HIKER_FNAME = 'ExclParams_HIKER.pkl'
EXCL_DSS_FNAME = 'ExclParams_DSS.pkl'
EXCL_IO_FNAME = 'ExclParams_IO.pkl'

# scenario basics
N_AGENTS = 2
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
DET_SIM_TIME_STEP = 0.1 # s
PROB_SIM_TIME_STEP = 0.025 # s
DET_SIM_END_TIME = 8 # s
PROB_SIM_END_TIME = 12 # s
V_NY_REL = -1.5
PRIOR_DIST_SD_MULT = 2
PRIOR_SPEED_SD_MULT = 2
AGENT_COLL_DISTS = []
for i_ag in range(N_AGENTS):
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
DEFAULT_PARAMS.T_delta = 40
DEFAULT_PARAMS.H_e = 1.5
DEFAULT_PARAMS.sigma_xdot = 0.1
DEFAULT_PARAMS.c_tau = 0.001
DEFAULT_PARAMS.T_P = 0.5
DEFAULT_PARAMS.T_s = 1
DEFAULT_PARAMS.D_s = 1
DEFAULT_PARAMS.thetaDot_0 = 0
DEFAULT_PARAMS.beta_O = 1 
DEFAULT_PARAMS.T_O1 = 0.05 
#DEFAULT_PARAMS.DeltaV_th_rel = 0.001 
DEFAULT_PARAMS.DeltaT = 0.5 
DEFAULT_PARAMS.V_0_rel = 4 
DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.5, 0, 0.5, 1]) 
#DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]) 


@dataclass
class ModelWithParams:
    """ A class for storing information on a model and a set of 
        parameterisations for it.
        
        Properties
        ----------
            model: str
                A string identifier for the model, e.g., oVAoBEvoAI.
            param_names: list
                A list of strings, names of the parameters of this model.
            n_params: int
                Number of parameters, equal to len(param_names).
            params_array:
                A numpy array with one column for each parameter in param_names,
                and one row for each parameterisation for the model.
            n_parameterisations: int
                Number of parameterisations, equal to params_array.shape[0].
            param_ranges:
                A list with one tuple for each parameter in param_names,
                indicating the lower and upper bound of the original parameter
                ranges searched for the parameter.
    """
    model: str
    param_names: list
    param_ranges: list
    params_array: np.ndarray
    def get_params_dict(self, idx_parameterisation):
        """ Return parameterisation number idx_parameterisation from
            self.params_array, as a dict with parameter names as dict keys and
            parameter values as dict values.
        """
        return dict(zip(self.param_names, 
                        self.params_array[idx_parameterisation, :]))
    def __post_init__(self):
        self.n_parameterisations = self.params_array.shape[0]
        self.n_params = self.params_array.shape[1]
        assert(self.n_params == len(self.param_names))
        
        

# class for defining scenarios to simulate
class SCPaperScenario:
    
    def get_full_metric_name(self, short_metric_name):
        return self.name + '_' + short_metric_name
    
    def get_dists_and_accs(self, i_variation=None, force_calc=False):
        """
        Get the initial distances and any constant accelerations for the
        scenario, typically only used outside of this class if the scenario
        has multiple kinematic variations.

        Parameters
        ----------
        i_variation : TYPE, optional
            Index of the kinematic scenario variation, should be None if the 
            scenario doesn't have any variations. The default is None.
        force_calc : TYPE, optional
            Set to True to force recalculation of the values to be returned 
            regardless if already stored internally. The default is False.

        Returns
        -------
        tuple
            Initial distances of the two agents.
        tuple
            Constant accelerations of the two agents (None if no constant
            acceleration for the agent).

        """
        if self.n_variations == 1 and not force_calc:
            # should already have been calculated in call from the constructor
            return self.initial_cp_distances, self.const_accs
        # we need to do the calculations
        # - get the initial TTCAs for the two agents, in this scenario variation
        initial_ttcas = []
        for i_agent in range(N_AGENTS):
            if i_agent == self.i_varied_agent:
                if i_variation == None:
                    raise Exception('Need to specify scenario variation.')
                initial_ttcas.append(self.initial_ttcas[i_agent][i_variation])
            else:
                initial_ttcas.append(self.initial_ttcas[i_agent])
        # - get the corresponding initial distances
        initial_cp_distances = (np.array(initial_ttcas) * self.initial_speeds 
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
            veh_dist_at_yield_start = (initial_cp_distances[i_VEH_AGENT] -
                                       self.veh_yielding_start_time
                                       * self.initial_speeds[i_VEH_AGENT])
            stop_dist = (veh_dist_at_yield_start 
                         - AGENT_COLL_DISTS[i_VEH_AGENT] 
                         - self.veh_yielding_margin)
            const_accs[i_VEH_AGENT] = ((self.veh_yielding_start_time, 
                -self.initial_speeds[i_VEH_AGENT] ** 2 / (2 * stop_dist)),)
        return initial_cp_distances, const_accs
        
    
    def __init__(self, name, initial_ttcas, ped_prio=False,
                 ped_start_standing=False, ped_standing_margin=COLLISION_MARGIN,
                 ped_const_speed=False, veh_const_speed=False, 
                 ped_initial_speed=AGENT_FREE_SPEEDS[i_PED_AGENT],
                 veh_initial_speed=AGENT_FREE_SPEEDS[i_VEH_AGENT],
                 veh_yielding=False, veh_yielding_start_time=0,
                 veh_yielding_margin=COLLISION_MARGIN,
                 inhibit_first_pass_before_time=None,
                 time_step=DET_SIM_TIME_STEP, end_time=DET_SIM_END_TIME,
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
        self.veh_yielding_start_time = veh_yielding_start_time
        self.veh_yielding_margin = veh_yielding_margin
        self.inhibit_first_pass_before_time = inhibit_first_pass_before_time
        self.time_step = time_step
        self.end_time = end_time
        self.stop_criteria = stop_criteria
        # figure out if there are any kinematic variations to consider
        self.initial_ttcas = initial_ttcas
        self.n_variations = 1
        self.i_varied_agent = None
        for i_agent in range(N_AGENTS):
            if type(initial_ttcas[i_agent]) == tuple:
                if not (self.i_varied_agent == None):
                    raise Exception('Kinematic variations for more than one agent not supported.')
                self.i_varied_agent = i_agent
                self.n_variations = len(initial_ttcas[i_agent])  
        # get and store initial speeds
        self.initial_speeds = np.copy(AGENT_FREE_SPEEDS)
        self.initial_speeds[i_VEH_AGENT] = veh_initial_speed
        if ped_start_standing:
            self.initial_speeds[i_PED_AGENT] = 0     
        else:  
            self.initial_speeds[i_PED_AGENT] = ped_initial_speed
        # set initial distances and constant accelerations here only if 
        # scenario has just a single variation
        if self.n_variations == 1:
            self.initial_cp_distances, self.const_accs = \
                self.get_dists_and_accs(i_variation=1, force_calc=True)
        # store metric info
        self.short_metric_names = metric_names
        if metric_names != None:
            self.full_metric_names = []
            for short_metric_name in self.short_metric_names:
                self.full_metric_names.append(
                    self.get_full_metric_name(short_metric_name))
        
    
# scenarios
# - just defining some shorter names for the simulation stopping criteria
HALFWAY_STOP = (sc_scenario.SimStopCriterion.ACTIVE_AGENT_HALFWAY_TO_CS,)
STOPPED_STOP = (sc_scenario.SimStopCriterion.BOTH_AGENTS_STOPPED,)
IN_CS_STOP = (sc_scenario.SimStopCriterion.ACTIVE_AGENT_IN_CS,)
MOVED_STOP = (sc_scenario.SimStopCriterion.BOTH_AGENTS_HAVE_MOVED,)
EXITED_STOP = (sc_scenario.SimStopCriterion.BOTH_AGENTS_EXITED_CS,)
# - one-agent scenarios for deterministic fits
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
                                                       stop_criteria = MOVED_STOP,
                                                       metric_names = ('veh_speed_at_ped_start',))

# - alternative scenario for testing short-stopping
ALT_SHORTSTOP_SCENARIO = SCPaperScenario('VehShortStopAlt', 
                                        initial_ttcas=(math.nan, (3.5, 4, 4.5)),
                                        ped_prio=True,
                                        ped_start_standing=True, 
                                        ped_const_speed=True,
                                        stop_criteria = (STOPPED_STOP 
                                                         + IN_CS_STOP),
                                        metric_names = ('veh_av_surpl_dec',
                                                        'veh_stop_margin'),
                                        end_time = 12)

# - scenarios for the probabilistic fits
N_PROB_SCEN_REPETITIONS = 5
TWO_AG_METRIC_NAMES = ('collision', 'ped_exit_time', 'veh_exit_time')
PROB_FIT_SCENARIOS = {}
PROB_FIT_SCENARIOS['Encounter'] = SCPaperScenario('Encounter', 
                                                  initial_ttcas=(3, 3), 
                                                  stop_criteria = EXITED_STOP,
                                                  metric_names = TWO_AG_METRIC_NAMES,
                                                  time_step = PROB_SIM_TIME_STEP,
                                                  end_time = PROB_SIM_END_TIME)
PROB_FIT_SCENARIOS['EncounterPedPrio'] = SCPaperScenario('EncounterPedPrio', 
                                                         initial_ttcas=(3, 3), 
                                                         ped_prio=True,
                                                         stop_criteria = EXITED_STOP,
                                                         metric_names = TWO_AG_METRIC_NAMES,
                                                         time_step = PROB_SIM_TIME_STEP,
                                                         end_time = PROB_SIM_END_TIME)
PROB_FIT_SCENARIOS['PedLead'] = SCPaperScenario('PedLead', 
                                                initial_ttcas=(3, 8), 
                                                stop_criteria = EXITED_STOP,
                                                metric_names = TWO_AG_METRIC_NAMES,
                                                time_step = PROB_SIM_TIME_STEP,
                                                end_time = PROB_SIM_END_TIME)
PROB_FIT_SCENARIOS['PedHesitateVehConst'] = SCPaperScenario('PedHesitateVehConst', 
                                                            initial_ttcas=(3, 8), 
                                                            veh_const_speed=True,
                                                            stop_criteria = IN_CS_STOP,
                                                            metric_names = ('ped_av_speed_to_CS',),
                                                            time_step = PROB_SIM_TIME_STEP,
                                                            end_time = PROB_SIM_END_TIME)


# HIKER scenarios

HIKER_VEH_SPEEDS_MPH = (25, 30, 35) # miles per hour
HIKER_VEH_TIME_GAPS = (2, 3, 4, 5) # s
HIKER_FIRST_VEH_PASSING_TIME = 3
HIKER_YIELD_START_DIST = 38.5 # m from front bumper to conflict point
HIKER_YIELD_END_DIST = 2.5 # m from front bumper to conflict point
HIKER_SIM_TIME_AFTER_VEH_STOP = 3 # if ped not moving 3 s after the car has yielded to a stop, terminate simulation
HIKER_CIT_METRIC_NAME = 'hiker_cit'

def get_hiker_scen_name(veh_speed_mph, veh_time_gap, veh_yielding):
    base = str(veh_speed_mph) + '_' + str(veh_time_gap)
    if veh_yielding:
        return base + '_y'
    else:
        return base
    
class HIKERScenario(SCPaperScenario):
    
    def __init__(self, name, veh_speed_mph, veh_time_gap, veh_yielding):
        half_ped_width = AGENT_WIDTHS[i_PED_AGENT] / 2
        veh_speed = veh_speed_mph * 1.609 / 3.6
        # get initial time for the second vehicle to conflict area 
        # - the edge of the conflict area is half a pedestrian width closer to
        # the second vehicle than the back of the first vehicle when the second
        # vehicle is passing the pedestrian
        veh_initial_ttca = (HIKER_FIRST_VEH_PASSING_TIME + veh_time_gap 
                            - half_ped_width / veh_speed)
        if veh_yielding:
            veh_const_speed = False
            # get distance to conflict area at time of yielding for the second veh.
            veh_dist_to_ca_at_yield = HIKER_YIELD_START_DIST - half_ped_width
            # now we can calculate the time to conflict area at yielding start
            # and thus also the time in the simulation when yielding should start
            veh_ttca_at_yield = veh_dist_to_ca_at_yield / veh_speed
            veh_yielding_start_time = veh_initial_ttca - veh_ttca_at_yield
            # ancestor class defines yielding margin as actual physical gap,
            # whereas HIKER scenario is again defined with ref to ped centre
            veh_yielding_margin = HIKER_YIELD_END_DIST - half_ped_width
            # calculate stopping deceleration, just so we can calculate a
            # sensible simulation end time
            veh_stop_dec = veh_speed ** 2 / (2 * (HIKER_YIELD_START_DIST 
                                                  - HIKER_YIELD_END_DIST))
            veh_stop_time = veh_yielding_start_time + veh_speed / veh_stop_dec
            end_time = veh_stop_time + HIKER_SIM_TIME_AFTER_VEH_STOP
        else: 
            veh_const_speed = True
            veh_yielding_start_time = None
            veh_yielding_margin = None
            end_time = veh_initial_ttca + 0.5 # just to definitely not end before the vehicle enters the conflict space
        super().__init__(
            name, initial_ttcas=(math.nan, veh_initial_ttca), ped_prio=False,
            ped_start_standing=True, ped_standing_margin=COLLISION_MARGIN,
            veh_const_speed=veh_const_speed, veh_initial_speed=veh_speed,
            veh_yielding=veh_yielding, 
            veh_yielding_start_time=veh_yielding_start_time,
            veh_yielding_margin=veh_yielding_margin,
            inhibit_first_pass_before_time=HIKER_FIRST_VEH_PASSING_TIME,
            time_step=PROB_SIM_TIME_STEP, end_time=end_time,
            stop_criteria = (sc_scenario.SimStopCriterion.AGENT_IN_CS,), 
            metric_names = (HIKER_CIT_METRIC_NAME,))

HIKER_SCENARIOS = {}
for veh_speed_mph in HIKER_VEH_SPEEDS_MPH:
    for veh_time_gap in HIKER_VEH_TIME_GAPS:
        for veh_yielding in (False, True):
            name = get_hiker_scen_name(veh_speed_mph, veh_time_gap, veh_yielding)
            HIKER_SCENARIOS[name] = HIKERScenario(name, veh_speed_mph, 
                                                  veh_time_gap, veh_yielding)


# implementations of metrics for analysing model simulation results

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

def metric_ped_av_speed_to_CS(sim):
    ped_agent = sim.agents[i_PED_AGENT]
    idx_past_CS_entry = np.nonzero(ped_agent.signed_CP_dists <= ped_agent.coll_dist)[0]
    if len(idx_past_CS_entry) == 0:
        return math.nan
    else:
        return np.mean(ped_agent.trajectory.long_speed[:idx_past_CS_entry[0]])

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
    
def metric_veh_stop_margin(sim):
    veh_agent = sim.agents[i_VEH_AGENT]
    idx_veh_stopped = np.nonzero(veh_agent.trajectory.long_speed <= 0.5)[0]
    if len(idx_veh_stopped) == 0:
        return math.nan
    else:
        return veh_agent.signed_CP_dists[idx_veh_stopped[0]] - veh_agent.coll_dist

def metric_veh_speed_at_ped_start(sim):
    idx_ped_moving = np.nonzero(sim.agents[i_PED_AGENT].trajectory.long_speed > 0)[0]
    if len(idx_ped_moving) == 0:
        return math.nan
    else:
        return sim.agents[i_VEH_AGENT].trajectory.long_speed[idx_ped_moving[0]]

def metric_collision(sim):
    collision_samples = np.full(len(sim.time_stamps), True)
    for agent in sim.agents:
        collision_samples = collision_samples & (
            np.abs(agent.signed_CP_dists) < agent.coll_dist)
    return np.any(collision_samples)

def metric_agent_entry_time(sim, i_agent):
    past_cs_entrance_samples = np.nonzero(sim.agents[i_agent].signed_CP_dists 
                                   <= sim.agents[i_agent].coll_dist)[0]
    if len(past_cs_entrance_samples) == 0:
        return math.nan
    else:
        return sim.time_stamps[past_cs_entrance_samples[0]]

def metric_ped_entry_time(sim):
    return metric_agent_entry_time(sim, i_PED_AGENT)

def metric_veh_entry_time(sim):
    return metric_agent_entry_time(sim, i_VEH_AGENT)

def metric_agent_exit_time(sim, i_agent):
    beyond_cs_samples = np.nonzero(sim.agents[i_agent].signed_CP_dists 
                                   < -sim.agents[i_agent].coll_dist)[0]
    if len(beyond_cs_samples) == 0:
        return math.nan
    else:
        return sim.time_stamps[beyond_cs_samples[0]]

def metric_ped_exit_time(sim):
    return metric_agent_exit_time(sim, i_PED_AGENT)

def metric_veh_exit_time(sim):
    return metric_agent_exit_time(sim, i_VEH_AGENT)

def metric_hiker_cit(sim):
    nonzero_ped_speed_samples = np.nonzero(sim.agents[
        i_PED_AGENT].trajectory.long_speed)[0]
    if len(nonzero_ped_speed_samples) == 0:
        # pedestrian didn't move before end of simulation
        return math.nan
    if sim.first_passer is sim.agents[i_VEH_AGENT]:
        # pedestrian moved, but vehicle entered conflict space first
        return math.nan
    return (sim.time_stamps[nonzero_ped_speed_samples[0]] 
            - HIKER_FIRST_VEH_PASSING_TIME)


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
                      i_variation=None, snapshots=(None, None), 
                      detailed_snapshots=False, noise_seeds=(None, None), 
                      zero_acc_after_exit=True, apply_stop_criteria=True):
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
    # - Kalman priors needed?
    if optional_assumptions[OptionalAssumption.oPF]:
        kalman_priors = []
        for i_agent in range(N_AGENTS):
            i_oth = 1 - i_agent
            oth_init_dist = initial_cp_dists[i_oth]
            oth_free_speed = AGENT_FREE_SPEEDS[i_oth]
            kalman_priors.append(sc_scenario_perception.KalmanPrior(
                cp_dist_mean = oth_init_dist, 
                cp_dist_stddev = PRIOR_DIST_SD_MULT * oth_init_dist, 
                speed_mean = oth_free_speed,
                speed_stddev = PRIOR_SPEED_SD_MULT * oth_free_speed))
    else:
        kalman_priors = (None, None)
    # - stop criteria?
    if apply_stop_criteria:
        stop_criteria = scenario.stop_criteria
    else:
        stop_criteria = ()
    
    # - set up the SCSimulation object
    sc_simulation = sc_scenario.SCSimulation(
        ctrl_types=AGENT_CTRL_TYPES, 
        widths=AGENT_WIDTHS, lengths=AGENT_LENGTHS, 
        goal_positions=AGENT_GOALS, initial_positions=initial_pos, 
        initial_speeds=scenario.initial_speeds, 
        const_accs=const_accs, zero_acc_after_exit=zero_acc_after_exit,
        start_time=0, end_time=scenario.end_time, time_step=scenario.time_step, 
        optional_assumptions=optional_assumptions, 
        params=params, params_k=params_k, kalman_priors=kalman_priors,
        inhibit_first_pass_before_time=scenario.inhibit_first_pass_before_time,
        noise_seeds=noise_seeds, agent_names=AGENT_NAMES, 
        snapshot_times=snapshots, detailed_snapshots=detailed_snapshots,
        stop_criteria=stop_criteria)
    
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


def set_params(params_obj, params_k_obj, params_dict):
    """
    Update params_obj and params_k_obj with values from params_dict.

    Parameters
    ----------
    params_obj : commotions.Parameters
        Has attributes corresponding to sc_scenario.SCAgent parameters.
    params_k_obj : dict with sc_scenario.CtrlType as keys and 
                   commotions.Parameters values
        Each value has attributes corresponding to sc_scenario.SCAgent value 
        function gain parameters.
    params_dict : dict with string parameter names (starting with 'k_' for
                  value function gain parameters) and parameter values
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
                setattr(params_k_obj[ctrl_type], param_attr, param_value)
        else:
            # other parameter
            setattr(params_obj, param_name, param_value)
            

def construct_model_and_simulate_scenario(
        model_name, params_dict, scenario, i_variation=None, 
        snapshots=(None, None), detailed_snapshots=False, 
        noise_seeds=(None, None), zero_acc_after_exit=True, 
        apply_stop_criteria=True, report_time=False):
    """ Convenience wrapper for simulate_scenario(), which first builds the model
        and parameterisation from model_name, the sc_fitting default parameters,
        and params_dict.
    """
    assumptions = sc_scenario.get_assumptions_dict_from_string(model_name)
    params = copy.deepcopy(DEFAULT_PARAMS)
    params_k = copy.deepcopy(get_default_params_k(model_name))
    set_params(params, params_k, params_dict)
    tic = time.perf_counter()
    return simulate_scenario(scenario, assumptions, params, params_k,
                             i_variation=i_variation, snapshots=snapshots, 
                             detailed_snapshots=detailed_snapshots, 
                             noise_seeds=noise_seeds, 
                             zero_acc_after_exit=zero_acc_after_exit,
                             apply_stop_criteria=apply_stop_criteria)
    toc = time.perf_counter()
    if report_time:
        print('Initialising and running simulation took %.3f s.' % (toc - tic,))


def run_dummy_prob_sim(verbose=True):
    """ Run a probabilistic simulation - doing this before starting parallelised
        fits on the ARC4 cluster prevents a problem with parallelisation for
        somewhat unclear reasons (see 2022-01-19 diary notes).
    """
    if verbose:
        print('Running one dummy probabilistic simulation')
    scenario = PROB_FIT_SCENARIOS['Encounter']
    sim = construct_model_and_simulate_scenario(
        'oVAoEAoSNvoPF', {'T': 0.2, 'DeltaV_th_rel': 0.01, 'tau_theta': 0.1}, 
        scenario)
    if verbose:
        get_metrics_for_scenario(scenario, sim, verbose=True)
        

def save_results(results, file_name, verbose=True):
    file_path = FIT_RESULTS_FOLDER + file_name
    if verbose:
        print(f'Saving "{file_path}"...')
    with open(file_path, 'wb') as file_obj:
        pickle.dump(results, file_obj)
    if verbose:
        print('\tDone.')


def load_results(file_name, verbose=True):
    file_path = FIT_RESULTS_FOLDER + file_name
    if verbose:
        print(f'Loading "{file_path}"...')
    with open(file_path, 'rb') as file_obj:
        results = pickle.load(file_obj)
        if verbose:
            print('\tDone.')
        return results
    
    
def results_exist(file_name, verbose=False):
    file_path = FIT_RESULTS_FOLDER + file_name
    exists = os.path.exists(file_path)
    if verbose:
        if exists:
            print(f'Found {file_path}')
        else:
            print(f'Did not find {file_path}')
    return exists


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
        set_params(self.params, self.params_k, params_dict)
    
    
    def simulate_scenario(self, scenario, i_variation='all', 
                          snapshots=(None, None), detailed_snapshots=False,
                          zero_acc_after_exit=True, apply_stop_criteria=True):
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
                                     snapshots=snapshots, 
                                     detailed_snapshots=detailed_snapshots,
                                     zero_acc_after_exit=zero_acc_after_exit,
                                     apply_stop_criteria=apply_stop_criteria)
        else:
            if i_variation == 'all':
                sims = []
                for i_var in range(self.n_scenario_variations):
                    sims.append(simulate_scenario(scenario, 
                                                  self.optional_assumptions, 
                                                  self.params, self.params_k, 
                                                  i_variation=i_var, 
                                                  snapshots=snapshots,
                                                  detailed_snapshots=detailed_snapshots,
                                                  zero_acc_after_exit=zero_acc_after_exit,
                                                  apply_stop_criteria=apply_stop_criteria))
                return sims
            else:
                return simulate_scenario(scenario, self.optional_assumptions, 
                                         self.params, self.params_k, 
                                         i_variation=i_variation, 
                                         snapshots=snapshots,
                                         detailed_snapshots=detailed_snapshots,
                                         zero_acc_after_exit=zero_acc_after_exit,
                                         apply_stop_criteria=apply_stop_criteria)
    
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
                 list_search=False, n_repetitions=1, 
                 parallel=False, n_workers=mp.cpu_count()-1,
                 verbosity=0, file_name_format=DET_FIT_FILE_NAME_FMT,
                 overwrite_existing=True):
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
            Definition of the parameter values to search. If list_search is
            False (default), the iterables can be of different length, and the
            values will be used to generate a parameter value grid to search. 
            If list_search is True, the iterables all need to be of the same
            length and will be interpreted as defining a list of
            parameterisations to search. In both cases, only those parameters 
            relevant given optional_assumptions will be included in the grid/list.
        list_search : bool, optional
            If True, param_arrays is interpreted as specifying one big list of 
            parameterisations to search (rather than as specifying one list of 
            values for each parameter separately, to be combined in a grid 
            search).
        n_repetitions : int, optional
            The number of times to test each parameterisation. If the scenarios
            have kinematical variations, this number must equal the number of
            variations, which has to be the same across scenarios. The default is 1.
        verbosity : int, optional
            Verbosity - see parameter_search.ParameterSearch. The default is 0.
        file_name_format : str, optional
            The file name to use for saving the results, including a %s for
            which the parameter search's name will be replaced.
            Default is sc_fitting.DET_FIT_FILE_NAME_FMT.
        overwrite_existing : bool, optional
            If False, aborts the parameter search if the output file already 
            exists. Default is True.

        Returns
        -------
        None.

        """
        # need to check if output file already exists?
        file_name = FIT_RESULTS_FOLDER + (file_name_format % name)
        if not overwrite_existing:
            if os.path.exists(file_name):
                print(f'Output file {file_name} already exists, not running'
                      f' parameter search for {name}.')
                return
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
        if list_search:
            free_param_arrays = []
        else:
            free_param_arrays = {}
        def consider_adding_free_param(param_name):
            if param_name in param_arrays:
                free_param_names.append(param_name)
                if list_search:
                    free_param_arrays.append(param_arrays[param_name])
                else:
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
        # value accumulation
        if optional_assumptions[OptionalAssumption.oEA]:
            consider_adding_free_param('T')
            consider_adding_free_param('DeltaV_th_rel')
        # value noise
        if optional_assumptions[OptionalAssumption.oAN]:
            consider_adding_free_param('sigma_V')
        # decision evidence accumulation
        if optional_assumptions[OptionalAssumption.oDA]:
            consider_adding_free_param('xi_th')
        # value-based behaviour estimation
        if optional_assumptions[OptionalAssumption.oBEv]:
            consider_adding_free_param('beta_V')
        # observation-based behaviour estimation
        if optional_assumptions[OptionalAssumption.oBEo]:
            consider_adding_free_param('beta_O')
            consider_adding_free_param('T_O1')
            consider_adding_free_param('T_Of')
            consider_adding_free_param('sigma_O')
        # noisy perception
        if (optional_assumptions[OptionalAssumption.oSNv] 
            or optional_assumptions[OptionalAssumption.oSNc]):
            consider_adding_free_param('c_tau')
        if optional_assumptions[OptionalAssumption.oSNv]:
            consider_adding_free_param('H_e')
            consider_adding_free_param('tau_theta')
        if optional_assumptions[OptionalAssumption.oSNc]:
            consider_adding_free_param('tau_d')
        if optional_assumptions[OptionalAssumption.oPF]:
            consider_adding_free_param('sigma_xdot')
            
        # check for unsupported assumptions
        unsupported_assumptions = ()
        for unsupp in unsupported_assumptions:
            if optional_assumptions[unsupp]:
                raise Exception(f'Found unsupported assumption: {unsupp}')
        # call inherited constructor
        super().__init__(tuple(free_param_names), tuple(metric_names), 
                         name=name, n_repetitions=n_repetitions, 
                         parallel=parallel, n_workers=n_workers,
                         verbosity=verbosity)
        # run the search
        if list_search:
            free_params_matrix =  np.array(free_param_arrays).T
            self.search_list(free_params_matrix)
        else:
            self.search_grid(free_param_arrays)
        # save the results
        self.save(file_name)
        

        
def do_params_plot(param_names, params_array, param_ranges=None, 
                   log=True, jitter=0, param_subsets=None, color='k', show=True,
                   do_alpha=True, model_name=''):
    def get_plot_lims(minv, maxv):
        ZOOM = 0.1
        if log:
            minp = minv * (maxv/minv) ** (-ZOOM)
            maxp = maxv * (maxv/minv) ** (ZOOM)
        else:
            minp = minv - (maxv-minv) * ZOOM
            maxp = maxv + (maxv-minv) * ZOOM
        return minp, maxp
    N_PARAM_BINS = 10
    PARAM_VAL_FOR_INF = 10 # for T_Of = Inf
    n_params = len(param_names)
    assert(params_array.shape[1] == n_params)
    # if oPF model, make sure to scale noise magnitude properly in plot
    if 'oPF' in model_name:
        if 'oSNc' in model_name:
            noise_param_name = 'tau_d'
        elif 'oSNv' in model_name:
            noise_param_name = 'tau_theta'
        else:
            raise Exception('Found oPF model without oSN*.')
        idx_noise_param = param_names.index(noise_param_name)
        params_array = np.copy(params_array)
        params_array[:, idx_noise_param] *= DEFAULT_PARAMS.c_tau
        if param_ranges != None:
            param_ranges = np.copy(param_ranges)
            param_ranges[idx_noise_param] = (
                np.array(param_ranges[idx_noise_param]) * DEFAULT_PARAMS.c_tau)
    # adapt plotting to number of parameters
    if n_params == 2:
        jitter = 0
        PARAM_ALPHA = 0.8
        PARAM_MS = 6
    elif n_params == 3:
        jitter = jitter / 2
        PARAM_ALPHA = 0.4
        PARAM_MS = 4
    else:
        PARAM_ALPHA = 0.2
        PARAM_MS = 4
    if do_alpha:
        HIST_ALPHA = 0.5
    else:
        PARAM_ALPHA = 1
        HIST_ALPHA = 1
    # get display param names
    display_param_names = []
    for param_name in param_names:
        display_param_names.append(sc_plot.get_display_param_name(param_name))
    # parse any parameter subsets
    if param_subsets == None:
        param_subsets = (np.arange(params_array.shape[0]),)
    # parse colors
    if not type(color) is tuple:
        color = (color,)
    assert(len(color) == len(param_subsets))
    # prepare figure
    figsize = min(12, 2 * n_params)
    fig, axs = plt.subplots(n_params, n_params, 
                            figsize=(figsize,figsize),
                            tight_layout=True)
    # get parameter ranges if not provided
    if param_ranges is None:
        param_ranges = []
        for i_param in range(n_params):
            param_ranges.append((np.amin(params_array[:, i_param]),
                                 np.amax(params_array[:, i_param])))
    # prepare for adding any jitter
    if jitter > 0:
        rng = np.random.default_rng()
    # loop through panels
    for i_x_param in range(n_params):
        for i_y_param in range(n_params):
            if n_params > 1:
                ax = axs[i_y_param, i_x_param]
            else:
                ax = axs
            if log:
                ax.set_xscale('log')
            xmin = param_ranges[i_x_param][0]
            xmax = param_ranges[i_x_param][1]
            ymin = param_ranges[i_y_param][0]
            ymax = param_ranges[i_y_param][1]
            if np.isinf(xmax):
                xmax = PARAM_VAL_FOR_INF
            xplotmin, xplotmax = get_plot_lims(xmin, xmax)
            ax.set_xlim(xplotmin, xplotmax)
            if i_x_param == i_y_param:
                max_bin_edge = xmax+(xmax-xmin)/N_PARAM_BINS
                if log:
                    bins = np.logspace(np.log10(xmin), np.log10(max_bin_edge),
                                       N_PARAM_BINS+1)
                else:
                    bins = np.linspace(xmin, max_bin_edge, N_PARAM_BINS+1)
                for i_subset, param_subset in enumerate(param_subsets):
                    ax.hist(params_array[param_subset, i_x_param], bins=bins, 
                            color=color[i_subset], alpha=HIST_ALPHA, ec=None)
            else:
                xdata = np.copy(params_array[:, i_x_param])
                ydata = np.copy(params_array[:, i_y_param])
                if log:
                    ax.set_yscale('log')
                if jitter > 0:
                    if log:
                        xdata *= (xmax/xmin) ** rng.normal(scale = jitter, 
                                                  size = xdata.shape)
                        ydata *= (ymax/ymin) ** rng.normal(scale = jitter, 
                                                  size = xdata.shape)
                    else:
                        xdata += rng.normal(scale = jitter * (xmax - xmin),
                                            size = xdata.shape)
                        ydata += rng.normal(scale = jitter * (ymax - ymin),
                                            size = xdata.shape)
                
                for i_subset, param_subset in enumerate(param_subsets):
                    ax.plot(xdata[param_subset], ydata[param_subset],'o', 
                            ms=PARAM_MS, alpha=PARAM_ALPHA, color=color[i_subset],
                            markeredgecolor='none')
                if np.isinf(ymax):
                    ymax = PARAM_VAL_FOR_INF
                yplotmin, yplotmax = get_plot_lims(ymin, ymax)
                ax.set_ylim(yplotmin, yplotmax)
            if param_names[i_x_param] == 'T':
                # fix for problem with overlapping x tick labels
                xticks = np.arange(xmin, xmax, 0.1)
                ax.set_xticks(xticks)
                xticklabels = []
                for i in range(len(xticks)):
                    xticklabels.append('')
                xticklabels[0] = f'{xticks[0]:.1f}'
                xticklabels[-1] = f'{xticks[-1]:.1f}'
                ax.set_xticklabels(xticklabels)
            if i_x_param == 0:
                ax.set_ylabel(display_param_names[i_y_param])
            if i_y_param == n_params-1:
                ax.set_xlabel(display_param_names[i_x_param]) 
    if show:
        plt.show()
   

def do_hiker_cit_cdf_plot(cit_data, fig_name='Crossing initiation CDFs', 
                          axs=None, xlabels=True, ylabels=True, titles=True,
                          legend=True, finalise=True, legend_kwargs={},
                          show_name_in_fig=False):
    
    def get_speed_alpha(i_speed):
        return (1 - float(i_speed)/(len(HIKER_VEH_SPEEDS_MPH)+1)) ** 2
    
    def get_yielding_color(veh_yielding):
        if veh_yielding:
            return 'green'
        else:
            return 'magenta'
        
    if axs == None:
        fig, axs = plt.subplots(nrows=1, ncols=len(HIKER_VEH_TIME_GAPS), 
                                sharex=True, sharey=True, num=fig_name,
                                figsize=(0.7*sc_plot.FULL_WIDTH, 
                                         0.25*sc_plot.FULL_WIDTH))
        
    for i_speed, veh_speed_mph in enumerate(HIKER_VEH_SPEEDS_MPH):
        for i_gap, veh_time_gap in enumerate(HIKER_VEH_TIME_GAPS):
            for i_yield, veh_yielding in enumerate((False, True)):
                scen_name = get_hiker_scen_name(veh_speed_mph,
                                                veh_time_gap,
                                                veh_yielding)
                ax = axs[i_gap]
                ecdf = ECDF(cit_data[scen_name]['crossing_time'])
                alpha = get_speed_alpha(i_speed)
                color = get_yielding_color(veh_yielding)
                ax.step(ecdf.x, ecdf.y, 'k-', lw=i_speed+1, color=color, 
                        alpha=alpha)
                ax.set_xlim(-1, 11)
                ax.set_ylim(-.1, 1.1)
            if titles:
                axs[i_gap].set_title(f'Gap {veh_time_gap} s')
            if xlabels:
                axs[i_gap].set_xlabel('Crossing onset time (s)')
    if ylabels:
        axs[0].set_ylabel('CDF (-)')   
    if legend:
        ax = axs[-1]
        leg_plots = []
        for i_speed, spd in enumerate(HIKER_VEH_SPEEDS_MPH):
            line, = ax.plot((-1, 0), (-10, -10), 'k', lw=i_speed+1, 
                                     alpha=get_speed_alpha(i_speed), label=f'{spd} mph')
            leg_plots.append(line)
        for veh_yielding in (False, True):
            if veh_yielding:
                label = 'Yielding'
            else:
                label = 'Const. speed'
            line, = ax.plot((-1, 0), (-10, -10), lw=2, 
                          color=get_yielding_color(veh_yielding),
                          alpha=get_speed_alpha(1), label=label)
            leg_plots.append(line)
        ax.legend(handles=leg_plots, **legend_kwargs)
    if show_name_in_fig:
        LABEL_LEFT = 0.03
        LABEL_BOTTOM = 0.93
        plt.annotate(fig_name, xy=(LABEL_LEFT, LABEL_BOTTOM), xycoords='figure fraction',
                     fontweight='bold', fontsize=sc_plot.PANEL_LABEL_FONT_SIZE)
        for i_ax, ax in enumerate(axs):
            ax.set_position([0.08 + i_ax*0.23, 0.17, 0.21, 0.62])
    if finalise:
        plt.tight_layout()
        plt.show()
     
    
    
# unit testing
if __name__ == "__main__":
        
    plt.close('all')
    
    
    if True:
    
        # # test scenario running
        # TEST_SCENARIO = SCPaperScenario(name='TestScenario', initial_ttcas=(6, 6),
        #                            veh_yielding=True, veh_yielding_start_time=2,
        #                            inhibit_first_pass_before_time=3,
        #                            time_step=PROB_SIM_TIME_STEP, end_time=15)   
        TEST_PARAMS = {'T_delta': 60}
        # sim = construct_model_and_simulate_scenario(model_name='oVA', 
        #                                             params_dict=TEST_PARAMS, 
        #                                             scenario=TEST_SCENARIO)
        # sim.do_plots(kinem_states=True)
        
        
        sim = construct_model_and_simulate_scenario(model_name='oVA', 
                                                    params_dict=TEST_PARAMS, 
                                                    scenario=HIKER_SCENARIOS['35_2_y'],
                                                    apply_stop_criteria=False,
                                                    snapshots=(None, None))
        sim.do_plots(kinem_states=True)
    
    
    if True:
    
        # test fitting functionality
        
        MODEL = ''
        
        PARAM_ARRAYS = {}
        PARAM_ARRAYS['k_c'] = (0.2, 2)
        PARAM_ARRAYS['k_sc'] = (0.2, 2)
    
        OPTIONAL_ASSUMPTIONS = sc_scenario.get_assumptions_dict_from_string(MODEL)
        
        # as grid, not parallel
        test_fit = SCPaperParameterSearch('test', ONE_AG_SCENARIOS,
                                          OPTIONAL_ASSUMPTIONS, 
                                          DEFAULT_PARAMS, 
                                          get_default_params_k(MODEL), 
                                          PARAM_ARRAYS, 
                                          n_repetitions=N_ONE_AG_SCEN_VARIATIONS,
                                          parallel=False,
                                          verbosity=5)
        
        # as grid, parallel
        test_fit = SCPaperParameterSearch('test', ONE_AG_SCENARIOS,
                                          OPTIONAL_ASSUMPTIONS, 
                                          DEFAULT_PARAMS, 
                                          get_default_params_k(MODEL), 
                                          PARAM_ARRAYS, 
                                          n_repetitions=N_ONE_AG_SCEN_VARIATIONS,
                                          parallel=True,
                                          verbosity=3)
    
        # as list, parallel
        test_fit = SCPaperParameterSearch('test', ONE_AG_SCENARIOS,
                                          OPTIONAL_ASSUMPTIONS, 
                                          DEFAULT_PARAMS, 
                                          get_default_params_k(MODEL), 
                                          PARAM_ARRAYS, list_search=True,
                                          n_repetitions=N_ONE_AG_SCEN_VARIATIONS,
                                          parallel=True,
                                          verbosity=3)
    
    
    