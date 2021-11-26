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
    DEFAULT_PARAMS_K_VA[ctrl_type]._da = 0.5
# - fixed value function gains, non-affordance-based
DEFAULT_PARAMS_K_NVA = {}
for ctrl_type in CtrlType:
    DEFAULT_PARAMS_K_NVA[ctrl_type] = commotions.Parameters()
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
    
    def __init__(self, name, initial_ttcas, ped_prio=False,
                 ped_start_standing=False, ped_standing_margin=COLLISION_MARGIN,
                 ped_const_speed=False, veh_const_speed=False, 
                 veh_yielding=False, veh_yielding_margin=COLLISION_MARGIN,
                 metric_names = None):
        self.name = name
        self.ped_prio = ped_prio
        self.initial_cp_distances = (np.array(initial_ttcas) * AGENT_FREE_SPEEDS 
                                     + np.array(AGENT_COLL_DISTS))
        self.initial_speeds = np.copy(AGENT_FREE_SPEEDS)
        self.const_accs = [None, None]
        if ped_start_standing:
            self.initial_cp_distances[i_PED_AGENT] = (
                AGENT_COLL_DISTS[i_PED_AGENT] + ped_standing_margin)
            self.initial_speeds[i_PED_AGENT] = 0
        if ped_const_speed:
            self.const_accs[i_PED_AGENT] = 0
        if veh_const_speed:
            self.const_accs[i_VEH_AGENT] = 0
        elif veh_yielding:
            stop_dist = (self.initial_cp_distances[i_VEH_AGENT] 
                         - AGENT_COLL_DISTS[i_VEH_AGENT] - veh_yielding_margin)
            self.const_accs[i_VEH_AGENT] = (
                -self.initial_speeds[i_VEH_AGENT] ** 2 / (2 * stop_dist))
        self.short_metric_names = metric_names
        self.full_metric_names = []
        for short_metric_name in self.short_metric_names:
            self.full_metric_names.append(
                self.get_full_metric_name(short_metric_name))
        
    
# scenarios
# - one-agent scenarios
ONE_AG_SCENARIOS = {}
ONE_AG_SCENARIOS['VehPrioAssert'] = SCPaperScenario('VehPrioAssert', 
                                                   initial_ttcas=(math.nan, 2),  
                                                   ped_start_standing=True, 
                                                   ped_const_speed=True,
                                                   metric_names = ('veh_av_speed',))
ONE_AG_SCENARIOS['VehShortStop'] = SCPaperScenario('VehShortStop', 
                                                  initial_ttcas=(math.nan, 4),
                                                  ped_prio = True,
                                                  ped_start_standing=True, 
                                                  ped_const_speed=True,
                                                  metric_names = ('veh_av_surpl_dec',))
ONE_AG_SCENARIOS['PedHesitateVehConst'] = SCPaperScenario('PedHesitateVehConst', 
                                                          initial_ttcas=(3, 8), 
                                                          veh_const_speed=True,
                                                          metric_names = ('ped_av_speed',))
ONE_AG_SCENARIOS['PedHesitateVehYield'] = SCPaperScenario('PedHesitateVehYield', 
                                                         initial_ttcas=(3, 3), 
                                                         ped_prio=True,
                                                         veh_yielding=True,
                                                         metric_names = ('ped_av_speed',))
ONE_AG_SCENARIOS['PedCrossVehYield'] = SCPaperScenario('PedCrossVehYield', 
                                                       initial_ttcas=(math.nan, 2), 
                                                       ped_prio=True,
                                                       ped_start_standing=True, 
                                                       veh_yielding=True,
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
    halfway_dist = (agent.signed_CP_dists[0] - agent.coll_dist) / 2
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
    idx_halfway = get_halfway_to_cs_sample(sim, i_VEH_AGENT)
    if math.isnan(idx_halfway):
        return math.nan
    else:
        veh_agent = sim.agents[i_VEH_AGENT]
        # get the decelerations needed to stop
        stop_dists = (veh_agent.signed_CP_dists[:idx_halfway] 
                      - veh_agent.coll_dist - COLLISION_MARGIN)
        stop_decs = (veh_agent.trajectory.long_speed[:idx_halfway] ** 2 
                     / (2 * stop_dists))
        # compare to actual decelerations
        actual_decs = -veh_agent.trajectory.long_acc[:idx_halfway]
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



def simulate_scenario(scenario, optional_assumptions, params, params_k, 
                      snapshots=(None, None)):
    # prepare the simulation
    # - initial position
    initial_pos = np.array([[0, -scenario.initial_cp_distances[i_PED_AGENT]],
                           [scenario.initial_cp_distances[i_VEH_AGENT], 0]])
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
        const_accs=scenario.const_accs,
        start_time=0, end_time=END_TIME, time_step=TIME_STEP, 
        optional_assumptions=optional_assumptions, 
        params=params, params_k=params_k, 
        agent_names=AGENT_NAMES, snapshot_times=snapshots)
    
    # run the simulation
    sc_simulation.run()
    
    # return the simulation object
    return sc_simulation



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
    
    
    def simulate_scenario(self, scenario, snapshots=(None, None)):
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
        return simulate_scenario(scenario, self.optional_assumptions, 
                                 self.params, self.params_k, snapshots)
        
    
    def get_metrics_for_params(self, params_dict):
        
        self.verbosity_push()
        self.report('Running simulations for parameterisation'
                    f' {params_dict}...')
        
        # set the model parameters as specified
        self.set_params(params_dict)
        
        # loop through the scenarios, simulate them, and calculate metrics
        metrics = {}
        for scenario in self.scenarios.values():
            
            self.verbosity_push()
            self.report(f'Simulating scenario "{scenario.name}"...')
            
            # run this scenario with the specified parameterisation
            sc_simulation = self.simulate_scenario(scenario)
            
            # calculate metric(s) for this scenario
            self.verbosity_push() 
            for short_metric_name in scenario.short_metric_names:
                metric_value = globals()[
                    METRIC_FCN_PREFIX + short_metric_name](sc_simulation)
                full_metric_name = scenario.get_full_metric_name(short_metric_name)
                metrics[full_metric_name] = metric_value
                self.report(f'Metric {full_metric_name} = {metrics[full_metric_name]}')
                
            """
            def store_metric(metric_name, value):
                full_metric_name = scenario.name + '_' + metric_name
                metrics[full_metric_name] = value
                self.report(f'Metric {full_metric_name} = {metrics[full_metric_name]}')
            # - basic prep and entry into conflict area metrics
            # -- pedestrian
            ped_agent = sc_simulation.agents[i_PED_AGENT]
            ped_entry_sample = ped_agent.ca_entry_sample
            ped_entered_ca = not math.isinf(ped_entry_sample)
            if not ped_entered_ca:
                ped_entry_sample = len(sc_simulation.time_stamps)
            store_metric('ped_entered', int(ped_entered_ca))
            # -- vehicle
            veh_agent = sc_simulation.agents[i_VEH_AGENT]
            veh_entry_sample = veh_agent.ca_entry_sample
            veh_entered_ca = not math.isinf(veh_entry_sample)
            if not veh_entered_ca:
                veh_entry_sample = len(sc_simulation.time_stamps)
            store_metric('veh_entered', int(veh_entered_ca))
            # - first passer metrics
            if sc_simulation.first_passer is None:
                ped_1st = False
                veh_1st = False
            else:
                first_passer_name = sc_simulation.first_passer.name
                ped_1st = (first_passer_name == PED_NAME)
                veh_1st = not ped_1st
            store_metric('ped_1st', int(ped_1st))
            store_metric('veh_1st', int(veh_1st))
            # - min ped speed before conflict area
            ped_min_speed_before_ca = np.min(
                ped_agent.trajectory.long_speed[:ped_entry_sample])
            store_metric('ped_min_speed_before', ped_min_speed_before_ca)
            # - max ped speed after entering the conflict area
            if ped_entered_ca:
                ped_max_speed_after_entry = np.max(
                    ped_agent.trajectory.long_speed[ped_entry_sample:])
            else:
                ped_max_speed_after_entry = math.nan
            store_metric('ped_max_speed_after', ped_max_speed_after_entry)
            # - min veh speed before conflict area
            veh_min_speed_before_ca = np.min(
                veh_agent.trajectory.long_speed[:veh_entry_sample])
            store_metric('veh_min_speed_before', veh_min_speed_before_ca)
            # - mean veh speed before conflict area
            veh_mean_speed_early_before_ca = np.mean(
                veh_agent.trajectory.long_speed[:math.ceil(veh_entry_sample/2)])
            store_metric('veh_mean_speed_early_before', 
                         veh_mean_speed_early_before_ca)
            # - max veh dec in multiples of the deceleration needed to stop
            # - just before conflict area
            # -- get the required deceleration to stop just at the conflict
            # -- area
            ca_dist_before_ca = veh_agent.trajectory.pos[
                0,:veh_entry_sample] - AGENT_COLL_DISTS[i_VEH_AGENT]
            stop_dec_before_ca = veh_agent.trajectory.long_speed[
                :veh_entry_sample] ** 2 / (2 * ca_dist_before_ca)
            # -- get the vehicle agent's actual deceleration before the
            # -- conflict area
            dec_before_ca = -veh_agent.trajectory.long_acc[:veh_entry_sample]
            # -- get the max relative deceleration
            veh_max_surplus_dec_before = np.max(dec_before_ca - stop_dec_before_ca)
            store_metric('veh_max_surplus_dec_before', veh_max_surplus_dec_before)
            # - veh speed at first sample where pedestrian increases speed
            # - before entering the conflict area, after the last time the 
            # - pedestrian decreases its speed before the vehicle reaches 
            # - zero speed (if it does)
            veh_zero_spd_samples = np.nonzero(
                veh_agent.trajectory.long_speed == 0)[0]
            veh_speed_at_ped_start = math.nan
            if ped_entered_ca and len(veh_zero_spd_samples) > 0:
                # first find the last speed decrease before vehicle zero speed
                veh_zero_spd_sample = veh_zero_spd_samples[0]
                ped_speed_diff_before_vzs = np.diff(
                    ped_agent.trajectory.long_speed[:veh_zero_spd_sample])
                dec_samples = np.nonzero(ped_speed_diff_before_vzs < 0)[0]
                # any speed decreases at all?
                if len(dec_samples) > 0:
                    last_dec_sample = dec_samples[-1]
                    ped_speed_diff_before_ca = np.diff(
                        ped_agent.trajectory.long_speed[:ped_entry_sample])
                    acc_samples_after_last_dec = np.nonzero(
                        (np.arange(ped_entry_sample-1) > last_dec_sample)
                        & (ped_speed_diff_before_ca > 0))[0]
                    # were there any speed increases after the last speed decrease?
                    if len(acc_samples_after_last_dec) > 0:
                        veh_speed_at_ped_start = veh_agent.trajectory.long_speed[
                            acc_samples_after_last_dec[0]+1]
            store_metric('veh_speed_at_ped_start', veh_speed_at_ped_start)
            """
            
            # plot simulation results?
            # (this will only work nicely if run with %matplotlib inline)
            self.verbosity_push()
            if self.verbose_now():
                sc_simulation.do_plots(trajs=False, action_val_ests = False, 
                                       surplus_action_vals = False,
                                       kinem_states = True, beh_accs = False, 
                                       beh_probs = True, action_vals = False, 
                                       sensory_prob_dens = False, 
                                       beh_activs = False)
                self.report('Showing plots, press [Enter] to continue...')
                input()
                    
            self.verbosity_pop(3)
    
        self.verbosity_pop()
                
        # return the dict of metrics
        return metrics
            
    
    def __init__(self, name, scenarios, optional_assumptions, 
                 default_params, default_params_k, param_arrays, 
                 verbosity=0):
        # build the list of metrics to calculate in this search, from the
        # scenario information
        self.scenarios = scenarios
        metric_names = []
        for scenario in self.scenarios.values():
            for metric_name in scenario.full_metric_names:
                metric_names.append(metric_name)
        # make local copies of the default params objects, to use for
        # parameterising simulations during the fitting
        self.optional_assumptions = copy.copy(optional_assumptions)
        self.params = copy.copy(default_params)
        self.params_k = copy.deepcopy(default_params_k)
        # make sure the value function gains are correct for the free speeds
        sc_scenario_helper.set_val_gains_for_free_speed(
            self.params_k[CtrlType.SPEED], AGENT_FREE_SPEEDS[i_PED_AGENT])
        sc_scenario_helper.set_val_gains_for_free_speed(
            self.params_k[CtrlType.ACCELERATION], AGENT_FREE_SPEEDS[i_VEH_AGENT])
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
                         name=name, verbosity=verbosity)
        # run the grid search
        self.search_grid(free_param_arrays)
        # save the results
        self.save(FIT_RESULTS_FOLDER + (DET_FIT_FILE_NAME_FMT % name))
        
        
# unit testing
if __name__ == "__main__":
        
    plt.close('all')
    
    PARAM_ARRAYS = {}
    PARAM_ARRAYS['k_c'] = (0.2, 2)
    
    OPTIONAL_ASSUMPTIONS = sc_scenario.get_assumptions_dict(False, 
                                                            oVA=False,
                                                            oVAa=False,
                                                            oBEo=False,
                                                            oBEv=False, 
                                                            oAI=False)
    
    DEFAULT_PARAMS, DEFAULT_PARAMS_K = sc_scenario.get_default_params(
        oVA=OPTIONAL_ASSUMPTIONS[OptionalAssumption.oVA])    
    
    test_fit = SCPaperParameterSearch('test', ONE_AG_SCENARIOS,
                                      OPTIONAL_ASSUMPTIONS, 
                                      DEFAULT_PARAMS, DEFAULT_PARAMS_K, 
                                      PARAM_ARRAYS, verbosity=5)
    
    