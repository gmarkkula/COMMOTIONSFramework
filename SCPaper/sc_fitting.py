# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:47:55 2021

@author: tragma
"""
import os
import math
import copy
import keyboard
import numpy as np
import matplotlib.pyplot as plt
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
AGENT_FREE_SPEEDS = np.array([1.3, 50 / 3.6]) # m/s 
AGENT_GOALS = np.array([[0, 5], [-50, 0]]) # m
COLLISION_MARGIN = 0.5 # m
TIME_STEP = 0.1 # s
END_TIME = 8 # s
V_NY_REL = -1.5


# deterministic fitting
class SCPaperScenario:
    
    def __init__(self, name, initial_ttcas, ped_prio=False,
                 ped_start_standing=False, ped_standing_margin=COLLISION_MARGIN,
                 ped_const_speed=False, veh_const_speed=False, 
                 veh_yielding=False, veh_yielding_margin=COLLISION_MARGIN):
        self.name = name
        self.ped_prio = ped_prio
        self.initial_cp_distances = (np.array(initial_ttcas) * AGENT_FREE_SPEEDS 
                                     + sc_scenario.SHARED_PARAMS.d_C)
        self.initial_speeds = np.copy(AGENT_FREE_SPEEDS)
        self.const_accs = [None, None]
        if ped_start_standing:
            self.initial_cp_distances[i_PED_AGENT] = (
                sc_scenario.SHARED_PARAMS.d_C + ped_standing_margin)
            self.initial_speeds[i_PED_AGENT] = 0
        if ped_const_speed:
            self.const_accs[i_PED_AGENT] = 0
        if veh_const_speed:
            self.const_accs[i_VEH_AGENT] = 0
        elif veh_yielding:
            stop_dist = (self.initial_cp_distances[i_VEH_AGENT] 
                         - sc_scenario.SHARED_PARAMS.d_C - veh_yielding_margin)
            self.const_accs[i_VEH_AGENT] = (
                -self.initial_speeds[i_VEH_AGENT] ** 2 / (2 * stop_dist))
        
    
DET1S_SCENARIOS = {}
DET1S_SCENARIOS['ActVehStatPed'] = SCPaperScenario('ActVehStatPed', 
                                                   initial_ttcas=(math.nan, 2),  
                                                   ped_start_standing=True, 
                                                   ped_const_speed=True)
DET1S_SCENARIOS['ActVehStatPedPrio'] = SCPaperScenario('ActVehStatPedPrio', 
                                                       initial_ttcas=(math.nan, 4),  
                                                       ped_prio = True,
                                                       ped_start_standing=True, 
                                                       ped_const_speed=True)
DET1S_SCENARIOS['ActPedLeading'] = SCPaperScenario('ActPedLeading', 
                                                   initial_ttcas=(3, 7), 
                                                   veh_const_speed=True)
DET1S_SCENARIOS['ActPedPrioEncounter'] = SCPaperScenario('ActPedPrioEncounter', 
                                                         initial_ttcas=(3, 3), 
                                                         ped_prio=True,
                                                         veh_yielding=True)
DET1S_METRICS_PER_SCEN = ['ped_entered', 'veh_entered', 'ped_1st', 'veh_1st', 
                         'ped_min_speed_before', 'ped_max_speed_after', 
                         'veh_min_speed_before', 'veh_mean_speed_early_before',
                         'veh_max_surplus_dec_before', 'veh_speed_at_ped_start']
DET1S_METRIC_NAMES = []
for scenario in DET1S_SCENARIOS.values():
    for metric_name in DET1S_METRICS_PER_SCEN:
        DET1S_METRIC_NAMES.append(scenario.name + '_' + metric_name)
DET_FIT_FILE_NAME_FMT = 'DetFit_%s.pkl'


class SCPaperDeterministicOneSidedFitting(parameter_search.ParameterSearch):
    
    def set_params(self, params_vector):
        """
        Set self.params and self.params_k to reflect a given model 
        parameterisation.

        Parameters
        ----------
        params_vector : 1D numpy array
            The vector of free parameter values, same order as in 
            self.param_names.

        Returns
        -------
        None.

        """    
        # loop through the provided free parameter values and assign them
        # to the correct attributes of the local parameter objects
        for i_param, param_name in enumerate(self.param_names):
            if param_name[0:2] == 'k_':
                # value function gain parameter
                params_obj = self.params_k
                param_attr = param_name[1:]
            else:
                # other parameter
                params_obj = self.params
                param_attr = param_name
            # if not hasattr(params_obj, param_attr):
            #     raise Exception(f'Could not find attribute "{param_attr}"'
            #                     ' in parameter object.')
            setattr(params_obj, param_attr, params_vector[i_param])
    
    
    def simulate_scenario(self, scenario):
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
        
        # prepare the simulation
        # - initial position
        initial_pos = np.array([[0, -scenario.initial_cp_distances[i_PED_AGENT]],
                               [scenario.initial_cp_distances[i_VEH_AGENT], 0]])
        # - pedestrian priority?
        if scenario.ped_prio:
            self.params.V_ny_rel = V_NY_REL
        else:
            self.params.V_ny_rel = 0
        # - set up the SCSimulation object
        sc_simulation = sc_scenario.SCSimulation(
            AGENT_CTRL_TYPES, AGENT_GOALS, initial_pos, 
            initial_speeds=scenario.initial_speeds, 
            const_accs=scenario.const_accs,
            start_time=0, end_time=END_TIME, time_step=TIME_STEP, 
            optional_assumptions=self.optional_assumptions, 
            params=self.params, params_k=self.params_k,
            agent_names=AGENT_NAMES)
        
        # run the simulation
        sc_simulation.run()
        
        # return the simulation object
        return sc_simulation
        
    
    def get_metrics_for_params(self, params_vector):
        
        self.verbosity_push()
        self.report('Running simulations for parameterisation'
                    f' {self.get_params_dict(params_vector)}...')
        
        # set the model parameters as specified
        self.set_params(params_vector)
        
        # loop through the scenarios, simulate them, and calculate metrics
        metrics = {}
        for scenario in DET1S_SCENARIOS.values():
            
            self.verbosity_push()
            self.report(f'Simulating scenario "{scenario.name}"...')
            
            # run this scenario with the specified parameterisation
            sc_simulation = self.simulate_scenario(scenario)
            
            # calculate metric(s) for this scenario
            self.verbosity_push() 
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
                0,:veh_entry_sample] - sc_scenario.SHARED_PARAMS.d_C
            stop_dec_before_ca = veh_agent.trajectory.long_speed[
                :veh_entry_sample] ** 2 / (2 * ca_dist_before_ca)
            # -- get the vehicle agent's actual deceleration before the
            # -- conflict area
            dec_before_ca = -veh_agent.trajectory.long_acc[:veh_entry_sample]
            # -- get the max relative deceleration
            veh_max_surplus_dec_before = np.max(dec_before_ca - stop_dec_before_ca)
            store_metric('veh_max_surplus_dec_before', veh_max_surplus_dec_before)
            # - veh speed at first sample where pedestrian increases speed
            # - before entering conflict area
            veh_speed_at_ped_start = math.nan
            if ped_entered_ca:
                # first find the last speed decrease before conflict area entry
                ped_speed_diff_before_ca = np.diff(ped_agent.trajectory.long_speed[
                    :ped_entry_sample])
                dec_samples = np.nonzero(ped_speed_diff_before_ca < 0)[0]
                # any speed decreases at all?
                if len(dec_samples) > 0:
                    last_dec_sample = dec_samples[-1]
                    acc_samples_after_last_dec = np.nonzero(
                        (np.arange(ped_entry_sample-1) > last_dec_sample)
                        & (ped_speed_diff_before_ca > 0))[0]
                    # were there any speed increases after the last speed decrease?
                    if len(acc_samples_after_last_dec) > 0:
                        veh_speed_at_ped_start = veh_agent.trajectory.long_speed[
                            acc_samples_after_last_dec[0]+1]
            store_metric('veh_speed_at_ped_start', veh_speed_at_ped_start)
            
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
                
        # return the metrics as a vector
        return self.get_metrics_array(metrics)
            
    
    def __init__(self, name, optional_assumptions, 
                 default_params, default_params_k, param_arrays, 
                 verbosity=0):
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
        super().__init__(tuple(free_param_names), DET1S_METRIC_NAMES, 
                         name=name, verbosity=verbosity)
        # run the grid search
        self.search_grid(free_param_arrays)
        # save the results
        self.save(FIT_RESULTS_FOLDER + (DET_FIT_FILE_NAME_FMT % name))
        
        
# unit testing
if __name__ == "__main__":
        
    plt.close('all')
    
    PARAM_ARRAYS = {}
    PARAM_ARRAYS['T_P'] = (0.5, 1)
    
    OPTIONAL_ASSUMPTIONS = sc_scenario.get_assumptions_dict(False, 
                                                            oVA=True,
                                                            oVAa=False,
                                                            oBEo=False,
                                                            oBEv=False, 
                                                            oAI=False)
    
    DEFAULT_PARAMS, DEFAULT_PARAMS_K = sc_scenario.get_default_params(
        oVA=OPTIONAL_ASSUMPTIONS[OptionalAssumption.oVA])    
    
    test_fit = SCPaperDeterministicOneSidedFitting('test', OPTIONAL_ASSUMPTIONS, 
                                           DEFAULT_PARAMS, DEFAULT_PARAMS_K, 
                                           PARAM_ARRAYS, verbosity=5)
    
    