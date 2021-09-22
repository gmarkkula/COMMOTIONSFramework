# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:47:55 2021

@author: tragma
"""

import copy
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import parameter_search
from sc_scenario import CtrlType, OptionalAssumption
import sc_scenario
import sc_scenario_helper



FIT_RESULTS_FOLDER = 'results/'

# scenario basics
AGENT_NAMES = ('P', 'V')
AGENT_CTRL_TYPES = (CtrlType.SPEED, CtrlType.ACCELERATION)
AGENT_GOALS = np.array([[0, 5], [-50, 0]])
TIME_STEP = 0.1 # s
END_TIME = 8 # s

# deterministic fitting
DET_FIT_SCENARIOS = ('DS1_PedLargeLead', 'DS2_PedSmallLead')
DET_FIT_SCENARIOS_PED_LEADS = (3, 1) # s
DET_FIT_PED_INITIAL_TTCP = 4 # s
DET_FIT_METRICS = ('DP1_PedGapAcc', 'DP2_CarAdv', 'DP3_CarAcc')
DET_FIT_FILE_NAME_FMT = 'DetFit_%s.pkl'


class SCPaperDeterministicFitting(parameter_search.ParameterSearch):
    
    def get_metrics_for_params(self, params_vector):
        
        self.verbosity_push()
        self.report('Running simulations for parameterisation'
                    f' {self.get_params_dict(params_vector)}...')
        
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
            if not hasattr(params_obj, param_attr):
                raise Exception(f'Could not find attribute "{param_attr}"'
                                ' in parameter object.')
            setattr(params_obj, param_attr, params_vector[i_param])
        # get the agent free speeds
        v_free_P = sc_scenario_helper.get_agent_free_speed(
            self.params_k[CtrlType.SPEED])
        v_free_V = sc_scenario_helper.get_agent_free_speed(
            self.params_k[CtrlType.ACCELERATION])
        # loop through the scenarios, simulate them, and calculate metrics
        metrics = {}
        for i_scenario, scenario in enumerate(DET_FIT_SCENARIOS):
            self.verbosity_push()
            self.report(f'Simulating scenario "{scenario}"...')
            veh_initial_ttcp = (DET_FIT_PED_INITIAL_TTCP 
                                + DET_FIT_SCENARIOS_PED_LEADS[i_scenario])
            initial_pos = np.array([[0, -DET_FIT_PED_INITIAL_TTCP * v_free_P],
                                   [veh_initial_ttcp * v_free_V, 0]])
            # set up the SCSimulation object
            sc_simulation = sc_scenario.SCSimulation(
                AGENT_CTRL_TYPES, AGENT_GOALS, initial_pos, 
                initial_speeds=(v_free_P, v_free_V),
                start_time=0, end_time=END_TIME, time_step=TIME_STEP, 
                optional_assumptions=self.optional_assumptions, 
                params=self.params, params_k=self.params_k,
                agent_names=AGENT_NAMES)
            # run the simulation
            sc_simulation.run()
            # calculate metric(s) for this scenario
            self.verbosity_push()
            def report_metric(metric_name):
                self.report(f'Metric {metric_name} = {metrics[metric_name]}')
            if scenario == 'DS1_PedLargeLead':
                # did the pedestrian pass first when it had a large lead time?
                metrics['DP1_PedGapAcc'] = 0
                report_metric('DP1_PedGapAcc')
            elif scenario == 'DS2_PedSmallLead':
                # did the car pass first when the pedestrian had a small lead?
                metrics['DP2_CarAdv'] = 1
                report_metric('DP2_CarAdv')
                # and did the car speed up before reaching the conflict point?
                metrics['DP3_CarAcc'] = 2
                report_metric('DP3_CarAcc')
            else:
                raise Exception(f'Unexpected scenario name: {scenario}')
            # plot simulation results?
            self.verbosity_push()
            if self.verbose_now():
                sc_simulation.do_plots(trajs=True, action_val_ests = True, 
                                       surplus_action_vals = True,
                                       kinem_states = True, beh_accs = True, 
                                       beh_probs = True, action_vals = True, 
                                       sensory_prob_dens = False, 
                                       beh_activs = True)
                self.report('Showing plots, hold Q key to continue...')
                while not keyboard.is_pressed('q'):
                    plt.pause(0.5)
            self.verbosity_pop()
            self.verbosity_pop()
            self.verbosity_pop()
    
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
        if optional_assumptions[OptionalAssumption.oBEv]:
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
        super().__init__(tuple(free_param_names), DET_FIT_METRICS, name=name,
                         verbosity=verbosity)
        # run the grid search
        self.search_grid(free_param_arrays)
        # save the results
        self.save(FIT_RESULTS_FOLDER + (DET_FIT_FILE_NAME_FMT % name))
        
        
# unit testing
if __name__ == "__main__":
    
    PARAM_ARRAYS = {}
    PARAM_ARRAYS['T_P'] = (0.5, 1)
    
    OPTIONAL_ASSUMPTIONS = sc_scenario.get_assumptions_dict(False, oVA=True)
    
    DEFAULT_PARAMS, DEFAULT_PARAMS_K = sc_scenario.get_default_params(
        oVA=OPTIONAL_ASSUMPTIONS[OptionalAssumption.oVA])
    
    test_fit = SCPaperDeterministicFitting('test', OPTIONAL_ASSUMPTIONS, 
                                           DEFAULT_PARAMS, DEFAULT_PARAMS_K, 
                                           PARAM_ARRAYS, verbosity=5)
    
    