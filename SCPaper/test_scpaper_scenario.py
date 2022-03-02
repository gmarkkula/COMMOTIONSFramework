# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:17:02 2021

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
import math
import time
import copy
import numpy as np
import sc_scenario
import sc_scenario_helper
import sc_fitting

# set model
MODEL = 'oVAoBEvoAI'
# # get assumptions and default parameters for chosen model
# assumptions = sc_scenario.get_assumptions_dict_from_string(MODEL)
# params = copy.deepcopy(sc_fitting.DEFAULT_PARAMS)
# params_k = copy.deepcopy(sc_fitting.get_default_params_k(MODEL))

# set remaining model parameters
# params.T = 0.5
# params.DeltaV_th_rel = 0.003
# params.sigma_V = 0.01
#params.tau_theta = 0.05
#params.sigma_xdot = 0.1
#sc_fitting.V_NY_REL = -2
#params.T_delta = 60
#params.thetaDot_1 = 0.00
#params.thetaDot_1 = 0.001
#params.beta_V = 1
#params.T_Of = 3.7
#params.sigma_O = 1.2
# for ctrl_type in sc_scenario_helper.CtrlType:
#     params_k[ctrl_type]._c = 0.431
#     params_k[ctrl_type]._sc = 0.02

# params.T_s = 0.5
# params.D_s = 0.5
# params.ctrl_deltas *= 2
params_dict = {'T_delta': 35.938136638046274, 'T': 0.5, 'DeltaV_th_rel': 0.001, 
          'beta_V': 61.615502775833434, 'tau_theta': 0.0049999999999999994}


if True:
    
    SCENARIO = sc_fitting.ONE_AG_SCENARIOS['VehShortStop']
    # SCENARIO = sc_fitting.PROB_FIT_SCENARIOS['Encounter']
    # SCENARIO = sc_fitting.SCPaperScenario('TestScenario', 
    #                                         initial_ttcas=(3, 8), 
    #                                         veh_const_speed=True,
    #                                         stop_criteria = sc_fitting.IN_CS_STOP,
    #                                         metric_names = ('ped_av_speed', 'ped_av_speed_to_CS'),
    #                                         time_step = sc_fitting.PROB_SIM_TIME_STEP,
    #                                         end_time = sc_fitting.PROB_SIM_END_TIME)
    #SCENARIO.end_time = 10
    #i_variations = range(SCENARIO.n_variations)
    i_variations = (0,)
    for i_var in i_variations:
        print(f'\n{SCENARIO.name} variation {i_var+1}/{SCENARIO.n_variations}:')
        tic = time.perf_counter()
        sim = sc_fitting.construct_model_and_simulate_scenario(MODEL, params_dict, SCENARIO,
                                                               i_variation=i_var, 
                                                               snapshots=(None, (1,)),
                                                               detailed_snapshots=True,
                                                               noise_seeds=(None, None), 
                                                               apply_stop_criteria=False)
        toc = time.perf_counter()
        print('Initialising and running simulation took %.3f s.' % (toc - tic,))
        sim.do_plots(kinem_states=True, beh_probs=True, beh_activs=False, 
                      action_val_ests=False, surplus_action_vals=False, looming=False,
                      veh_stop_dec=False)
        metrics = sc_fitting.get_metrics_for_scenario(SCENARIO, sim, verbose=True)


if False:
    
    SCENARIOS = sc_fitting.HIKER_SCENARIOS
        
    print(f'\n*** Looping through {SCENARIOS.keys()}:')
    
    for scenario in SCENARIOS.values():

        for i_var in range(scenario.n_variations):
            print(f'\n{scenario.name} variation {i_var+1}/{scenario.n_variations}:')
            tic = time.perf_counter()
            # sim = sc_fitting.simulate_scenario(scenario, assumptions, params, params_k, 
            #                                    i_variation=i_var, snapshots=(None, None),
            #                                    noise_seeds=(None, None), 
            #                                    apply_stop_criteria=True)
            sim = sc_fitting.construct_model_and_simulate_scenario(
                MODEL, params_dict, scenario, i_variation=i_var, 
                snapshots=(None, None), detailed_snapshots=False, 
                noise_seeds=(None, None), apply_stop_criteria=True)
            toc = time.perf_counter()
            print('Initialising and running simulation took %.3f s.' % (toc - tic,))
            sim.do_plots(kinem_states=True, beh_probs=True, beh_activs=False, 
                          action_val_ests=False, surplus_action_vals=True, looming=False,
                          veh_stop_dec=False)
            metrics = sc_fitting.get_metrics_for_scenario(scenario, sim, verbose=True)


if False:
    n = 100
    vs = np.zeros(n)
    for i in range(n):
        tic = time.perf_counter()
        sim = sc_fitting.simulate_scenario(SCENARIO, assumptions, params, params_k, 
                                           i_variation=0, snapshots=(None, None),
                                           noise_seeds=(None, None), apply_stop_criteria=True)
        toc = time.perf_counter()
        print('Initialising and running simulation took %.3f s.' % (toc - tic,))
        # sim.do_plots(kinem_states=True, beh_probs=True, beh_activs=True, 
        #               action_val_ests=True, surplus_action_vals=True, looming=False,
        #               veh_stop_dec=False)
        metrics = sc_fitting.get_metrics_for_scenario(SCENARIO, sim, verbose=True)
        vs[i] = metrics['PedHesitateVehConst_ped_av_speed_to_CS']
    
    mean_v = np.mean(vs)
    ci_radius = 1.96 * np.std(vs)/math.sqrt(n)
    print(f'Average ped_av_speed_to_CS: {mean_v:.3f} m/s\n'
          f'CI: [{mean_v - ci_radius:.3f}, {mean_v + ci_radius:.3f}] m/s')