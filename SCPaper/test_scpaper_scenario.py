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
import sc_scenario
import sc_scenario_helper
import sc_fitting

# set model
MODEL = 'oBEvoAI'
# get assumptions and default parameters for chosen model
assumptions = sc_scenario.get_assumptions_dict_from_string(MODEL)
params = copy.deepcopy(sc_fitting.DEFAULT_PARAMS)
params_k = copy.deepcopy(sc_fitting.get_default_params_k(MODEL))

# set remaining model parameters
#params.T_delta = 40
#params.thetaDot_1 = 0.005
params.beta_V = 1
#params.T_Of = 4
#params.sigma_O = 0.1
for ctrl_type in sc_scenario_helper.CtrlType:
    params_k[ctrl_type]._c = 0.55651188
    params_k[ctrl_type]._sc = 0.055651188


# set scenario to run
SCENARIO = sc_fitting.ONE_AG_SCENARIOS['PedCrossVehYield']
# SCENARIO = sc_fitting.SCPaperScenario('TestScenario', 
#                                       initial_ttcas=(math.nan, 4),  
#                                       ped_prio = True,
#                                       ped_start_standing=True, 
#                                       ped_const_speed=True,
#                                       metric_names = ('veh_av_speed', 'veh_av_surpl_dec',))

# simulate and plot
tic = time.perf_counter()
sim = sc_fitting.simulate_scenario(SCENARIO, assumptions, params, params_k, 
                                   i_scenario_variation=0, snapshots=(None, None)
                                   )
toc = time.perf_counter()
print('Initialising and running simulation took %.3f s.' % (toc - tic,))
sim.do_plots(kinem_states=True, beh_probs=True, beh_activs=True, 
             action_val_ests=True, surplus_action_vals=True, looming=True,
             veh_stop_dec=True)
sc_fitting.get_metrics_for_scenario(SCENARIO, sim, verbose=True)