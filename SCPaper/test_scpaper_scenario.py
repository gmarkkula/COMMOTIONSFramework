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
MODEL = 'oVAoAN'
# get assumptions and default parameters for chosen model
assumptions = sc_scenario.get_assumptions_dict_from_string(MODEL)
params = copy.deepcopy(sc_fitting.DEFAULT_PARAMS)
params_k = copy.deepcopy(sc_fitting.get_default_params_k(MODEL))

# set remaining model parameters
params.T = 0.2
params.DeltaV_th_rel = 0.002
params.sigma_V = 0.1
params.tau_theta = 0.02
#params.T_delta = 40
#params.thetaDot_1 = 0.005
#params.beta_V = 1
#params.T_Of = 4
#params.sigma_O = 0.1
# for ctrl_type in sc_scenario_helper.CtrlType:
#     params_k[ctrl_type]._c = 0.55651188
#     params_k[ctrl_type]._sc = 0.055651188


# set scenario to run
#SCENARIO = sc_fitting.ONE_AG_SCENARIOS['PedCrossVehYield']
# SCENARIO = sc_fitting.PROB_FIT_SCENARIOS['PedHesitateVehConst']
SCENARIO = sc_fitting.SCPaperScenario('TestScenario', 
                                        initial_ttcas=(3, 80), 
                                        veh_const_speed=True,
                                        stop_criteria = sc_fitting.IN_CS_STOP,
                                        metric_names = ('ped_av_speed_to_CS',),
                                        time_step = sc_fitting.PROB_SIM_TIME_STEP,
                                        end_time = sc_fitting.PROB_SIM_END_TIME)

# simulate and plot
n = 1
vs = np.zeros(n)
for i in range(n):
    tic = time.perf_counter()
    sim = sc_fitting.simulate_scenario(SCENARIO, assumptions, params, params_k, 
                                       i_variation=0, snapshots=(1.2, None),
                                       noise_seeds=(0, None), apply_stop_criteria=True)
    toc = time.perf_counter()
    print('Initialising and running simulation took %.3f s.' % (toc - tic,))
    sim.do_plots(kinem_states=True, beh_probs=True, beh_activs=True, 
                  action_val_ests=True, surplus_action_vals=True, looming=False,
                  veh_stop_dec=False)
    metrics = sc_fitting.get_metrics_for_scenario(SCENARIO, sim, verbose=True)
    vs[i] = metrics['TestScenario_ped_av_speed_to_CS']

mean_v = np.mean(vs)
ci_radius = 1.96 * np.std(vs)/math.sqrt(n)
print(f'Average ped_av_speed_to_CS: {mean_v:.3f} m/s\n'
      f'CI: [{mean_v - ci_radius:.3f}, {mean_v + ci_radius:.3f}] m/s')