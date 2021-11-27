# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:17:02 2021

@author: tragma
"""
import time
import copy
import sc_scenario
import sc_fitting

# set model
MODEL = 'oVAaoVAloBEv'
# get assumptions and default parameters for chosen model
assumptions = sc_scenario.get_assumptions_dict_from_string(MODEL)
params = copy.deepcopy(sc_fitting.DEFAULT_PARAMS)
params_k = copy.deepcopy(sc_fitting.get_default_params_k(MODEL))

# set remaining model parameters
params.T_delta = 10
params.thetaDot_1 = 0.005
params.beta_V = 1

# set scenario to run
SCENARIO = sc_fitting.ONE_AG_SCENARIOS['PedHesitateVehYield']

# simulate and plot
tic = time.perf_counter()
sim = sc_fitting.simulate_scenario(SCENARIO, assumptions, params, params_k, 
                                   snapshots=((1.5,), None)
                                   )
toc = time.perf_counter()
print('Initialising and running simulation took %.3f s.' % (toc - tic,))
sim.do_plots(kinem_states=True, beh_probs=True, beh_activs=True, 
             action_val_ests=True, surplus_action_vals=True)
sc_fitting.get_metrics_for_scenario(SCENARIO, sim, verbose=True)