# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 08:01:59 2022

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
import os, contextlib
import matplotlib.pyplot as plt
import sc_scenario
import sc_fitting
from sc_fitting import i_VEH_AGENT
import sc_plot
import do_2_analyse_deterministic_fits
from do_2_analyse_deterministic_fits import (get_max_crit_parameterisations,
                                             get_best_parameterisations_for_crit,
                                             get_best_scen_var_for_paramet)


OVERWRITE_SAVED_SIM_RESULTS = False


MODEL_NAMES = ('oBEvoAI', 'oVAoBEvoAI')
FOCUS_MODEL = 'oVAoBEvoAI' 
SCENARIO = sc_fitting.ONE_AG_SCENARIOS['VehShortStop']
CRITERION = 'Short-stopping'
SIM_RESULTS_FNAME = 'fig_2_SimResults.pkl'


print('Running do_2...')       
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull):
        do_2_analyse_deterministic_fits.DO_PLOTS = False
        do_2_analyse_deterministic_fits.SAVE_RETAINED_MODELS = False
        det_fits = do_2_analyse_deterministic_fits.do()




# get simulation results, by loading existing, or looping through models and simulating
if sc_fitting.results_exist(SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_SIM_RESULTS:
    sims = sc_fitting.load_results(SIM_RESULTS_FNAME)
else:
    # get the model parameterisations that were most successful at short-stopping,
    # and the scenario variation for which the more advanced model was most 
    # successful
    idx_params = []
    for i_model, model_name in enumerate(MODEL_NAMES):
        det_fit = det_fits[model_name]
        idx_max_crit_params = get_max_crit_parameterisations(det_fit)
        idx_best_params_for_crit = get_best_parameterisations_for_crit(
            det_fit, CRITERION, idx_params_subset=idx_max_crit_params)
        idx_model_param = idx_best_params_for_crit[0]
        idx_params.append(idx_model_param)
        if model_name == FOCUS_MODEL:
            i_scen_var = get_best_scen_var_for_paramet(det_fit, idx_model_param, 
                                                       SCENARIO.name, verbose=True)          
    # run simulations
    sims = {}
    for i_model, model_name in enumerate(MODEL_NAMES):
        det_fit = det_fits[model_name]
        params_array = det_fit.results.params_matrix[idx_params[i_model], :]
        params_dict = det_fit.get_params_dict(params_array)
        print(f'Simulating {model_name} with '
          f' {params_dict} in variant #{i_scen_var+1} scenario {SCENARIO.name}...')
        sim = sc_fitting.construct_model_and_simulate_scenario(
            model_name, params_dict, SCENARIO, i_variation=i_scen_var, 
            apply_stop_criteria=False, zero_acc_after_exit=False)
        sims[model_name] = sim
    # save simulation results
    sc_fitting.save_results(sims, SIM_RESULTS_FNAME)
    


# scenario time series plots
N_TS_ROWS = 5
i_BEHS = (sc_scenario.i_PASS1ST, sc_scenario.i_PASS2ND)
BEH_COLORS = ('green', 'red')
ACTION_LINE_STYLES = ('--', '-')
ts_fig, ts_axs = plt.subplots(nrows=N_TS_ROWS, ncols=len(MODEL_NAMES),
                              sharex='col', sharey='row')
for i_model, model_name in enumerate(MODEL_NAMES):
    sim = sims[model_name]
    veh_agent = sim.agents[i_VEH_AGENT]
    i_no_action = veh_agent.i_no_action
    i_dec_action = i_no_action-2
    i_actions = (i_no_action, i_dec_action)
    for i_row in range(N_TS_ROWS):
        ax = ts_axs[i_row, i_model]
        for idx_action, i_action in enumerate(i_actions):
            for idx_beh, i_beh in enumerate(i_BEHS):
                plot_this = True
                if i_row == 0:
                    # V_a|b
                    plot_y = veh_agent.states.action_vals_given_behs[i_action, 
                                                                     i_beh, :]
                elif i_row == 1:
                    # V_b|a
                    plot_y = veh_agent.states.beh_vals_given_actions[i_beh, 
                                                                     i_action, :]
                elif i_row == 2:
                    # P_b|a
                    plot_y = veh_agent.states.beh_probs_given_actions[i_beh, 
                                                                      i_action, :]
                elif i_row == 3:
                    # DeltaV_a
                    if idx_beh == 0:
                        plot_y = veh_agent.states.est_action_surplus_vals[
                            i_action, :]
                    else:
                        plot_this = False
                elif i_row == 4:
                    # acceleration
                    if idx_beh == 0:
                        plot_y = veh_agent.trajectory.long_acc
                    else:
                        plot_this = False
                if plot_this:
                    ax.plot(sim.time_stamps, plot_y, lw=1, 
                            ls=ACTION_LINE_STYLES[idx_action],
                            color=BEH_COLORS[idx_beh])
                    
    
    
# anticipation horizon plots

