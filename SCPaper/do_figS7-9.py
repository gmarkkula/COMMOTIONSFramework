# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:51:06 2022

@author: tragma
"""
import os
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import sc_fitting
import sc_plot
import do_2_analyse_deterministic_fits
from do_2_analyse_deterministic_fits import (get_max_crit_parameterisations, 
                                             get_best_parameterisations_for_crit,
                                             get_best_scen_var_for_paramet)

SAVE_PDF = False
if SAVE_PDF:
    SCALE_DPI = 1
else:
    SCALE_DPI = 0.5

MODEL_NAMES = ('oBEo', 'oBEvoAI', 'oVAoBEvoAI', 'oVAoBEooBEvoAI', 'oVAaoBEvoAI', 'oVAaoVAloBEvoAI')

SCENARIOS = sc_fitting.ONE_AG_SCENARIOS
#SCENARIO_NAMES = SCENARIOS.keys()
N_COLS = len(SCENARIOS)

SIM_RESULTS_FNAME = 'fig_S7-9_SimResults.pkl'
OVERWRITE_SIM_RESULTS = False

  


# run simulations or load existing simulation results
if sc_fitting.results_exist(SIM_RESULTS_FNAME) and not OVERWRITE_SIM_RESULTS:
    sim_results = sc_fitting.load_results(SIM_RESULTS_FNAME)
else:
    
    print('Running do_2...')       
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            do_2_analyse_deterministic_fits.DO_PLOTS = False
            do_2_analyse_deterministic_fits.SAVE_RETAINED_MODELS = False
            det_fits = do_2_analyse_deterministic_fits.do()
            
    sc_fitting.AGENT_GOALS *= 10 # to not have the base model reach goal within simulation
            
    sim_results = {}
    for i_model, model_name in enumerate(MODEL_NAMES):
        sim_results[model_name] = {}
        # get the parameterisation to simulate
        det_fit = det_fits[model_name]
        idx_max_crit_params = get_max_crit_parameterisations(det_fit)
        idx_param = idx_max_crit_params[0]
        params_array = det_fit.results.params_matrix[idx_param, :]
        params_dict = det_fit.get_params_dict(params_array)
        # simulate across all scenarios
        print(f'*** Simulating for model "{model_name}" ({params_dict})...')
        for i_scenario, scenario in enumerate(SCENARIOS.values()):
            print(f'\t--- Scenario "{scenario.name}..."')
            # determine which scenario variant to run
            i_variation = get_best_scen_var_for_paramet(
                det_fit, idx_param, scenario.name, verbose=True)
            sim_results[model_name][scenario.name] = \
                sc_fitting.construct_model_and_simulate_scenario(
                    model_name, params_dict, scenario,
                    i_variation=i_variation, 
                    zero_acc_after_exit=False, 
                    apply_stop_criteria=False)
                    
    # save simulation results
    sc_fitting.save_results(sim_results, SIM_RESULTS_FNAME)
                
                
            
            
            
# plot
plt.close('all')
for i_model_name, model_name in enumerate(MODEL_NAMES):
    fig, axs = plt.subplots(nrows=4, ncols=N_COLS, sharex='col',
                            figsize=(sc_plot.FULL_WIDTH, 0.5*sc_plot.FULL_WIDTH),
                            dpi=sc_plot.DPI * SCALE_DPI)
    for i_scenario, scenario_name in enumerate(sim_results[model_name].keys()):
        sim = sim_results[model_name][scenario_name]
        kinem_axs = axs[0:3, i_scenario]
        # get the active agent and set colours
        act_agent = None
        i_act_agent = None
        for i_agent, agent in enumerate(sim.agents):
            if agent.const_acc == None:
                act_agent = agent
                i_act_agent = i_agent
                break
        act_agent.plot_color = sc_plot.COLORS['active agent blue']
        act_agent.other_agent.plot_color = sc_plot.COLORS['passive agent grey']
        act_agent.other_agent.plot_dashes = (1, 1)
        i_plot_agents = (1-i_act_agent, i_act_agent)
        agent_alpha = (1, 1)
            
        # plot kinematic states
        kinem_axs[2].axhline(0, ls='--', lw=0.5, color='lightgray')
        sim.do_kinem_states_plot(kinem_axs, veh_stop_dec=False, 
                                 i_plot_agents=i_plot_agents,
                                 axis_labels=False, plot_fill=True, fill_r = 1,
                                 hlines=False)
        # plot beh probs
        ax = axs[3, i_scenario]
            
       

            
            
            