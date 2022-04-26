# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:51:06 2022

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
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import sc_scenario
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

MODEL_NAMES = ('oBEo', 'oBEvoAI', 'oBEooBEvoAI',
               'oVAoBEo', 'oVAoBEvoAI', 'oVAoBEooBEvoAI', 
               'oVAaoBEvoAI', 'oVAaoVAloBEvoAI')
MODEL_FOCUS_CRITS = ('Gap acceptance hesitation', 'none', 'none',
                     'Gap acceptance hesitation', 'Priority assertion', 'Priority assertion',
                     'none', 'Priority assertion')

SCENARIOS = sc_fitting.ONE_AG_SCENARIOS
N_COLS = len(SCENARIOS)
SCENARIO_CRITERIA = ('Priority assertion', 'Short-stopping', 'Gap acceptance hesitation',
          'Yield acceptance hesitation', 'Early yield acceptance')

SIM_RESULTS_FNAME = 'fig_S7-9_SimResults.pkl'
OVERWRITE_SIM_RESULTS = False

PED_V_LIMS = (-.5, 2.5)
V_LIMS = ((12.5, 14.5), (-1, 17), PED_V_LIMS, PED_V_LIMS, PED_V_LIMS)
T_MAXS = (3.5, 9.5, 9.5, 10.5, 6.5)

i_PLOT_BEH = sc_scenario.i_PASS1ST
assert len(sc_fitting.DEFAULT_PARAMS.ctrl_deltas) == 5
i_PLOT_ACTIONS = (0, 4, 2)
ACTION_LINE_STYLES = ('-', '-', '--')
ACTION_LINE_COLS = ('salmon', 'turquoise', 'black')
ACTION_LINE_WIDTHS = (2, 1, 0.5)


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
        if MODEL_FOCUS_CRITS[i_model] == 'none':
            idx_param = idx_max_crit_params[0]
        else:
            idx_crit_params = get_best_parameterisations_for_crit(
                det_fit, MODEL_FOCUS_CRITS[i_model], 
                idx_params_subset=idx_max_crit_params)
            idx_param = idx_crit_params[0]
        params_array = det_fit.results.params_matrix[idx_param, :]
        params_dict = det_fit.get_params_dict(params_array)
        # simulate across all scenarios
        print(f'*** Simulating for model "{model_name}" ({params_dict})...')
        for i_scenario, scenario in enumerate(SCENARIOS.values()):
            print(f'\t--- Scenario "{scenario.name}..."')
            # determine which scenario variant to run
            i_variation = get_best_scen_var_for_paramet(
                det_fit, idx_param, scenario.name, verbose=True)
            scenario.end_time = 12
            sim_results[model_name][scenario.name] = \
                sc_fitting.construct_model_and_simulate_scenario(
                    model_name, params_dict, scenario,
                    i_variation=i_variation, 
                    zero_acc_after_exit=False, 
                    apply_stop_criteria=False)
            # check if criterion in question is met here
            crit = SCENARIO_CRITERIA[i_scenario]
            if crit in det_fit.criterion_names[0]:
                idx_crit = det_fit.criterion_names[0].index(crit)
                crit_met = det_fit.main_criteria_matrix[idx_crit, idx_param]
            elif crit in det_fit.criterion_names[1]:
                idx_crit = det_fit.criterion_names[1].index(crit)
                crit_met = det_fit.sec_criteria_matrix[idx_crit, idx_param]
            else:
                raise Exception(f'Unexpected criterion name "{crit}".')
            sim_results[model_name][scenario.name].crit_met = crit_met    
                    
    # save simulation results
    sc_fitting.save_results(sim_results, SIM_RESULTS_FNAME)
                
                
            
            
            
# plot
plt.close('all')
for i_model_name, model_name in enumerate(MODEL_NAMES):
    fig, axs = plt.subplots(nrows=4, ncols=N_COLS, sharex='col',
                            figsize=(sc_plot.FULL_WIDTH, 0.4*sc_plot.FULL_WIDTH),
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
        sim.do_kinem_states_plot(kinem_axs, 
                                 veh_stop_dec=(scenario_name=='VehShortStop'), 
                                 i_plot_agents=i_plot_agents,
                                 axis_labels=False, plot_fill=True, fill_r = 1,
                                 hlines=False)
        kinem_axs[0].set_xlim(0, T_MAXS[i_scenario])
        kinem_axs[1].set_ylim(V_LIMS[i_scenario][0], V_LIMS[i_scenario][1])
        kinem_axs[2].set_ylim(-4, 17)
        # plot beh probs
        ax = axs[3, i_scenario]
        for idx_action, i_action in enumerate(i_PLOT_ACTIONS):
            ax.plot(sim.time_stamps, 
                    act_agent.states.beh_probs_given_actions[i_PLOT_BEH, i_action, :],
                    ACTION_LINE_STYLES[idx_action], 
                    c=ACTION_LINE_COLS[idx_action],
                    lw=ACTION_LINE_WIDTHS[idx_action])
            ax.set_ylim(-0.1, 1.1)
        # add title
        axs[0, i_scenario].set_title(SCENARIO_CRITERIA[i_scenario] + f'\n({sim.crit_met})',
                                     fontsize=sc_plot.DEFAULT_FONT_SIZE)
            
       

            
            
            