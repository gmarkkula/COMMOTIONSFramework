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
                                             get_best_scen_var_for_paramet, 
                                             i_MAIN, i_SEC)

SAVE_PDF = True
if SAVE_PDF:
    SCALE_DPI = 1
else:
    SCALE_DPI = 0.5

MODEL_NAMES = ('oBEo', 'oBEvoAI', 'oBEooBEvoAI',
               'oVAoBEo', 'oVAoBEvoAI', 'oVAoBEooBEvoAI', 
               'oVAaoBEvoAI', 'oVAaoVAloBEvoAI', 'oVAaoVAloBEvoAI_gapacches')
MODEL_FOCUS_CRITS = ('Gap acceptance hesitation', 'none', 'none',
                     'Gap acceptance hesitation', 'Priority assertion', 'Priority assertion',
                     'none', 'Priority assertion', 'Gap acceptance hesitation')

SCENARIOS = sc_fitting.ONE_AG_SCENARIOS
SCENARIO_NAMES = SCENARIOS.keys()
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
        clean_model_name = model_name.split('_')[0]
        det_fit = det_fits[clean_model_name]
        # - get a pool to choose from
        if '_' in model_name:
            # get a pool of model parameterisations achieving the focus criterion, 
            # and as many main criteria as possible
            assert(MODEL_FOCUS_CRITS[i_model] != 'none')
            if MODEL_FOCUS_CRITS[i_model] in det_fit.criterion_names[i_MAIN]:
                idx_focus_crit = det_fit.criterion_names[i_MAIN].index(
                    MODEL_FOCUS_CRITS[i_model])
                bidx_focus_met = det_fit.main_criteria_matrix[idx_focus_crit,:]
            elif MODEL_FOCUS_CRITS[i_model] in det_fit.criterion_names[i_SEC]:
                idx_focus_crit = det_fit.criterion_names[i_SEC].index(
                    MODEL_FOCUS_CRITS[i_model])
                bidx_focus_met = det_fit.sec_criteria_matrix[idx_focus_crit,:]
            else:
                raise Exception(f'Unexpected criterion "{MODEL_FOCUS_CRITS[i_model]}".')
            n_max_main_met = np.amax(det_fit.n_main_criteria_met[bidx_focus_met])
            bidx_params_pool = (bidx_focus_met &
                                (det_fit.n_main_criteria_met == n_max_main_met))
            idx_params_pool = np.nonzero(bidx_params_pool)[0]
        else:
            # get a pool of model parameterisations achieving as many main 
            # criteria as possible
            idx_params_pool = get_max_crit_parameterisations(det_fit)
        # - consider the focus criterion when choosing a parameterisation from the pool
        if MODEL_FOCUS_CRITS[i_model] == 'none':
            idx_param = idx_params_pool[0]
        else:
            idx_crit_params = get_best_parameterisations_for_crit(
                det_fit, MODEL_FOCUS_CRITS[i_model], 
                idx_params_subset=idx_params_pool)
            idx_param = idx_crit_params[0]
        params_array = det_fit.results.params_matrix[idx_param, :]
        params_dict = det_fit.get_params_dict(params_array)
        sim_results[model_name]['params_dict'] = params_dict
        # simulate across all scenarios
        sim_results
        print(f'*** Simulating for model "{model_name}" ({params_dict})...')
        for i_scenario, scenario in enumerate(SCENARIOS.values()):
            print(f'\t--- Scenario "{scenario.name}..."')
            # determine which scenario variant to run
            i_variation = get_best_scen_var_for_paramet(
                det_fit, idx_param, scenario.name, verbose=True)
            scenario.end_time = 12
            sim_results[model_name][scenario.name] = \
                sc_fitting.construct_model_and_simulate_scenario(
                    clean_model_name, params_dict, scenario,
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
AX_W = 0.13
AX_H = 0.13
AX_PAD_W = 0.06
AX_PAD_H = 0.025
AX_TOP = 0.825
AX_LEFT = 0.7 * (1 - (N_COLS * AX_W + (N_COLS-1) * AX_PAD_W))
plt.close('all')
for i_model_name, model_name in enumerate(MODEL_NAMES):
    fig, axs = plt.subplots(nrows=4, ncols=N_COLS, sharex='col',
                            figsize=(sc_plot.FULL_WIDTH, 0.45*sc_plot.FULL_WIDTH),
                            dpi=sc_plot.DPI * SCALE_DPI)
    for i_scenario, scenario_name in enumerate(SCENARIO_NAMES):
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
            if ('oBEvoAI' in model_name) or (idx_action == 2):
                ax.plot(sim.time_stamps, 
                        act_agent.states.beh_probs_given_actions[i_PLOT_BEH, i_action, :],
                        ACTION_LINE_STYLES[idx_action], 
                        c=ACTION_LINE_COLS[idx_action],
                        lw=ACTION_LINE_WIDTHS[idx_action])
        ax.set_ylim(-0.1, 1.1)
        # axes
        for i_row, ax in enumerate(axs[:, i_scenario]):
            sc_plot.leave_only_yaxis(ax)
            ax_left = AX_LEFT + (AX_W + AX_PAD_W) * i_scenario
            ax_bottom = AX_TOP - AX_H - (AX_H + AX_PAD_H) * i_row
            ax.set_position([ax_left, ax_bottom, AX_W, AX_H])
        sc_plot.add_linked_time_axis(axs[-1, i_scenario])
        # add title
        if sim.crit_met:
            crit_met_str = '\n(OK)'
        else:
            crit_met_str = '\n'
        axs[0, i_scenario].set_title(SCENARIO_CRITERIA[i_scenario] + crit_met_str,
                                     fontsize=sc_plot.DEFAULT_FONT_SIZE)
    # y axis labels
    axs[0, 0].set_ylabel('$a$ (m/s²)')
    axs[1, 0].set_ylabel('$v$ (m/s)')
    axs[2, 0].set_ylabel(r'$d_\mathrm{CP}$ (m)')
    axs[3, 0].set_ylabel(r'$P(\mathrm{first})$ (-)')
    # model variant and parameterisation 
    LABEL_LEFT = 0.03
    LABEL_BOTTOM = 0.96
    plt.annotate(model_name.split('_')[0], xy=(LABEL_LEFT, LABEL_BOTTOM), 
                 xycoords='figure fraction',
                 fontweight='bold', fontsize=sc_plot.PANEL_LABEL_FONT_SIZE)
    plt.annotate(sc_plot.get_display_params_str(sim_results[model_name]['params_dict']), 
                 xy=(LABEL_LEFT, LABEL_BOTTOM - 0.04), 
                 xycoords='figure fraction')
    # save?
    if SAVE_PDF:
        file_name = sc_plot.FIGS_FOLDER + f'figS7-9_{model_name}.pdf'
        print(f'Saving {file_name}...')
        plt.savefig(file_name, bbox_inches='tight')

    
            
       

            
            
            