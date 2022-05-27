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
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import sc_scenario
from sc_scenario_helper import AccessOrder
import sc_fitting
from sc_fitting import i_VEH_AGENT
import sc_plot
import do_2_analyse_deterministic_fits
from do_2_analyse_deterministic_fits import (get_max_crit_parameterisations,
                                             get_best_parameterisations_for_crit,
                                             get_best_scen_var_for_paramet)

PLOT_MODEL_STATES = True
OVERWRITE_SAVED_SIM_RESULTS = False


SAVE_PDF = True
if SAVE_PDF:
    SCALE_DPI = 1
else:
    SCALE_DPI = 0.5


MODEL_NAMES = ('oBEvoAI', 'oVAoBEvoAI')
MODEL_DISPLAY_NAME = ('Snapshot payoffs', 'Affordance-based values')
#MODEL_NAMES = ('oVAoBEvoAI',)
FOCUS_MODEL = 'oVAoBEvoAI' 
SCENARIO = sc_fitting.ONE_AG_SCENARIOS['VehShortStop']
CRITERION = 'Short-stopping'
SNAPSHOT_TIME = 0.5
SIM_RESULTS_FNAME = 'fig_2_SimResults.pkl'
i_BEHS = (sc_scenario.i_PASS1ST, sc_scenario.i_PASS2ND)
BEH_COLORS = (sc_plot.COLORS['Passing first green'],
              sc_plot.COLORS['Passing second red']) 
ACTION_LINE_STYLES = ('--', '-') # no action; deceleration action
ANN_TEXT_COL = 'gray'

def get_actions_to_plot(veh_agent):
    i_no_action = veh_agent.i_no_action
    i_dec_action = i_no_action-2
    i_actions = (i_no_action, i_dec_action)
    return i_actions


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
        params_dict['T_delta'] = 10
        if model_name == FOCUS_MODEL:
            snapshots = (None, (SNAPSHOT_TIME,))
        else:
            snapshots = (None, None)
        print(f'Simulating {model_name} with '
          f' {params_dict} in variant #{i_scen_var+1} scenario {SCENARIO.name}...')
        sim = sc_fitting.construct_model_and_simulate_scenario(
            model_name, params_dict, SCENARIO, i_variation=i_scen_var, 
            apply_stop_criteria=False, zero_acc_after_exit=False, 
            snapshots=snapshots, detailed_snapshots=True)
        sim.do_plots(kinem_states=True, veh_stop_dec=True)
        sims[model_name] = sim
    # save simulation results
    sc_fitting.save_results(sims, SIM_RESULTS_FNAME)
  


plt.close('all')

N_TS_ROWS = 4
fig, axs = plt.subplots(nrows=N_TS_ROWS, ncols=len(MODEL_NAMES)+1,
                        sharex='col', figsize=(0.65*sc_plot.FULL_WIDTH, 
                                               0.4*sc_plot.FULL_WIDTH),
                        dpi=sc_plot.DPI * SCALE_DPI)
i_AH_COL = len(MODEL_NAMES)





# model state time series plots
AX_W = 0.18
AX_HS = np.full(4, 0.14)
AX_HS[2] = 0.22
AX_HS[3] = 0.1
AX_Y_TOP = 0.91
AX_Y_PAD = 0.04
def get_ax_y(i_row):
    return AX_Y_TOP - np.sum(AX_HS[0:i_row+1]) - AX_Y_PAD * (i_row-1)
if PLOT_MODEL_STATES:
    drawn_vline = False
    for i_model, model_name in enumerate(MODEL_NAMES):
        sim = sims[model_name]
        veh_agent = sim.agents[i_VEH_AGENT]
        i_actions = get_actions_to_plot(veh_agent)
        ax_x = 0.1 + 0.26 * i_model
        # plot action/behaviour-dependent states
        for i_row in range(3):
            ax = axs[i_row, i_model]
            for idx_action, i_action in enumerate(i_actions):
                for idx_beh, i_beh in enumerate(i_BEHS):
                    if i_row == 2:
                        if i_model==1 and not drawn_vline:
                            ax.axvline(SNAPSHOT_TIME, c='gray', lw=0.75, ls=':')
                            drawn_vline = True
                            con = ConnectionPatch(
                                xyA=(SNAPSHOT_TIME,-0.05), coordsA=ax.transData, 
                                xyB=(-12, -0.05), coordsB=axs[2, i_AH_COL].transData,
                                color='gray', lw=0.75, arrowstyle ='<-', ls=':')
                            fig.add_artist(con)
                        # V_a|b
                        plot_y = veh_agent.states.action_vals_given_behs[i_action, 
                                                                         i_beh, :]
                        ylabel = '$V_{a|b}$ (-)'
                    elif i_row == 0:
                        # V_b|a
                        plot_y = veh_agent.states.beh_vals_given_actions[i_beh, 
                                                                         i_action, :]
                        ylabel = '$V_{b|a}$ (-)'
                        # ax.set_title(MODEL_DISPLAY_NAME[i_model] + '\n',
                        #              fontsize=sc_plot.DEFAULT_FONT_SIZE)
                    elif i_row == 1:
                        # P_b|a
                        plot_y = veh_agent.states.beh_probs_given_actions[i_beh, 
                                                                          i_action, :]
                        ylabel = '$P_{b|a}$ (-)'
                        ax.set_ylim(-.1, 1.1)
                    # elif i_row == 3:
                    #     # DeltaV_a
                    #     if idx_beh == 0:
                    #         plot_y = veh_agent.states.est_action_surplus_vals[
                    #             i_action, :]
                    #     else:
                    #         plot_this = False
                    ax.plot(sim.time_stamps, plot_y, lw=1, 
                            ls=ACTION_LINE_STYLES[idx_action],
                            color=BEH_COLORS[idx_beh])
                    if i_model == 0:
                        ax.set_ylabel(ylabel + '\n\n', va='center')
                    sc_plot.leave_only_yaxis(ax)
                    ax_y = get_ax_y(i_row)
                    ax.set_position([ax_x, ax_y, AX_W, AX_HS[i_row]])
        # plot acceleration
        ax = axs[3, i_model]
        ax.plot(sim.time_stamps, veh_agent.get_stop_accs(), lw=1, 
                ls=':', color='k')
        ax.plot(sim.time_stamps, veh_agent.trajectory.long_acc, lw=1, 
                ls='-', color='k')
        ax.set_ylim(-5.5, 0.5)
        if i_model == 0:
            ax.set_ylabel('Acc. (m/s$^2$)\n\n', va='center')
        else:
            ax.annotate('Yield', xy=(0.7, -0.1), color=ANN_TEXT_COL)
            ax.annotate('Model', xy=(2.5, -4), color=ANN_TEXT_COL)
        sc_plot.leave_only_yaxis(ax)
        AX_H_DIFF = 0.06
        ax_y = get_ax_y(3)
        ax.set_position([ax_x, ax_y, AX_W, AX_HS[3]])
        # add a separate time axis
        sc_plot.add_linked_time_axis(ax, nudge_down=0.04)
                    
sc_plot.add_panel_label('A', (0.05, 0.92))
sc_plot.add_panel_label('B', (0.31, 0.92))

    
# anticipation horizon plots
N_AH_ROWS = 3
sim = sims[FOCUS_MODEL]
veh_agent = sim.agents[i_VEH_AGENT]
i_actions = get_actions_to_plot(veh_agent)
AX_X = 0.68
AX_W = 0.26
AX_H = 0.18
AX_Y_DIST = 0.2
for i_row in range(N_AH_ROWS):
    for idx_action, i_action in enumerate(i_actions):
        for idx_beh, i_beh in enumerate(i_BEHS):
            access_order_values = veh_agent.snapshot_act_val_details[i_action, i_beh]
            for i_access_ord, access_ord in enumerate(AccessOrder):
                ax = axs[i_row, i_AH_COL]
                acc_ord_val = access_order_values[access_ord]
                if acc_ord_val.value == -math.inf:
                    # invalid action/behaviour/outcome combination
                    continue
                time_stamps = acc_ord_val.details.time_stamps
                if i_row == 0:
                    # speed
                    plot_y = acc_ord_val.details.speeds
                    ax.set_ylabel('Speed (m/s)')
                elif i_row == 1:
                    # values
                    plot_y = acc_ord_val.details.kinematics_values
                    ax.set_ylabel('Reward\nrate (-)\n', va='center')
                    ax.set_ylim(-0.25, 0.1)
                elif i_row == 2:
                    # discounted cumulative values
                    # (inherent access value + discounted kinematics values +
                    # discounted post-anticipation values)
                    plot_y = (
                        np.cumsum(acc_ord_val.details.discounted_values) + 
                        acc_ord_val.details.inh_access_value)
                    #plot_y[-1] += acc_ord_val.details.post_value_discounted
                    plot_y = veh_agent.squash_value(plot_y)
                    ax.set_ylabel('Cumulative\nvalue (-)\n', va='center')
                    ax.set_ylim(-0.5, 0.05)
                if access_ord == AccessOrder.EGOFIRST:
                    alpha = 0.2
                else:
                    alpha = 1
                color = BEH_COLORS[idx_beh]
                line, = ax.plot(time_stamps, plot_y, lw=1, 
                                ls=ACTION_LINE_STYLES[idx_action],
                                color=color, alpha=alpha)
                sc_plot.leave_only_yaxis(ax)
                ax_y = 0.62 - i_row * AX_Y_DIST
                ax.set_position([AX_X, ax_y, AX_W, AX_H])

# add a separate time axis
sc_plot.add_linked_time_axis(ax, nudge_down=0.02)

             
# hide the unused subplot   
axs[-1, -1].axis('off')


# annotate
ax = axs[0, 2]
ANN_LINE_COL = 'lightgray'
ax.text(s='Pass before yielding pedestrian', x=-1.5, y=16.5, color=ANN_TEXT_COL)
# ax.annotate('Passing in front of yielding pedestrian', xy=(8, 14), xytext=(-4, 19),
#             arrowprops={'arrowstyle': '-', 'lw': 1, 'color': ANN_LINE_COL}, 
#             color=ANN_TEXT_COL)
# ax.annotate('Yielding to crossing pedestrian', xy=(10, 9), xytext=(8, 5),
#             arrowprops={'arrowstyle': '-', 'lw': 1, 'color': ANN_LINE_COL}, 
#             color=ANN_TEXT_COL)
ax.text(s='        Yield to\ncrossing pedestrian', x=8, y=5, color=ANN_TEXT_COL)
# ax.annotate('Yielding to yielding pedestrian', xy=(10, 9), xytext=(8, 5),
#             arrowprops={'arrowstyle': '-', 'lw': 1, 'color': ANN_LINE_COL}, 
#             color=ANN_TEXT_COL)
ax.text(s='Yield to yielding pedestrian', x=0.5, y=-2.5, color=ANN_TEXT_COL)

# add legends
# actions
ax = axs[0, 2]
ACTION_LABELS = ('None', 'Decelerate')
leg_handles = []
for idx_action in range(2):
    line, = ax.plot((-1, -1), (-1, -1), lw=1, ls=ACTION_LINE_STYLES[idx_action],
            color='lightgray', label=ACTION_LABELS[idx_action])
    leg_handles.append(line)
legend = ax.legend(handles=leg_handles, frameon=False, loc=(-.45, 1.5),
                   title='Driver action ($a$):')
legend._legend_box.align = 'left'
# behaviours
ax = axs[1, 2]
BEH_LABELS = ('Pass first', 'Pass second')
leg_handles = []
for idx_beh in range(2):
    line, = ax.plot((-1, -1), (-1, -1), lw=1, ls='-',
            color=BEH_COLORS[idx_beh], label=BEH_LABELS[idx_beh])
    leg_handles.append(line)
legend = ax.legend(handles=leg_handles, frameon=False, loc=(0.2, 2.6),
                   title="Pedestrian behavior ($b$):")
legend._legend_box.align = 'left'


sc_plot.add_panel_label('C', (0.61, 0.76))


if SAVE_PDF:
    file_name = sc_plot.FIGS_FOLDER + 'fig3A-C.pdf'
    print(f'Saving {file_name}...')
    plt.savefig(file_name, bbox_inches='tight')



plt.show()
    
