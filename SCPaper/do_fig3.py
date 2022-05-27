# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 08:36:50 2022

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
import itertools
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import sc_fitting
import sc_plot


plt.close('all')

SAVE_PDF = True
if SAVE_PDF:
    SCALE_DPI = 1
else:
    SCALE_DPI = 0.5

OVERWRITE_SAVED_SIM_RESULTS = False 
PARALLEL = True

SCENARIO = sc_fitting.PROB_FIT_SCENARIOS['PedHesitateVehConst']
SCENARIO.end_time = 15
MODEL_NAMES = ('oVAoEAoAN', 'oVAoEAoSNvoPF')
N_PARAMETS_PER_MODEL = 500
SIM_RESULTS_FNAME = 'fig_3_SimResults.pkl'
ALPHA = 0.1


def run_one_sim(model_name, i_sim, params_dict):
    print(f'Simulating {model_name} with parameterisation'
          f' #{i_sim+1}/{N_PARAMETS_PER_MODEL}:'
          f' {params_dict}...')
    return sc_fitting.construct_model_and_simulate_scenario(
        model_name, params_dict, SCENARIO, apply_stop_criteria=False,
        zero_acc_after_exit=False)



if __name__ == '__main__':

    # initialise random number generator
    rng = np.random.default_rng(seed=0)
    
    # load info on retained probabilistic models
    ret_models = sc_fitting.load_results(sc_fitting.RETAINED_PROB_FNAME)
    
    
    # get simulation results, by loading existing, or looping through models and simulating
    if sc_fitting.results_exist(SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_SIM_RESULTS:
        sim_results = sc_fitting.load_results(SIM_RESULTS_FNAME)
    else:
        if PARALLEL:
            print('Starting pool of workers...')
            pool = mp.Pool(mp.cpu_count()-1)
        sim_results = {}
        for model_name in MODEL_NAMES:
            found_model = False
            for ret_model in ret_models:
                if ret_model.model == model_name:
                    found_model = True
                    break
            if not found_model:
                raise Exception('Model {model_name} not found among retained models.')
                
            # draw retained parameterisations at random and simulate
            n_ret_paramets = ret_model.params_array.shape[0]
            params_dicts = []
            for i_sim in range(N_PARAMETS_PER_MODEL):
                idx_paramet = rng.integers(n_ret_paramets)
                params_array = ret_model.params_array[idx_paramet, :]
                params_dict = dict(zip(ret_model.param_names, params_array))
                params_dicts.append(dict( zip(ret_model.param_names, params_array)))
            sim_iter = ((model_name, i, params_dicts[i]) 
                        for i in range(N_PARAMETS_PER_MODEL))
            if PARALLEL:
                sims = list(pool.starmap(run_one_sim, sim_iter))
            else:
                sims = list(itertools.starmap(run_one_sim, sim_iter))
                
             # store the needed info in a data structure to be saved
            sim_results[model_name] = {}   
            for i_sim, sim in enumerate(sims):
                ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
                veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
                # allocate arrays and store info shared across simulations
                if i_sim == 0:
                    sim_results[model_name]['n_simulations'] = N_PARAMETS_PER_MODEL
                    sim_results[model_name]['time_stamps'] = sim.time_stamps
                    sim_results[model_name]['ped_coll_dist'] = ped_agent.coll_dist
                    n_time_steps = len(sim.time_stamps)
                    for vector_name in ('ped_acc', 'ped_speed', 'ped_CP_dist', 
                                        'perc_ttc', 'V_none', 'V_dec', 'DeltaV'):
                        sim_results[model_name][vector_name] = np.full(
                            (N_PARAMETS_PER_MODEL, n_time_steps), np.nan)
                # get and store needed vectors
                # - pedestrian kinematic states
                sim_results[model_name]['ped_acc'][i_sim, :] = \
                    ped_agent.trajectory.long_acc
                sim_results[model_name]['ped_speed'][i_sim, :] = \
                    ped_agent.trajectory.long_speed
                sim_results[model_name]['ped_CP_dist'][i_sim, :] = \
                    ped_agent.signed_CP_dists
                # - pedestrian's perceived vehicle TTCs
                perc_cs_dist = (ped_agent.perception.states.x_perceived[0, :] 
                                - veh_agent.coll_dist) 
                perc_speed = ped_agent.perception.states.x_perceived[1, :] 
                perc_ttc = perc_cs_dist / perc_speed
                sim_results[model_name]['perc_ttc'][i_sim, :] = perc_ttc
                # - pedestrian action values
                i_no_action = ped_agent.i_no_action
                i_dec_action = i_no_action-2
                sim_results[model_name]['V_none'][i_sim, :] = \
                    ped_agent.states.mom_action_vals[i_no_action, :]
                sim_results[model_name]['V_dec'][i_sim, :] = \
                    ped_agent.states.mom_action_vals[i_dec_action, :]
                sim_results[model_name]['DeltaV'][i_sim, :] = \
                    ped_agent.states.est_action_surplus_vals[i_dec_action, :]
        # save simulation results]
        sc_fitting.save_results(sim_results, SIM_RESULTS_FNAME)
       
       
    
    
    def get_quantiles(x_array, y_arrays, qs=(0.2, 0.5, 0.8), step=10):
        n_data = y_arrays.shape[1]
        n_qs = len(qs)
        assert(len(x_array) == n_data)
        quantiles = np.full((n_qs, n_data), np.nan)
        # outside_edge_prop = (1 - prop)/2
        # edge_props = (outside_edge_prop, 1 - outside_edge_prop)
        for i_data in range(0, n_data, step):
            data = y_arrays[:, i_data:i_data+step]
            quantiles[:, i_data] = np.quantile(data, qs)
        #     ecdf = ECDF(data.flat)
        #     for i_edge, edge_prop in enumerate(edge_props):
        #         i_edge_loc = np.nonzero(ecdf.y >= edge_prop)[0][0]
        #         edges[i_edge, i_data] = ecdf.x[i_edge_loc]
        x_out = np.copy(x_array[::step])
        return x_out, quantiles[:, ::step]
        
    
    def do_state_panel_plots(ax, sim_result, state_vector_name, i_examples=None, 
                             color='k', posinf_replace=None, plot_median=True,
                             subset=None):
        time_stamps = sim_result['time_stamps']
        states = sim_result[state_vector_name]
        if not type(subset) == type(None):
            states = states[subset, :]
        if not posinf_replace == None:
            states[states == np.inf] = posinf_replace
        quantile_time_stamps, quantile_ys = get_quantiles(time_stamps, states)
        if not type(i_examples) == type(None):
            if type(i_examples) == int:
                i_examples = (i_examples,)
            for i_example in i_examples:
                ex_ys = states[i_example, :]
                ax.plot(time_stamps, ex_ys, lw=0.5, color=color, alpha=0.3)
        ax.fill_between(quantile_time_stamps, quantile_ys[0, :], quantile_ys[2, :], 
                color=color, alpha=0.3, lw=0)
        if plot_median:
            ax.plot(quantile_time_stamps, quantile_ys[1, :], lw=1, color=color, alpha=1)
        
        
        
    # do plotting
    print('Plotting...')
    fig, axs = plt.subplot_mosaic(layout=('024\n'
                                          '025\n'
                                          '136\n'
                                          '136'),
                                  figsize=(0.73*sc_plot.FULL_WIDTH, 
                                           0.25*sc_plot.FULL_WIDTH), 
                                  dpi=sc_plot.DPI * SCALE_DPI)

    
    
    
    # - kinematic states
    AX_LEFT = 0.12
    AX_W = 0.18
    AX_X_DIST = 0.24
    AX_H = 0.35
    AX_Y_DIST = 0.4
    #i_EXS = np.arange(0, 15, 3)
    i_EXS = np.arange(5)
    i_STATES_EX = 0 # 0 passes before, 5 passes behind
    STATES_EX_COL = 'blue'
    STATES_EX_ALPHA = 0.3
    veh_entry_t = SCENARIO.initial_ttcas[sc_fitting.i_VEH_AGENT]
    veh_exit_t = veh_entry_t + (2 * sc_fitting.AGENT_COLL_DISTS[sc_fitting.i_VEH_AGENT]
                                / SCENARIO.initial_speeds[sc_fitting.i_VEH_AGENT])
    for i_model, model_name in enumerate(MODEL_NAMES):
        kin_axs = []
        ax_x = AX_LEFT + i_model * AX_X_DIST
        for i_plot in range(2):
            kin_axs.append(axs[str(i_model*2 + i_plot)])
            #print(kin_axs[i_plot].get_position())
            ax_y = 0.54 - AX_Y_DIST * i_plot
            kin_axs[i_plot].set_position((ax_x, ax_y, AX_W, AX_H))
        sim_result = sim_results[model_name]
        ped_coll_dist = sim_result['ped_coll_dist']
        time_stamps = sim_result['time_stamps']
        # plot the example highlighted in the internal model states plot
        if i_model == 1:
            SE_LW = 1.5
            kin_axs[0].plot(time_stamps, sim_result['ped_speed'][i_STATES_EX, :], 
                            c=STATES_EX_COL, lw=SE_LW, alpha=STATES_EX_ALPHA)
            kin_axs[1].plot(time_stamps, sim_result['ped_CP_dist'][i_STATES_EX, :], 
                            c=STATES_EX_COL, lw=SE_LW, alpha=STATES_EX_ALPHA)
        # plot vehicle passage
        kin_axs[1].fill(np.array((veh_entry_t, veh_exit_t, veh_exit_t, veh_entry_t)),
                np.array((1, 1, -1, -1)) * ped_coll_dist,
                c='red', edgecolor='none', alpha=0.3)
        if i_model == 0:
            kin_axs[1].text(s='Vehicle\npassing', x=veh_exit_t+0.5, y=ped_coll_dist, 
                            color='red', alpha=0.3)
        # find simulations where pedestrian passes first vs second, and plot quantile fills
        idx_veh_entry = np.nonzero(time_stamps >= veh_entry_t)[0][0]
        bidxs_ped_first = sim_result['ped_CP_dist'][:, idx_veh_entry] <= ped_coll_dist
        for i_order in range(2):
            if i_order == 0:
                bidxs_order = bidxs_ped_first
            else:
                bidxs_order = ~bidxs_ped_first
            if np.count_nonzero(bidxs_order) > 1:
                for i_plot, vector_name in enumerate(('ped_speed', 'ped_CP_dist')):
                    ax = kin_axs[i_plot]
                    do_state_panel_plots(ax, sim_result, 
                                         vector_name, i_examples=i_EXS, 
                                         plot_median=True, subset=bidxs_order,
                                         color='k')
                    sc_plot.leave_only_yaxis(ax)
        kin_axs[1].set_xlabel('Time (s)')   
        kin_axs[1].set_ylim(-ped_coll_dist-1, 6)    
        kin_axs[0].set_ylim(-0.1, 4.1)     
        if i_model == 0:
            kin_axs[0].set_ylabel('Speed (m/s)')
            kin_axs[1].set_ylabel('Position (m)')
        # add separate time axis
        sc_plot.add_linked_time_axis(kin_axs[-1])
    
    # add panel labels
    # LABEL_Y = 0.89
    # sc_plot.add_panel_label('A', (0.09, LABEL_Y))
    # sc_plot.add_panel_label('B', (0.37, LABEL_Y))
    LABEL_Y = 0.92
    sc_plot.add_panel_label('D', (0.07, LABEL_Y))
    sc_plot.add_panel_label('E', (0.35, LABEL_Y))
    
    # - internal model states
    N_ST_PLOTS = 3
    V_YLIM = (0.21, 0.25)
    st_axs = []
    for i_plot in range(N_ST_PLOTS):
        st_axs.append(axs[str(4 + i_plot)])
    sim_result = sim_results['oVAoEAoSNvoPF']
    time_stamps = sim_result['time_stamps']
    # vehicle TTA
    ax = st_axs[0]
    # veh_ttcss = veh_entry_t - np.arange(0, SCENARIO.end_time, SCENARIO.time_step) 
    # ax.plot(time_stamps, veh_ttcss, 'r--', alpha=0.5)
    do_state_panel_plots(ax, sim_result, 'perc_ttc', i_STATES_EX, 'k', posinf_replace=100) 
    ax.set_ylim(-1, 12)
    ax.set_ylabel('TTA (s)')
    # value of non-action
    ax = st_axs[1]
    NON_ACT_COL = 'green'
    do_state_panel_plots(ax, sim_result, 'V_none', i_STATES_EX, NON_ACT_COL)
    ax.set_ylim(V_YLIM[0], V_YLIM[1])
    ax.text(4, 0.247, 'Maintain speed', c=NON_ACT_COL)
    # value of decelerating
    ax = st_axs[1]
    DEC_COL = 'magenta'
    do_state_panel_plots(ax, sim_result, 'V_dec', i_STATES_EX, DEC_COL)
    ax.set_ylim(V_YLIM[0], V_YLIM[1])
    ax.text(6, 0.225, 'Slow down', c=DEC_COL)
    ax.set_ylabel('$V_a$ (-)')
    # # accumulated surplus value of decelerating
    # ax = st_axs[2]
    # do_state_panel_plots(ax, sim_result, 'DeltaV', i_STATES_EX, 'magenta')   
    # ax.set_ylim(-0.015, 0.005)
    # ax.set_ylabel('$\Delta V_a$ (-)')
    # acceleration
    ax = st_axs[2]
    ax.plot(time_stamps, sim_result['ped_acc'][i_STATES_EX, :], 
                   c=STATES_EX_COL, lw=0.5, alpha=STATES_EX_ALPHA)     
    ax.set_ylabel('Acc. (m/s$^2$)')
    AX_X = 0.66
    AX_W = 0.25
    AX_H = 0.245
    AX_Y_DIST = 0.29
    for i_plot in range(N_ST_PLOTS):
        st_axs[i_plot].set_xlim(-.1, 9.1)
        sc_plot.leave_only_yaxis(st_axs[i_plot])
        ax_y = 0.73 - AX_Y_DIST * i_plot
        st_axs[i_plot].set_position((AX_X, ax_y, AX_W, AX_H))
    # add separate time axis
    sc_plot.add_linked_time_axis(st_axs[-1])
    
    # add panel label
    sc_plot.add_panel_label('F', (0.58, 0.9))
    
    
    if SAVE_PDF:
        file_name = sc_plot.FIGS_FOLDER + 'fig3D-F.pdf'
        print(f'Saving {file_name}...')
        plt.savefig(file_name, bbox_inches='tight')
        
        
    plt.show()