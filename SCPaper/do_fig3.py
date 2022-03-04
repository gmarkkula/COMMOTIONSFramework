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
import math
import pickle
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import sc_fitting


plt.close('all')


OVERWRITE_SAVED_SIM_RESULTS = False
    

SCENARIO = sc_fitting.PROB_FIT_SCENARIOS['PedHesitateVehConst']
SCENARIO.end_time = 15
MODEL_NAMES = ('oVAoEAoAN', 'oVAoEAoSNvoPF')
N_PARAMETS_PER_MODEL = 50
SIM_RESULTS_FNAME = 'fig_3_SimResults.pkl'
ALPHA = 0.1

# initialise random number generator
rng = np.random.default_rng(seed=0)

# load info on retained probabilistic models
ret_models = sc_fitting.load_results(sc_fitting.RETAINED_PROB_FNAME)


# get simulation results, by loading existing, or looping through models and simulating
if sc_fitting.results_exist(SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_SIM_RESULTS:
    sims = sc_fitting.load_results(SIM_RESULTS_FNAME)
else:
    sims = {}
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
        sims[model_name] = []
        for i_sim in range(N_PARAMETS_PER_MODEL):
            idx_paramet = rng.integers(n_ret_paramets)
            params_array = ret_model.params_array[idx_paramet, :]
            params_dict = dict(zip(ret_model.param_names, params_array))
            print(f'Simulating {model_name} with parameterisation'
                  f' #{i_sim+1}/{N_PARAMETS_PER_MODEL}: {params_dict}...')
            sim = sc_fitting.construct_model_and_simulate_scenario(
                model_name, params_dict, SCENARIO, apply_stop_criteria=False,
                zero_acc_after_exit=False)
            sims[model_name].append(sim)
    # save simulation results
    sc_fitting.save_results(sims, SIM_RESULTS_FNAME)
   

        
    
    
# get time series vectors for distribution analysis
n_time_steps = math.floor(SCENARIO.end_time / SCENARIO.time_step)
ttcss = np.full((N_PARAMETS_PER_MODEL, n_time_steps), np.nan)
V_none = np.full((N_PARAMETS_PER_MODEL, n_time_steps), np.nan)
V_dec = np.full((N_PARAMETS_PER_MODEL, n_time_steps), np.nan)
DeltaV = np.full((N_PARAMETS_PER_MODEL, n_time_steps), np.nan)
for i_sim, sim in enumerate(sims['oVAoEAoSNvoPF']):
    ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
    veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
    perc_cs_dist = ped_agent.perception.states.x_perceived[0, :] - veh_agent.coll_dist
    perc_speed = ped_agent.perception.states.x_perceived[1, :]
    perc_ttcs = perc_cs_dist / perc_speed
    ttcss[i_sim, :] = perc_ttcs
    i_no_action = ped_agent.i_no_action
    i_dec_action = i_no_action-2
    V_none[i_sim, :] = ped_agent.states.mom_action_vals[i_no_action, :]
    V_dec[i_sim, :] = ped_agent.states.mom_action_vals[i_dec_action, :]
    DeltaV[i_sim, :] = ped_agent.states.est_action_surplus_vals[i_dec_action, :]
    


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
    

def do_state_panel_plots(ax, time_stamps, states, i_example, color, posinf_replace=None):
    if not posinf_replace == None:
        states[states == np.inf] = posinf_replace
    quantile_time_stamps, quantile_ys = get_quantiles(time_stamps, states)
    ex_ys = states[i_example, :]
    ax.fill_between(quantile_time_stamps, quantile_ys[0, :], quantile_ys[2, :], 
            color=color, alpha=0.3, lw=0)
    ax.plot(time_stamps, ex_ys, lw=0.5, color=color, alpha=0.5)
    ax.plot(quantile_time_stamps, quantile_ys[1, :], lw=1, color=color)
    
# do plotting
print('Plotting...')
veh_entry_t = SCENARIO.initial_ttcas[sc_fitting.i_VEH_AGENT]
veh_exit_t = veh_entry_t + (2 * sc_fitting.AGENT_COLL_DISTS[sc_fitting.i_VEH_AGENT]
                            / SCENARIO.initial_speeds[sc_fitting.i_VEH_AGENT])
kin_fig, kin_axs = plt.subplots(nrows=2, ncols=len(MODEL_NAMES), figsize=(8, 5),
                        sharex='col', sharey='row', tight_layout=True)
for i_model, model_name in enumerate(MODEL_NAMES):
    for i_sim, sim in enumerate(sims[model_name]):
        ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
        # # acc
        # ax = axs[0, i_model]
        # ax.plot(sim.time_stamps, ped_agent.trajectory.long_acc,
                             # 'k', alpha=ALPHA)
        #ax.set_xlim(-0.5, 6.5)
        #ax.set_ylim(-3, 3)
        # speed
        ax = kin_axs[0, i_model]
        ax.plot(sim.time_stamps, ped_agent.trajectory.long_speed,
                             'k', alpha=ALPHA)
        ax.set_ylim(-.1, 4.1)
        # distance
        ax = kin_axs[1, i_model]
        ax.plot(sim.time_stamps, ped_agent.signed_CP_dists,
                             'k', alpha=ALPHA)
        ax.set_ylim(-ped_agent.coll_dist-1, 6)
        if i_sim == 0:
            ax.fill(np.array((veh_entry_t, veh_exit_t, veh_exit_t, veh_entry_t)),
                    np.array((1, 1, -1, -1)) * ped_agent.coll_dist,
                    c='red', edgecolor='none', alpha=0.3)
            
    kin_axs[-1, i_model].set_xlabel('Time (s)')            
kin_axs[0, 0].set_ylabel('$v$ (m/s)')
kin_axs[1, 0].set_ylabel('$d_{CP}$ (m)')



st_fig, st_axs = plt.subplots(nrows=4, ncols=1, sharex='col', tight_layout=True)
i_SIM_EX = 3 # 0 speeds up, 3 stops, 7 decelerates then accelerates
sim = sims['oVAoEAoSNvoPF'][i_SIM_EX]
ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
do_state_panel_plots(st_axs[0], sim.time_stamps, ttcss, i_SIM_EX, 'k', posinf_replace=100)
veh_ttcss = veh_entry_t - np.arange(0, SCENARIO.end_time, SCENARIO.time_step) 
st_axs[0].plot(sim.time_stamps, veh_ttcss, 'k--', alpha=0.5)
do_state_panel_plots(st_axs[1], sim.time_stamps, V_none, i_SIM_EX, 'blue')
do_state_panel_plots(st_axs[1], sim.time_stamps, V_dec, i_SIM_EX, 'red')
do_state_panel_plots(st_axs[2], sim.time_stamps, DeltaV, i_SIM_EX, 'green')
st_axs[3].plot(sim.time_stamps, ped_agent.trajectory.long_acc, 'k', lw=0.5)         
st_axs[0].set_ylim(-1, 12)
st_axs[1].set_ylim(0.21, 0.25)
st_axs[2].set_ylim(-0.015, 0.005)
st_axs[2].set_xlim(0, 8)