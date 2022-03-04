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
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sc_fitting


plt.close('all')


OVERWRITE_SAVED_SIM_RESULTS = False
    

SCENARIO = sc_fitting.PROB_FIT_SCENARIOS['PedHesitateVehConst']
SCENARIO.end_time = 15
MODEL_NAMES = ('oVAoEAoAN', 'oVAoEAoSNvoPF')
N_PARAMETS_PER_MODEL = 50
SIM_RESULTS_FNAME = 'fig_ProbHesitationSimResults.pkl'
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
    
    

# do plotting
veh_entry_t = SCENARIO.initial_ttcas[sc_fitting.i_VEH_AGENT]
veh_exit_t = veh_entry_t + (2 * sc_fitting.AGENT_COLL_DISTS[sc_fitting.i_VEH_AGENT]
                            / SCENARIO.initial_speeds[sc_fitting.i_VEH_AGENT])
veh_ttcss = veh_entry_t - np.arange(0, SCENARIO.end_time, SCENARIO.time_step) 
kin_fig, kin_axs = plt.subplots(nrows=2, ncols=len(MODEL_NAMES), figsize=(8, 5),
                        sharex='col', sharey='row', tight_layout=True)
st_fig, st_axs = plt.subplots(nrows=4, ncols=1, sharex='col', tight_layout=True)
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
        
        # states plotting
        if 'oSNv' in model_name:
            veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
            perc_cs_dist = ped_agent.perception.states.x_perceived[0, :] - veh_agent.coll_dist
            perc_speed = ped_agent.perception.states.x_perceived[1, :]
            perc_ttcs = perc_cs_dist / perc_speed
            ax = st_axs[0]
            ax.plot(sim.time_stamps, perc_ttcs, 'k', alpha=ALPHA)
            
            i_no_action = ped_agent.i_no_action
            i_dec_action = i_no_action-2
            ax = st_axs[1]
            ax.plot(sim.time_stamps, ped_agent.states.mom_action_vals[i_no_action, :], 'k', alpha=ALPHA)
            ax = st_axs[2]
            ax.plot(sim.time_stamps, ped_agent.states.mom_action_vals[i_dec_action, :], 'k', alpha=ALPHA)
            ax = st_axs[3]
            # ax.plot(sim.time_stamps, ped_agent.states.est_action_vals[i_dec_action, :]
            #         - ped_agent.states.est_action_vals[i_no_action, :], 'k', alpha=ALPHA)
            ax.plot(sim.time_stamps, ped_agent.states.est_action_surplus_vals[i_dec_action, :], 'k', alpha=ALPHA)
            
            # i_actions = get_actions_to_plot(ped_agent)
            # for idx_action, i_action in enumerate(i_actions):
            #     ax = st_axs[idx_action*2+1]
            #     ax.plot(sim.time_stamps, ped_agent.states.mom_action_vals[i_action, :], 'k', alpha=ALPHA)
            #     ax = st_axs[idx_action*2+2]
            #     ax.plot(sim.time_stamps, ped_agent.states.est_action_vals[i_action, :], 'k', alpha=ALPHA)
            
    st_axs[0].plot(sim.time_stamps, veh_ttcss, 'r-', alpha=0.5)
    kin_axs[-1, i_model].set_xlabel('Time (s)')            
kin_axs[0, 0].set_ylabel('$v$ (m/s)')
kin_axs[1, 0].set_ylabel('$d_{CP}$ (m)')
st_axs[0].set_ylim(-5, 15)
for i in (1, 2):
    st_axs[i].set_ylim(0.2, 0.25)
st_axs[3].set_ylim(-0.02, 0.01)

            