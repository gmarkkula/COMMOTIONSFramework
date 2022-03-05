# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 07:42:17 2022

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
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import sc_fitting


OVERWRITE_SAVED_SIM_RESULTS = False
    

TTCP = 3
INITIAL_TTCAS = []
for i_agent in range(2):
    INITIAL_TTCAS.append(TTCP - sc_fitting.AGENT_COLL_DISTS[i_agent] 
                         / sc_fitting.AGENT_FREE_SPEEDS[i_agent])
SCENARIO_END_TIME = 20
SCENARIOS = (sc_fitting.PROB_FIT_SCENARIOS['Encounter'], 
             sc_fitting.PROB_FIT_SCENARIOS['EncounterPedPrio'])


MODEL_NAMES = ('oVAoEAoSNvoPF', 'oVAoBEvoAIoEAoSNvoPF')
N_PARAMETS_PER_MODEL_AND_SCEN = 50
SIM_RESULTS_FNAME = 'fig_4_SimResults.pkl'
ALPHA = 0.1


def run_one_sim(model_name, i_sim, params_dict, scenario):
    print(f'Simulating {model_name} with parameterisation'
          f' #{i_sim+1}/{N_PARAMETS_PER_MODEL_AND_SCEN}:'
          f' {params_dict} in scenario {scenario.name}...\n')
    return sc_fitting.construct_model_and_simulate_scenario(
        model_name, params_dict, scenario, apply_stop_criteria=False,
        zero_acc_after_exit=False)


if __name__ == '__main__':
    

    # initialise random number generator
    rng = np.random.default_rng(seed=0)
    
    # load info on retained probabilistic and combined models - and get as dict instead
    ret_models_tmp = sc_fitting.load_results(sc_fitting.RETAINED_PROB_FNAME)
    ret_models = {}
    for ret_model in ret_models_tmp:
        ret_models[ret_model.model] = ret_model
    ret_models_tmp = sc_fitting.load_results(sc_fitting.RETAINED_COMB_FNAME)
    for ret_model in ret_models_tmp:
        ret_models[ret_model.model] = ret_model
    del ret_models_tmp
    
    
    # get simulation results, by loading existing, or looping through models and simulating
    if sc_fitting.results_exist(SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_SIM_RESULTS:
        sims = sc_fitting.load_results(SIM_RESULTS_FNAME)
    else:
        print('Starting pool of workers...')
        pool = mp.Pool(mp.cpu_count()-1)
        sims = {}
        for model_name in MODEL_NAMES:
            ret_model = ret_models[model_name]
                
            # draw retained parameterisations at random and simulate
            n_ret_paramets = ret_model.params_array.shape[0]
            sims[model_name] = {}
            # - get list of parameterisations
            params_dicts = []
            for i_sim in range(N_PARAMETS_PER_MODEL_AND_SCEN):
                idx_paramet = rng.integers(n_ret_paramets)
                params_array = ret_model.params_array[idx_paramet, :]
                params_dicts.append(dict(
                    zip(ret_model.param_names, params_array)))
            # - loop through scenarios and run simulations
            for scenario in SCENARIOS:
                scenario.end_time = SCENARIO_END_TIME
                sim_iter = ((model_name, i, params_dicts[i], scenario) 
                            for i in range(N_PARAMETS_PER_MODEL_AND_SCEN))
                sims[model_name][scenario.name] = list(
                    pool.starmap(run_one_sim, sim_iter))   
                # sim = sc_fitting.construct_model_and_simulate_scenario(
                #     model_name, params_dict, scenario, apply_stop_criteria=False,
                #     zero_acc_after_exit=False)
                #sims[model_name][scenario_name].append(sim)
        # save simulation results
        sc_fitting.save_results(sims, SIM_RESULTS_FNAME)
    
    
    plt.close('all')
    
    N_ROWS = 4
    for model_name in MODEL_NAMES:
        fig, axs = plt.subplots(nrows=N_ROWS, ncols=len(SCENARIOS), figsize=(6, 8),
                                sharex='row', sharey='row', tight_layout=True,
                                num=model_name)
        axs = axs.reshape((N_ROWS, len(SCENARIOS)))
        for i_scenario, scenario in enumerate(SCENARIOS):
            
            n_sims = len(sims[model_name][scenario.name])
            
            # distance-distance plot
            ax = axs[0, i_scenario]
            for sim in sims[model_name][scenario.name]:
                veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
                ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
                ax.plot(-veh_agent.signed_CP_dists, -ped_agent.signed_CP_dists, 'k',
                        alpha=ALPHA)
                # if sc_fitting.metric_collision(sim):
                #     sim.do_plots(kinem_states=True)
            ax.set_xlim(-50, 50)
            ax.set_xlabel('Vehicle position (m)')
            ax.set_ylim(-6, 6)
            ax.set_ylabel('Pedestrian position (m)')
            ax.fill(np.array((1, 1, -1, -1)) * veh_agent.coll_dist, 
                    np.array((-1, 1, 1, -1)) * ped_agent.coll_dist, color='r',
                    alpha=0.3, edgecolor=None)
  
            # vehicle speed
            ax = axs[1, i_scenario]
            for sim in sims[model_name][scenario.name]:
                veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
                ax.plot(sim.time_stamps, veh_agent.trajectory.long_speed, 
                        'k-', alpha=ALPHA)
                ax.set_ylim(-1, 15)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Vehicle speed (m/s)')
                
            # pedestrian speed
            ax = axs[2, i_scenario]
            for sim in sims[model_name][scenario.name]:
                ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
                ax.plot(sim.time_stamps, ped_agent.trajectory.long_speed, 
                        'k-', alpha=ALPHA)
                ax.set_ylim(-.1, 4.1)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Pedestrian speed (m/s)')
                
            # exit time distributions
            ax = axs[3, i_scenario]
            ped_exit_times = np.full(n_sims, np.nan)
            veh_exit_times = np.full(n_sims, np.nan)
            for i_sim, sim in enumerate(sims[model_name][scenario.name]):
                ped_exit_times[i_sim] = sc_fitting.metric_ped_exit_time(sim)
                veh_exit_times[i_sim] = sc_fitting.metric_veh_exit_time(sim)
            ped_exit_times[np.isnan(ped_exit_times)] = 22
            veh_exit_times[np.isnan(veh_exit_times)] = 22
            ax.hist(ped_exit_times, bins = np.arange(23), color='b', alpha=0.5)
            ax.hist(veh_exit_times, bins = np.arange(23), color='g', alpha=0.5)
            ax.set_ylim(0, 30)
            
 
                
        
                
        
                
                
                
            