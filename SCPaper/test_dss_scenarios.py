# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:35:10 2022

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
    
import math
import itertools
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sc_scenario
import sc_fitting


OVERWRITE_SAVED_SIM_RESULTS = False

PARALLEL = True

PED_STEP_DELAY = 1
SCENARIO_END_TIME = 20
ALPHA = 0.1



def get_scen_name(zebra, gap):
    if zebra:
        return f'z{gap}'
    else:
        return f'nz{gap}'

GAPS = np.arange(3, 8)    

SCENARIOS = {}
for zebra in (False, True):
    for gap in GAPS:
        scen_name = get_scen_name(zebra, gap)
        SCENARIOS[scen_name] = sc_fitting.SCPaperScenario(
            scen_name, initial_ttcas = (math.nan, gap - PED_STEP_DELAY),  
            ped_start_standing = True, ped_standing_margin = 2,
            ped_prio = zebra,
            time_step = sc_fitting.PROB_SIM_TIME_STEP,
            end_time = SCENARIO_END_TIME,
            stop_criteria = (sc_scenario.SimStopCriterion.BOTH_AGENTS_EXITED_CS,),
            metric_names = ('ped_entry_time', 'ped_exit_time',
                            'veh_entry_time', 'veh_exit_time'))



MODEL_NAMES = ('oVAoBEvoAIoEAoSNvoPF',)
N_PARAMETS_PER_MODEL_AND_SCEN = 50
SIM_RESULTS_FNAME = 'fig_dss_SimResults.pkl'


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
            for scenario in SCENARIOS.values():
                sim_iter = ((model_name, i, params_dicts[i], scenario) 
                            for i in range(N_PARAMETS_PER_MODEL_AND_SCEN))
                if PARALLEL:
                    sims[model_name][scenario.name] = list(
                        pool.starmap(run_one_sim, sim_iter))   
                else:
                    sims[model_name][scenario.name] = list(
                        itertools.starmap(run_one_sim, sim_iter))   
                # sim = sc_fitting.construct_model_and_simulate_scenario(
                #     model_name, params_dict, scenario, apply_stop_criteria=False,
                #     zero_acc_after_exit=False)
                #sims[model_name][scenario_name].append(sim)
        # save simulation results
        sc_fitting.save_results(sims, SIM_RESULTS_FNAME)

        
    for model_name in MODEL_NAMES:
        i_sim = 0
        n_excluded = 0
        first_scen_name = tuple(SCENARIOS.keys())[0]
        n_sims = len(sims[model_name][first_scen_name])
        while i_sim <= n_sims - 1:
            exclude_sim = False
            for scenario in SCENARIOS.values():
                sim = sims[model_name][scenario.name][i_sim]
                ped_entry_time = sc_fitting.metric_ped_entry_time(sim)
                veh_entry_time = sc_fitting.metric_veh_entry_time(sim)
                if np.isnan(ped_entry_time) or np.isnan(veh_entry_time):
                    exclude_sim = True
                    break
            if exclude_sim:
                for scenario in SCENARIOS.values():
                    sims[model_name][scenario.name].pop(i_sim)
                n_excluded += 1
                n_sims -= 1
            else:
                i_sim += 1
        print(f'Excluded {n_excluded} parameterisations for {model_name}.')
        
    plt.close('all')
    
    N_ROWS = 5
    for model_name in MODEL_NAMES:
        fig, axs = plt.subplots(nrows=N_ROWS, ncols=len(SCENARIOS), figsize=(15, 8),
                                sharex='row', sharey='row', tight_layout=True,
                                num=model_name)
        axs = axs.reshape((N_ROWS, len(SCENARIOS)))
        all_pets = np.empty(0)
        for i_scenario, scenario in enumerate(SCENARIOS.values()):
            
            n_sims = len(sims[model_name][scenario.name])
            
            # distance-distance plot
            ax = axs[0, i_scenario]
            for sim in sims[model_name][scenario.name]:
                veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
                ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
                ax.plot(-veh_agent.signed_CP_dists, -ped_agent.signed_CP_dists, 'k',
                        alpha=ALPHA)
                ax.plot(-veh_agent.signed_CP_dists[-1], -ped_agent.signed_CP_dists[-1], 'ro',
                        alpha=ALPHA)
                # if sc_fitting.metric_collision(sim):
                #     sim.do_plots(kinem_states=True)
            ax.set_xlim(-50, 50)
            ax.set_xlabel('Vehicle pos. (m)')
            ax.set_ylim(-6, 6)
            ax.set_ylabel('Pedestrian pos. (m)')
            ax.set_title(scenario.name)
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
                
            # get entry and exit times and pets
            ped_entry_times = np.full(n_sims, np.nan)
            veh_entry_times = np.full(n_sims, np.nan)
            ped_exit_times = np.full(n_sims, np.nan)
            veh_exit_times = np.full(n_sims, np.nan)
            pets = np.full(n_sims, np.nan)
            for i_sim, sim in enumerate(sims[model_name][scenario.name]):
                ped_entry_times[i_sim] = sc_fitting.metric_ped_entry_time(sim)
                veh_entry_times[i_sim] = sc_fitting.metric_veh_entry_time(sim)
                ped_exit_times[i_sim] = sc_fitting.metric_ped_exit_time(sim)
                veh_exit_times[i_sim] = sc_fitting.metric_veh_exit_time(sim)
                # get pet
                if np.isnan(ped_entry_times[i_sim]) or np.isnan(veh_entry_times[i_sim]):
                    pets[i_sim] = np.nan
                elif ped_entry_times[i_sim] < veh_entry_times[i_sim]:
                    # ped entering before veh
                    pets[i_sim] = veh_entry_times[i_sim] - ped_exit_times[i_sim]
                else:
                    # veh entering before ped
                    pets[i_sim] = ped_entry_times[i_sim] - veh_exit_times[i_sim]      
            all_pets = np.concatenate((all_pets, pets))
                
            # exit time distributions
            ax = axs[3, i_scenario]
            ped_exit_times[np.isnan(ped_exit_times)] = 22
            veh_exit_times[np.isnan(veh_exit_times)] = 22
            ax.hist(ped_exit_times, bins = np.arange(23), color='b', alpha=0.5)
            ax.hist(veh_exit_times, bins = np.arange(23), color='g', alpha=0.5)
            ax.set_ylim(0, 30)
            if i_scenario == 0:
                ax.legend(('Ped.', 'Veh.'))
            ax.set_xlabel('CS exit time (s)')
            ax.set_ylabel('Count (-)')
            
            # pet distribution
            ax = axs[4, i_scenario]
            #ax.hist(pets, bins=np.arange(0, 10, 0.5))
            sns.ecdfplot(pets, ax=ax)
            ax.set_xlim(0, 6)
            ax.set_xlabel('PET (s)')
 
        
        fig, ax = plt.subplots(num='Ped cross ' + model_name, figsize=(4, 4),
                               tight_layout=True)
        ped_cross = {}
        for zebra in (True, False):
            ped_cross[zebra] = np.zeros(len(GAPS))
            for i_gap, gap in enumerate(GAPS):
                scen_name = get_scen_name(zebra, gap)
                for i_sim, sim in enumerate(sims[model_name][scen_name]):
                    ped_entry_time = sc_fitting.metric_ped_entry_time(sim)
                    veh_entry_time = sc_fitting.metric_veh_entry_time(sim)
                    if ped_entry_time < veh_entry_time:
                        ped_cross[zebra][i_gap] += 1
                ped_cross[zebra][i_gap] /= len(sims[model_name][scen_name])
            if zebra:
                ls = '--'
            else:
                ls = '-'
            ax.plot(GAPS, ped_cross[zebra], 'k-o', ls=ls)
        ax.legend(('Zebra', 'Non-zebra'))
        ax.set_xlabel('Time gap (s)')
        ax.set_ylabel('Prop. ped. crossing first (-)')
        
    