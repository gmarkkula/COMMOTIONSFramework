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
import math
import itertools
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sc_scenario
import sc_fitting
import sc_plot


PLOT_INTERACTION_OUTCOMES = False
OVERWRITE_SAVED_SIM_RESULTS_IO = False

PLOT_HIKER_CIT_CDFS = False

PLOT_DSS_CROSS_PROBS = True
OVERWRITE_SAVED_SIM_RESULTS_DSS = True

PARALLEL = True

MODEL_NAME = 'oVAoBEvoAIoEAoSNvoPF'

SCENARIO_END_TIME = 20
    
SCENARIOS_IO = (sc_fitting.PROB_FIT_SCENARIOS['Encounter'], 
                sc_fitting.PROB_FIT_SCENARIOS['EncounterPedPrio'],
                sc_fitting.PROB_FIT_SCENARIOS['PedLead'])

def get_dss_scen_name(zebra, gap):
    if zebra:
        return f'z{gap}'
    else:
        return f'nz{gap}'

DSS_GAPS = np.arange(3, 8)   
PED_STEP_DELAY = 1 # s
PED_START_MARGIN = 2 # m
SCENARIOS_DSS = []
for zebra in (False, True):
    for gap in DSS_GAPS:
        scen_name = get_dss_scen_name(zebra, gap)
        SCENARIOS_DSS.append(sc_fitting.SCPaperScenario(
            scen_name, initial_ttcas = (math.nan, gap - PED_STEP_DELAY),  
            ped_start_standing = True, ped_standing_margin = PED_START_MARGIN,
            ped_prio = zebra,
            time_step = sc_fitting.PROB_SIM_TIME_STEP,
            end_time = SCENARIO_END_TIME,
            stop_criteria = (sc_scenario.SimStopCriterion.BOTH_AGENTS_EXITED_CS,),
            metric_names = ('ped_entry_time', 'ped_exit_time',
                            'veh_entry_time', 'veh_exit_time')))

N_PARAMETERISATIONS_IO = 10
N_PARAMETERISATIONS_DSS = 30

SIM_RESULTS_FNAME_IO = 'fig_4_SimResults_InteractionOutcomes.pkl'
SIM_RESULTS_FNAME_DSS = 'fig_4_SimResults_DSS.pkl'

ALPHA = 0.1


def run_one_sim(model_name, i_sim, params_dict, scenario, n_sims):
    print(f'Simulating {model_name} with parameterisation'
          f' #{i_sim+1}/{n_sims}:'
          f' {params_dict} in scenario {scenario.name}...\n')
    return sc_fitting.construct_model_and_simulate_scenario(
        model_name, params_dict, scenario, apply_stop_criteria=False,
        zero_acc_after_exit=False)


def get_sim_results(file_name, overwrite, n_parameterisations, 
                    scenarios, exclude_non_progress=True):
    
    if sc_fitting.results_exist(file_name) and not overwrite:
        
        sims = sc_fitting.load_results(file_name)
        
    else:
        
        # load info on retained combined models - and get as dict instead
        ret_models = {}
        ret_models_tmp = sc_fitting.load_results(sc_fitting.RETAINED_COMB_FNAME)
        for ret_model in ret_models_tmp:
            ret_models[ret_model.model] = ret_model
        del ret_models_tmp
        ret_model = ret_models[MODEL_NAME]
            
        # initialise random number generator
        rng = np.random.default_rng(seed=0)
            
        # draw retained parameterisations at random and simulate
        if PARALLEL:
            print('Starting pool of workers...')
            pool = mp.Pool(mp.cpu_count()-1)
        n_ret_paramets = ret_model.params_array.shape[0]
        sims = {}
        # - get list of parameterisations
        params_dicts = []
        for i_sim in range(n_parameterisations):
            idx_paramet = rng.integers(n_ret_paramets)
            params_array = ret_model.params_array[idx_paramet, :]
            params_dicts.append(dict(
                zip(ret_model.param_names, params_array)))
        # - loop through scenarios and run simulations
        for scenario in scenarios:
            scenario.end_time = SCENARIO_END_TIME
            sim_iter = ((MODEL_NAME, i, params_dicts[i], scenario, 
                         n_parameterisations) 
                        for i in range(n_parameterisations))
            if PARALLEL:
                sims[scenario.name] = list(
                    pool.starmap(run_one_sim, sim_iter)) 
            else:
                sims[scenario.name] = list(
                    itertools.starmap(run_one_sim, sim_iter))
        # save simulation results
        sc_fitting.save_results(sims, file_name)
    
    if exclude_non_progress:
        # excluding parameterisations which get stuck (either agent not entering 
        # the conflict space in one or more scenarios)
        i_sim = 0
        n_excluded = 0
        n_sims = len(sims[scenarios[0].name])
        while i_sim <= n_sims - 1:
            exclude_sim = False
            for scenario in scenarios:
                sim = sims[scenario.name][i_sim]
                ped_entry_time = sc_fitting.metric_ped_entry_time(sim)
                veh_entry_time = sc_fitting.metric_veh_entry_time(sim)
                if np.isnan(ped_entry_time) or np.isnan(veh_entry_time):
                    exclude_sim = True
                    break
            if exclude_sim:
                for scenario in scenarios:
                    sims[scenario.name].pop(i_sim)
                n_excluded += 1
                n_sims -= 1
            else:
                i_sim += 1
        print(f'Excluded {n_excluded} parameterisations for {MODEL_NAME}.')
    
    return sims



if __name__ == '__main__':
    
    
    plt.close('all')

    fig, ax = plt.subplots(num='Model illustration')
    model_im = plt.imread(sc_plot.FIGS_FOLDER + 'oVAoBEvoAIoEAoSNvoPF.emf')
    ax.imshow(model_im)
    ax.axis('off')
        
    
    if PLOT_INTERACTION_OUTCOMES:
        
        # get simulation results (load existing results or run simulations)
        sims = get_sim_results(SIM_RESULTS_FNAME_IO,
                               OVERWRITE_SAVED_SIM_RESULTS_IO,
                               N_PARAMETERISATIONS_IO, 
                               SCENARIOS_IO)

        # do plotting
        N_ROWS = 3
        fig, axs = plt.subplots(nrows=N_ROWS, ncols=len(SCENARIOS_IO), figsize=(6, 8),
                                sharex='row', sharey='row', tight_layout=True,
                                num='Interactive model simulations')
        print(f'Plotting for {MODEL_NAME}...')
        axs = axs.reshape((N_ROWS, len(SCENARIOS_IO)))
        all_pets = np.empty(0)
        for i_scenario, scenario in enumerate(SCENARIOS_IO):
            
            n_sims = len(sims[scenario.name])
            
            # distance-distance plot
            ax = axs[0, i_scenario]
            for sim in sims[scenario.name]:
                veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
                ped_agent = sim.agents[sc_fitting.i_PED_AGENT]
                ax.plot(-veh_agent.signed_CP_dists, -ped_agent.signed_CP_dists, 'k',
                        alpha=ALPHA)
                ax.plot(-veh_agent.signed_CP_dists[-1], -ped_agent.signed_CP_dists[-1], 'ro',
                        alpha=ALPHA)
                # if sc_fitting.metric_collision(sim):
                #     sim.do_plots(kinem_states=True)
            ax.set_xlim(-50, 50)
            ax.set_xlabel('Vehicle position (m)')
            ax.set_ylim(-6, 6)
            ax.set_ylabel('Pedestrian position (m)')
            ax.set_title(scenario.name)
            ax.fill(np.array((1, 1, -1, -1)) * veh_agent.coll_dist, 
                    np.array((-1, 1, 1, -1)) * ped_agent.coll_dist, color='r',
                    alpha=0.3, edgecolor=None)
  
            # vehicle speed
            ax = axs[1, i_scenario]
            for sim in sims[scenario.name]:
                veh_agent = sim.agents[sc_fitting.i_VEH_AGENT]
                ax.plot(sim.time_stamps, veh_agent.trajectory.long_speed, 
                        'k-', alpha=ALPHA)
                ax.set_ylim(-1, 18)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Vehicle speed (m/s)')
                
            # pedestrian speed
            ax = axs[2, i_scenario]
            for sim in sims[scenario.name]:
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
            for i_sim, sim in enumerate(sims[scenario.name]):
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
            
            plt.show()
                    
            
    
    if PLOT_HIKER_CIT_CDFS:
        print('Loading HIKER CIT data...')
        with open(sc_fitting.DATA_FOLDER + '/' + sc_fitting.HIKER_DATA_FILE_NAME,
                  'rb') as file_obj:
            observed_cits = pickle.load(file_obj)
        sc_fitting.do_hiker_cit_cdf_plot(observed_cits, fig_name='Observed CITs')
        model_cits = sc_fitting.load_results(
            sc_fitting.MODEL_CIT_FNAME_FMT % MODEL_NAME)
        sc_fitting.do_hiker_cit_cdf_plot(model_cits, fig_name='Model CITs', legend=False)
       
        
    if PLOT_DSS_CROSS_PROBS:
        
        # get model results (load existing results or run simulations)
        sims = get_sim_results(SIM_RESULTS_FNAME_DSS,
                               OVERWRITE_SAVED_SIM_RESULTS_DSS,
                               N_PARAMETERISATIONS_DSS, 
                               SCENARIOS_DSS)
        # get empirical results
        dss_df = pd.read_csv(sc_fitting.DATA_FOLDER + '/DSS_outcomes.csv')
        # plot
        fig, axs = plt.subplots(ncols=2, nrows=1,
                               num='DSS crossing probs', figsize=(8, 4),
                               tight_layout=True)
        for i_source, source in enumerate(('data', 'model')):
            ax = axs[i_source]
            ped_cross = {}
            for zebra in (True, False):
                ped_cross[zebra] = np.zeros(len(DSS_GAPS))
                for i_gap, gap in enumerate(DSS_GAPS):
                    scen_name = get_dss_scen_name(zebra, gap)
                    if source == 'data':
                        scenario_rows = ((dss_df['TTA'] == gap)
                                         & (dss_df['zebra'] == int(zebra)))
                        scenario_ped_first = dss_df['ped_crossed_first'][scenario_rows]
                        ped_cross[zebra][i_gap] = (scenario_ped_first.sum()
                                                   / len(scenario_ped_first))
                    else:
                        for i_sim, sim in enumerate(sims[scen_name]):
                            ped_entry_time = sc_fitting.metric_ped_entry_time(sim)
                            veh_entry_time = sc_fitting.metric_veh_entry_time(sim)
                            if ped_entry_time < veh_entry_time:
                                ped_cross[zebra][i_gap] += 1
                        ped_cross[zebra][i_gap] /= len(sims[scen_name])
                if zebra:
                    ls = '--'
                else:
                    ls = '-'
                ax.plot(DSS_GAPS, ped_cross[zebra], 'k-o', ls=ls)
            ax.legend(('Zebra', 'Non-zebra'))
            ax.set_xlabel('Time gap (s)')
            ax.set_ylabel('Prop. ped. crossing first (-)')
        

        
                
            