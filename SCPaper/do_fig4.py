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
from sc_fitting import i_PED_AGENT, i_VEH_AGENT
import sc_plot



SAVE_PDF = True
if SAVE_PDF:
    SCALE_DPI = 1
    text_y_nudge = 0.02 # not sure why this is needed
else:
    SCALE_DPI = 0.5
    text_y_nudge = 0

PLOT_INTERACTION_OUTCOMES = True
OVERWRITE_SAVED_SIM_RESULTS_IO = False

PLOT_HIKER_CIT_CDFS = True

PLOT_DSS_CROSS_PROBS = True
OVERWRITE_SAVED_SIM_RESULTS_DSS = False

PARALLEL = True

MODEL_NAME = 'oVAoBEvoAIoEAoSNvoPF'

SCENARIO_END_TIME = 20
    
SCENARIOS_IO = (sc_fitting.PROB_FIT_SCENARIOS['Encounter'], 
                sc_fitting.PROB_FIT_SCENARIOS['EncounterPedPrio'],
                sc_fitting.PROB_FIT_SCENARIOS['PedLead'])
SCENARIO_DISPLAY_NAMES_IO = ('Encounter', 'Encounter w. ped. prio.', 'Pedestrian lead')

N_HIKER_GAPS = 4

def get_dss_scen_name(zebra, gap):
    if zebra:
        return f'z{gap}'
    else:
        return f'nz{gap}'

DSS_GAPS = np.arange(3, 8)   
PED_STEP_DELAY = 1 # s
PED_START_MARGIN = 1.95 # m
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

N_PARAMETERISATIONS_IO = 50
N_PARAMETERISATIONS_DSS = 500

SIM_RESULTS_FNAME_IO = 'fig_4_SimResults_InteractionOutcomes.pkl'
SIM_RESULTS_FNAME_DSS = 'fig_4_SimResults_DSS.pkl'

ALPHA = 0.1

PLOT_REJECTED = False

def run_one_sim(model_name, i_sim, params_dict, scenario, n_sims):
    print(f'Simulating {model_name} with parameterisation'
          f' #{i_sim+1}/{n_sims}:'
          f' {params_dict} in scenario {scenario.name}...\n')
    return sc_fitting.construct_model_and_simulate_scenario(
        model_name, params_dict, scenario, apply_stop_criteria=False,
        zero_acc_after_exit=False)


def get_sim_results(file_name, overwrite, n_parameterisations, 
                    scenarios, exclude_non_progress=True,
                    excl_params_file_name=None):
    
    if sc_fitting.results_exist(file_name) and not overwrite:
        
        sim_results = sc_fitting.load_results(file_name)
        
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
        sim_results = {}
        sim_results['n_parameterisations'] = n_parameterisations
        # - get list of parameterisations
        params_dicts = []
        for i_sim in range(n_parameterisations):
            idx_paramet = rng.integers(n_ret_paramets)
            params_array = ret_model.params_array[idx_paramet, :]
            params_dicts.append(dict(
                zip(ret_model.param_names, params_array)))
        sim_results['parameterisations'] = params_dicts
        # - loop through scenarios and run simulations
        for scenario in scenarios:
            scenario.end_time = SCENARIO_END_TIME
            sim_iter = ((MODEL_NAME, i, params_dicts[i], scenario, 
                         n_parameterisations) 
                        for i in range(n_parameterisations))
            if PARALLEL:
                sims = list(
                    pool.starmap(run_one_sim, sim_iter)) 
            else:
                sims = list(
                    itertools.starmap(run_one_sim, sim_iter))
            # get and store needed info from simulations
            n_time_steps = len(sims[0].time_stamps)
            sim_results[scenario.name] = {}
            sim_results[scenario.name]['time_stamps'] = sims[0].time_stamps
            for i_agent in range(sc_scenario.N_AGENTS):
                sim_results[scenario.name][i_agent] = {}
                sim_results[scenario.name][i_agent]['coll_dist'] = \
                    sims[0].agents[i_agent].coll_dist
                # allocate arrays
                sim_results[scenario.name][i_agent]['cp_dist'] = np.full(
                    (n_parameterisations, n_time_steps), np.nan)
                sim_results[scenario.name][i_agent]['speed'] = np.full(
                    (n_parameterisations, n_time_steps), np.nan)
                sim_results[scenario.name][i_agent]['entry_time'] = np.full(
                    n_parameterisations, np.nan)
                # get and store data from individual simulations
                for i_sim, sim in enumerate(sims):
                    sim_results[scenario.name][i_agent]['cp_dist'][i_sim, :] = \
                        sim.agents[i_agent].signed_CP_dists
                    sim_results[scenario.name][i_agent]['speed'][i_sim, :] = \
                        sim.agents[i_agent].trajectory.long_speed
                    sim_results[scenario.name][i_agent]['entry_time'][i_sim] = \
                        sc_fitting.metric_agent_entry_time(sim, i_agent)
        # save simulation results
        sc_fitting.save_results(sim_results, file_name)
    
    n_sims = sim_results['n_parameterisations']
    if exclude_non_progress:
        # excluding parameterisations which get stuck (either agent not entering 
        # the conflict space in one or more scenarios)
        i_sim = 0
        rejected = np.full(n_sims, False)
        n_non_progress = np.zeros(n_sims)
        for i_sim in range(n_sims):
            for scenario in scenarios:
                ped_entry_time = sim_results[scenario.name][
                    sc_fitting.i_PED_AGENT]['entry_time'][i_sim]
                veh_entry_time = sim_results[scenario.name][
                    sc_fitting.i_VEH_AGENT]['entry_time'][i_sim]
                if np.isnan(ped_entry_time) or np.isnan(veh_entry_time):
                    rejected[i_sim] = True
                    n_non_progress[i_sim] += 1
                    if PLOT_REJECTED:
                        plot_sim(sim_results, scenario, i_sim)
        sim_results['idx_retained'] = np.nonzero(~rejected)[0]
        n_rejected = np.count_nonzero(rejected)
        print(f'Rejected {n_rejected} parameterisations for {MODEL_NAME}.')
        if excl_params_file_name != None:
            # save info on the rejected parameterisations
            params_dicts = sim_results['parameterisations']
            param_names = params_dicts[0].keys()
            n_params = len(param_names)
            full_params_array = np.full((len(params_dicts), n_params), np.nan)
            #rej_params_array = np.full((n_rejected, n_params), np.nan)
            #i_rej_param = 0
            for i_sim in range(n_sims):
                for i_param, param_name in enumerate(param_names):
                    full_params_array[i_sim, i_param] = \
                            params_dicts[i_sim][param_name]
            excl_params = {}
            excl_params[MODEL_NAME] = {}
            excl_params[MODEL_NAME]['params_array'] = full_params_array
            excl_params[MODEL_NAME]['n_non_progress'] = n_non_progress
            excl_params[MODEL_NAME]['rejected'] = rejected
            sc_fitting.save_results(excl_params, excl_params_file_name)
    else:
        sim_results['idx_retained'] = np.arange(n_sims)
    
    
    return sim_results


def plot_sim(sim_results, scenario, i_sim):
    print('*** Plotting example ***')
    print(scenario.name)
    print(sim_results['parameterisations'][i_sim])
    sim_fig, sim_axs = plt.subplots(nrows=2, ncols=1)
    for i_agent in range(2):
        sim_axs[0].plot(sim_results[scenario.name]['time_stamps'], 
                        sim_results[scenario.name][i_agent]['speed'][i_sim, :])
        sim_axs[1].plot(sim_results[scenario.name]['time_stamps'], 
                        sim_results[scenario.name][i_agent]['cp_dist'][i_sim, :])
        sim_axs[1].set_ylim(-5, 5)
    plt.show()
    input('Press Enter to continue...')
    


if __name__ == '__main__':
    
    
    plt.close('all')
    
    
    fig, axs = plt.subplots(nrows=3, ncols=7,
                            figsize=(sc_plot.FULL_WIDTH, 
                                     0.5*sc_plot.FULL_WIDTH), 
                            dpi=sc_plot.DPI * SCALE_DPI)
        
    
    if PLOT_INTERACTION_OUTCOMES:
        
        # get simulation results (load existing results or run simulations)
        sim_results = get_sim_results(SIM_RESULTS_FNAME_IO,
                               OVERWRITE_SAVED_SIM_RESULTS_IO,
                               N_PARAMETERISATIONS_IO, 
                               SCENARIOS_IO)

        # do plotting
        N_ROWS = 3
        AX_W = 0.09
        AX_H = 0.15
        for i_scenario, scenario in enumerate(SCENARIOS_IO):
                        
            # distance-distance plot
            ax = axs[0, i_scenario]
            ax_x = 0.62 + 0.13 * i_scenario
            for idx in sim_results['idx_retained']:
                veh_cp_dist = sim_results[scenario.name][i_VEH_AGENT]['cp_dist'][idx, :]
                ped_cp_dist = sim_results[scenario.name][i_PED_AGENT]['cp_dist'][idx, :]
                ax.plot(-veh_cp_dist, -ped_cp_dist, 'k', alpha=ALPHA)
                ax.plot(-veh_cp_dist[-1], -ped_cp_dist[-1], 'ro', alpha=ALPHA)
                # if sc_fitting.metric_collision(sim):
                #     sim.do_plots(kinem_states=True)
            if i_scenario == 2:
                ax.set_xlim(-125, 15)
            else:
                ax.set_xlim(-50, 50)
            if i_scenario == 1:
                ax.set_xlabel('Vehicle position (m)')
            ax.set_ylim(-6, 6)
            if i_scenario == 0:
                ax.set_ylabel('Ped. position (m)')
            ax.set_title(SCENARIO_DISPLAY_NAMES_IO[i_scenario] + '\n', 
                         fontsize=sc_plot.DEFAULT_FONT_SIZE)
            veh_coll_dist = sim_results[scenario.name][i_VEH_AGENT]['coll_dist']
            ped_coll_dist = sim_results[scenario.name][i_PED_AGENT]['coll_dist']
            ax.fill(np.array((1, 1, -1, -1)) * veh_coll_dist, 
                    np.array((-1, 1, 1, -1)) * ped_coll_dist, color='r',
                    alpha=0.3, edgecolor=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax_y = 0.65
            ax.set_position((ax_x, ax_y, AX_W, AX_H))
  
            # vehicle speed
            time_stamps = sim_results[scenario.name]['time_stamps']
            ax = axs[1, i_scenario]
            for idx in sim_results['idx_retained']:
                veh_speed = sim_results[scenario.name][i_VEH_AGENT]['speed'][idx, :]
                ax.plot(time_stamps, veh_speed, 'k-', alpha=ALPHA)
            ax.set_ylim(-1, 18)
            ax.set_xlabel('Time (s)')
            if i_scenario == 0:
                ax.set_ylabel('Veh. speed (m/s)')
            sc_plot.leave_only_yaxis(ax)
            ax_y = 0.38
            ax.set_position((ax_x, ax_y, AX_W, AX_H))
                
            # pedestrian speed
            ax = axs[2, i_scenario]
            for idx in sim_results['idx_retained']:
                ped_speed = sim_results[scenario.name][i_PED_AGENT]['speed'][idx, :]
                ax.plot(time_stamps, ped_speed, 'k-', alpha=ALPHA)
            ax.set_ylim(-.1, 4.1)
            ax.set_xlabel('Time (s)')
            if i_scenario == 0:
                ax.set_ylabel('Ped. speed (m/s)')
            sc_plot.leave_only_yaxis(ax) 
            ax_y = 0.18
            ax.set_position((ax_x, ax_y, AX_W, AX_H))
            if i_scenario == 1:
                xlabel = 'Time (s)'
            else:
                xlabel = ''
            sc_plot.add_linked_time_axis(ax, label=xlabel)
        
            
    
    if PLOT_HIKER_CIT_CDFS:
        AX_W = 0.08
        AX_H = 0.13
        for i_source, source in enumerate(('data', 'model')):
            # get CITs to plot
            if source == 'data':
                # observed CDFs
                with open(sc_fitting.DATA_FOLDER + '/' + sc_fitting.HIKER_DATA_FILE_NAME,
                          'rb') as file_obj:
                    cits = pickle.load(file_obj)
            else:
                # model CDFs
                cits = sc_fitting.load_results(
                    sc_fitting.MODEL_CIT_FNAME_FMT % MODEL_NAME)
            # plot
            cit_axs = []
            for i_gap in range(len(sc_fitting.HIKER_VEH_TIME_GAPS)):
                cit_axs.append(axs[i_source, 3+i_gap])
            sc_fitting.do_hiker_cit_cdf_plot(cits, axs=cit_axs, legend=(i_source==1),
                                             titles=False, finalise=False,
                                             legend_kwargs={'frameon': False,
                                                            'fontsize': 
                                                                sc_plot.DEFAULT_FONT_SIZE-1,
                                                            'loc': (1, 1.3)})
            for i_ax, ax in enumerate(cit_axs):
                sc_plot.leave_only_yaxis(ax) 
                ax_x = 0.06 + i_ax * 0.10
                ax_y = 0.71 - i_source * 0.15
                ax.set_position((ax_x, ax_y, AX_W, AX_H))
                if i_source == 0:
                    ax.set_title(f'Gap {sc_fitting.HIKER_VEH_TIME_GAPS[i_ax]} s',
                                 fontsize=sc_plot.DEFAULT_FONT_SIZE)
                    ylabel = 'Observed'
                else:
                    sc_plot.add_linked_time_axis(ax, label='')
                    ylabel = 'Model'
            cit_axs[0].set_ylabel(ylabel + '\nCDF (-)')
        plt.annotate('Crossing initiation time (s)', (0.26, 0.46 + text_y_nudge), 
                     xycoords='figure fraction', ha='center')

       
        
    if PLOT_DSS_CROSS_PROBS:
        
        AX_Y = 0.12
        AX_W = 0.15
        AX_H = 0.22
        # get model results (load existing results or run simulations)
        sim_results = get_sim_results(SIM_RESULTS_FNAME_DSS,
                               OVERWRITE_SAVED_SIM_RESULTS_DSS,
                               N_PARAMETERISATIONS_DSS, 
                               SCENARIOS_DSS, 
                               excl_params_file_name=sc_fitting.EXCL_DSS_FNAME)
        # get empirical results
        dss_df = pd.read_csv(sc_fitting.DATA_FOLDER + '/DSS_outcomes.csv')
        # plot
        for i_source, source in enumerate(('data', 'model')):
            ax = axs[2, 3+i_source]
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
                        for idx in sim_results['idx_retained']:
                            ped_entry_time = sim_results[scen_name][
                                sc_fitting.i_PED_AGENT]['entry_time'][idx]
                            veh_entry_time = sim_results[scen_name][
                                sc_fitting.i_VEH_AGENT]['entry_time'][idx]
                            if ped_entry_time < veh_entry_time:
                                ped_cross[zebra][i_gap] += 1
                        ped_cross[zebra][i_gap] /= len(sim_results['idx_retained'])
                if zebra:
                    ls = ':'
                    color = 'gray'
                else:
                    ls = '-'
                    color = 'black'
                ax.plot(DSS_GAPS, ped_cross[zebra], 'k-o', ls=ls, color=color,
                        ms=4)
            ax.set_xlabel('Gap (s)')
            if i_source == 0:
                ax.set_ylabel('$P$(pedestrian first) (-)')
                ax.legend(('Zebra', 'Non-zebra'), frameon=False, 
                          loc=(0.65, 0.14), fontsize=sc_plot.DEFAULT_FONT_SIZE-1)
                title = 'Observed'
            else:
                title = 'Model'
            ax.set_title(title, fontsize=sc_plot.DEFAULT_FONT_SIZE)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax_x = 0.10 + i_source * 0.23
            ax.set_position((ax_x, AX_Y, AX_W, AX_H))
            
        # hide unused subplots
        for i_col in range(5, 7):
            axs[2, i_col].set_visible(False)

        
                
    # add panel labels
    sc_plot.add_panel_label('B', (0.60, 0.87 + text_y_nudge))
    sc_plot.add_panel_label('A', (0.01, 0.88 + text_y_nudge))
    sc_plot.add_panel_label('C', (0.04, 0.38 + text_y_nudge))
    
    
    if SAVE_PDF:
        file_name = sc_plot.FIGS_FOLDER + 'fig4.pdf'
        print(f'Saving {file_name}...')
        plt.savefig(file_name, bbox_inches='tight')
        
    
    plt.show()
            