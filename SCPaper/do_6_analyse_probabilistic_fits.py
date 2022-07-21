# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:02:40 2021

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
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import collections
import parameter_search
import sc_fitting
import sc_plot


ExampleParameterisation = collections.namedtuple(
    'ExampleParameterisation',['i_parameterisation', 'params_array', 
                                'params_dict', 'crit_dict'])

# constants
DO_PLOTS = True # if False, all plots are disabled
DO_TIME_SERIES_PLOTS = True
N_CRIT_FOR_TS_PLOT = 5
DO_PARAMS_PLOTS = False
DO_RETAINED_PARAMS_PLOT = True # supplementary figure
DO_CRIT_PLOT = False # supplementary figure
CRIT_PLOT_MODELS = ('oVAoAN', 'oVAoEAoAN', 'oVAoDAoAN',
                     'oVAoSNc', 'oVAoEAoSNc', 'oVAoDAoSNc', 
                     'oVAoSNcoPF', 'oVAoEAoSNcoPF', 'oVAoDAoSNcoPF',
                     'oVAoSNv', 'oVAoEAoSNv', 'oVAoDAoSNv', 
                     'oVAoSNvoPF', 'oVAoEAoSNvoPF', 'oVAoDAoSNvoPF') # all models - just reordered
CRIT_PLOT_FIG_NO = 13
DO_OUTCOME_PLOT = False
N_CRIT_FOR_PARAMS_PLOT = 4
SAVE_FIGS = False
RET_PARAMS_PLOTS_TO_SAVE = ('oVAoEAoSNv', 'oVAoEAoSNvoPF')
RET_PARAMS_PLOT_FIRST_FIG_NO = 14
MODELS_TO_ANALYSE = 'all'
ASSUMPTIONS_TO_NOT_ANALYSE = 'none'
HESITATION_SPEED_FRACT = 0.95
CRITERIA = ('Collision-free encounter', 
            'Collision-free encounter with pedestrian priority', 
            'Collision-free pedestrian lead situation', 
            'Pedestrian hesitation in constant-speed scenario',
            'Pedestrian progress in constant-speed scenario')
N_INTERACTIVE_SCENARIOS = 3
PED_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_PED_AGENT]
VEH_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_VEH_AGENT]
PARAMS_JITTER = 0.015
#N_MAIN_CRIT_FOR_RETAINING = 3



def do(prob_fit_file_name_fmt, retained_fits_file_name,
       ylabel_rotation='vertical'):

    # find pickle files from probabilistic fitting
    prob_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                                (prob_fit_file_name_fmt % '*'))
    prob_fit_files.sort()
    print(prob_fit_files)
    
    # need to prepare for criterion plot?
    if DO_PLOTS:
        plt.close('all')
        if SAVE_FIGS:
            SCALE_DPI = 1
        else:
            SCALE_DPI = 0.5
        if DO_CRIT_PLOT:
            if CRIT_PLOT_MODELS == 'all':
                nrows = len(prob_fit_files)
            else:
                nrows = len(CRIT_PLOT_MODELS)
            fig_height = 2 + nrows
            crit_fig, crit_axs = plt.subplots(nrows=nrows, 
                                              ncols=len(CRITERIA),
                                              sharex='col', sharey=True, 
                                              figsize=(1.6*sc_plot.FULL_WIDTH, 
                                                       fig_height), 
                                              tight_layout=True,
                                              dpi=sc_plot.DPI * SCALE_DPI)
    
    # loop through the fitting results files
    prob_fits = {}
    retained_models = []
    for i_prob_fit_file, prob_fit_file in enumerate(prob_fit_files):
        print()
        prob_fit = parameter_search.load(prob_fit_file, verbose=True)
        if ((not(MODELS_TO_ANALYSE == 'all') and not (prob_fit.name in MODELS_TO_ANALYSE))
            or ASSUMPTIONS_TO_NOT_ANALYSE in prob_fit.name):
            print(f'Skipping model {prob_fit.name}.')
            continue
        prob_fits[prob_fit.name] = prob_fit
        n_parameterisations = prob_fit.results.metrics_matrix.shape[0]
        print(f'Analysing model {prob_fit.name},'
              f' {n_parameterisations} parameterisations...')
        
        # calculate criterion vectors
        criteria_matrix = np.full((len(CRITERIA), n_parameterisations), False)
        for i_crit, crit in enumerate(CRITERIA):
            
            # prepare criterion plot axes?
            if DO_PLOTS and DO_CRIT_PLOT:
                do_this_crit_plot = True
                if CRIT_PLOT_MODELS == 'all':
                    i_row = i_prob_fit_file
                elif prob_fit.name in CRIT_PLOT_MODELS:
                    i_row = CRIT_PLOT_MODELS.index(prob_fit.name)
                else:
                    do_this_crit_plot = False
                if do_this_crit_plot:
                    ax = crit_axs[i_row, i_crit]
                    if i_row == 0:
                        if crit == 'Pedestrian hesitation in constant-speed scenario':
                            title_str = 'Gap acceptance hesitation'
                        else:
                            title_str = crit
                        ax.set_title(title_str, fontsize=8)
                    if i_crit == 0:
                        ax.set_ylabel(prob_fit.name, fontsize=8, 
                                      rotation=ylabel_rotation)
                else:
                    ax = None # just to make sure
            
            # criterion-specific calculations
            if 'Collision-free' in crit:
                # note that the next line of code assumes ordering of criteria is
                # the same as the ordering in sc_fitting.PROB_FIT_SCENARIOS
                coll_metric_name = list(prob_fit.scenarios.values())[
                    i_crit].get_full_metric_name('collision')
                collisions = prob_fit.get_metric_results(coll_metric_name)
                coll_free_rep = np.logical_not(collisions)
                # criterion met for parameterisation if no collisions for any of the repetitions
                crit_met = np.all(coll_free_rep, axis=1)
                # criterion plot?
                if DO_PLOTS and DO_CRIT_PLOT and do_this_crit_plot:
                    perc_coll_free_reps = 100 * (np.sum(coll_free_rep, axis=1)
                                                 / prob_fit.n_repetitions)
                    ecdf = ECDF(perc_coll_free_reps)
                    ax.step(ecdf.x, ecdf.y, 'r-', lw=1)
                    ax.set_xlim(-5, 105)
                    if i_row == nrows-1:
                        ax.set_xlabel('Collision-free repetitions (%)')
                
            elif crit == 'Pedestrian hesitation in constant-speed scenario':
                ped_av_speed = prob_fit.get_metric_results('PedHesitateVehConst_ped_av_speed_to_CS')
                crit_met_all = ((ped_av_speed 
                                 < HESITATION_SPEED_FRACT * PED_FREE_SPEED)
                                | np.isnan(ped_av_speed))
                # criterion met for parameterisation if met for enough of the repetitions
                crit_met = np.sum(crit_met_all, axis=1) >= 4
                # criterion plot?
                if DO_PLOTS and DO_CRIT_PLOT and do_this_crit_plot:
                    metric = ped_av_speed / PED_FREE_SPEED
                    metric_sorted = np.sort(metric, axis=1)
                    for n_reps in range(prob_fit.n_repetitions):
                        ecdf = ECDF(metric_sorted[:, n_reps])
                        ax.step(ecdf.x, ecdf.y, 'k-', lw=2, 
                                alpha=(n_reps+1)/prob_fit.n_repetitions)
                    ax.set_xlim(0.2, 1.2)
                    ax.axvline(HESITATION_SPEED_FRACT, ls=':', color='gray')
                    if i_row == nrows-1:
                        ax.set_xlabel('$\overline{v}_\mathrm{p}/v_\mathrm{p,free}$ (-)')
                        
            elif crit == 'Pedestrian progress in constant-speed scenario':
                ped_av_speed = prob_fit.get_metric_results('PedHesitateVehConst_ped_av_speed_to_CS')
                n_non_progress = np.count_nonzero(np.isnan(ped_av_speed), axis=1)
                crit_met = n_non_progress == 0
                if DO_PLOTS and DO_CRIT_PLOT and do_this_crit_plot:
                    ecdf = ECDF(100 * n_non_progress / prob_fit.n_repetitions)
                    ax.step(ecdf.x, ecdf.y, 'm-', lw=1)
                    if i_row == nrows-1:
                        ax.set_xlabel('Non-progress repetitions (%)')
            
            else:
                raise Exception(f'Unexpected criterion "{crit}".')
                
            criteria_matrix[i_crit, :] = crit_met
            # print some info
            n_crit_met = np.count_nonzero(crit_met)
            print(f'\t\t{crit}: Found {n_crit_met}'
                  f' ({100 * n_crit_met / n_parameterisations:.1f} %) parameterisations.') 
     
        
        # - look across multiple criteria
        all_criteria_met = np.all(criteria_matrix, axis=0)
        n_all_criteria_met = np.count_nonzero(all_criteria_met)
        print(f'\tAll criteria met: Found {n_all_criteria_met}'
              f' ({100 * n_all_criteria_met / n_parameterisations:.1f} %)'
              ' parameterisations.')  
        n_criteria_met = np.sum(criteria_matrix, axis=0)
        n_max_criteria_met = np.max(n_criteria_met)
        met_max_criteria = n_criteria_met == n_max_criteria_met
        n_met_max_criteria = np.count_nonzero(met_max_criteria)
        print(f'\tMax no of criteria met was {n_max_criteria_met},'
              f' for {n_met_max_criteria} parameterisations.')
        # -- NaNs
        print(f'\tNaNs in criteria: {np.sum(np.isnan(criteria_matrix), axis=1)}')
        # -- store these analysis results as object attributes
        prob_fit.criteria_matrix = criteria_matrix
        prob_fit.n_criteria_met = n_criteria_met
        
        # retain models and parameterisations meeting all criteria
        if n_max_criteria_met == len(CRITERIA):
            param_ranges = []
            for i_param in range(prob_fit.n_params):
                param_ranges.append((np.amin(prob_fit.results.params_matrix[:, i_param]),
                                     np.amax(prob_fit.results.params_matrix[:, i_param])))
            retained_models.append(sc_fitting.ModelWithParams(
                model=prob_fit.name, param_names=copy.copy(prob_fit.param_names), 
                param_ranges=param_ranges,
                params_array=np.copy(prob_fit.results.params_matrix[all_criteria_met])))
            retained_models[-1].tested_params_array = prob_fit.results.params_matrix
            retained_models[-1].idx_retained = all_criteria_met
        
        
        # pick a maximally sucessful parameterisations, and provide simulation 
        # plots if requested
        i_parameterisation = np.nonzero(met_max_criteria)[0][0]
        params_array = prob_fit.results.params_matrix[i_parameterisation, :]
        params_dict = prob_fit.get_params_dict(params_array)
        crit_dict = {crit : criteria_matrix[i_crit, i_parameterisation]
                     for i_crit, crit in enumerate(CRITERIA)}
        prob_fit.example = ExampleParameterisation(
            i_parameterisation=i_parameterisation, params_array=params_array,
            params_dict=params_dict, crit_dict=crit_dict)
        if n_max_criteria_met >= N_CRIT_FOR_TS_PLOT and DO_PLOTS and DO_TIME_SERIES_PLOTS:
            print('\tLooking at one of the parameterisations meeting'
                  f' {n_max_criteria_met} criteria:')
            print(f'\t\t{params_dict}')
            print(f'\t\t{crit_dict}')
            prob_fit.set_params(params_dict)
            for scenario in prob_fit.scenarios.values():
                print(f'\n\n\t\t\tScenario "{scenario.name}"')
                sc_simulation = prob_fit.simulate_scenario(scenario, 
                                                           apply_stop_criteria=False,
                                                           zero_acc_after_exit=False)
                be_plots = 'oBE' in prob_fit.name
                sc_simulation.do_plots(kinem_states=True, beh_probs=be_plots)
                sc_fitting.get_metrics_for_scenario(scenario, sc_simulation, verbose=True)
        if n_max_criteria_met >= N_CRIT_FOR_PARAMS_PLOT and DO_PLOTS and DO_PARAMS_PLOTS:
            sc_fitting.do_crit_params_plot(prob_fit, criteria_matrix, log=True)
            
    # end for loop through prob_fit_files
    
    
    if DO_PLOTS and DO_CRIT_PLOT:
        plt.show()
        if SAVE_FIGS:
            file_name = sc_plot.FIGS_FOLDER + f'figS{CRIT_PLOT_FIG_NO}.pdf'
            print(f'Saving {file_name}...')
            plt.savefig(file_name, bbox_inches='tight')  
            
        
    # provide info on retained models
    print('\n\n*** Retained models ***')
    for ret_model in retained_models:
        n_ret_params = ret_model.params_array.shape[0]
        n_total = prob_fits[ret_model.model].n_parameterisations
        print(f'\nModel {ret_model.model}\nRetaining {n_ret_params}'
              f' out of {n_total}'
              f' ({100 * n_ret_params / n_total:.1f} %) parameterisations, across:')
        print(ret_model.param_names)
        if DO_PLOTS and DO_RETAINED_PARAMS_PLOT:
            sc_fitting.do_params_plot(ret_model.param_names, 
                                      ret_model.params_array, 
                                      ret_model.param_ranges, 
                                      log=True, jitter=PARAMS_JITTER,
                                      model_name=ret_model.model)
            if SAVE_FIGS and ret_model.model in RET_PARAMS_PLOTS_TO_SAVE:
                fig_number = (RET_PARAMS_PLOT_FIRST_FIG_NO 
                              + RET_PARAMS_PLOTS_TO_SAVE.index(ret_model.model))
                file_name = sc_plot.FIGS_FOLDER + f'figS{fig_number}.png'
                print(f'Saving {file_name}...')
                plt.savefig(file_name, bbox_inches='tight', dpi=sc_plot.DPI) 
        print('\n***********************')
        
    
    # save the retained models
    sc_fitting.save_results(retained_models, retained_fits_file_name)        
        
        
    # provide interaction outcome plots for the retained models?
    if DO_PLOTS and DO_OUTCOME_PLOT:
        fig_height = 2 + len(retained_models)
        fig_width = 4 * N_INTERACTIVE_SCENARIOS
        outc_fig, outc_axs = plt.subplots(nrows=len(retained_models), 
                                          ncols=2*N_INTERACTIVE_SCENARIOS,
                                          sharex='col', sharey=False, 
                                          figsize=(fig_width, fig_height), 
                                          tight_layout=True)
        for i_ret_model, ret_model in enumerate(retained_models):
            prob_fit = prob_fits[ret_model.model]
            for i_scenario in range(N_INTERACTIVE_SCENARIOS):
                
                idx_retained = np.nonzero(prob_fit.n_criteria_met == len(CRITERIA))[0]
                assert(len(idx_retained > 0))
                
                # first-exiter
                # - array for storing results
                n_total = len(idx_retained) * prob_fit.n_repetitions
                first_exiter = np.full(n_total, np.nan)
                # - get model exit times for both agents, for the retained parameterisations
                scenario = list(prob_fit.scenarios.values())[i_scenario]
                ped_metric_name = scenario.get_full_metric_name('ped_exit_time')
                ped_exit_t = prob_fit.get_metric_results(ped_metric_name)[idx_retained, :]
                veh_metric_name = scenario.get_full_metric_name('veh_exit_time')
                veh_exit_t = prob_fit.get_metric_results(veh_metric_name)[idx_retained, :]
                # - rearrange exit times into 1D arrays
                ped_exit_t = np.reshape(ped_exit_t, -1)
                veh_exit_t = np.reshape(veh_exit_t, -1)
                # - get cases where agents exited conflict space within simulation
                bidx_ped_exited = np.logical_not(np.isnan(ped_exit_t))
                bidx_veh_exited = np.logical_not(np.isnan(veh_exit_t))
                # - both agents exited
                bidx_both_exited = bidx_ped_exited & bidx_veh_exited
                bidx_both_exited_ped_first = bidx_both_exited & (ped_exit_t < veh_exit_t)
                first_exiter[bidx_both_exited_ped_first] = sc_fitting.i_PED_AGENT
                bidx_both_exited_veh_first = bidx_both_exited & (ped_exit_t > veh_exit_t)
                first_exiter[bidx_both_exited_veh_first] = sc_fitting.i_VEH_AGENT
                # - only ped exited
                bidx_only_ped_exited = bidx_ped_exited & np.logical_not(bidx_veh_exited)
                first_exiter[bidx_only_ped_exited] = sc_fitting.i_PED_AGENT
                # - only ped exited
                bidx_only_veh_exited = np.logical_not(bidx_ped_exited) & bidx_veh_exited
                first_exiter[bidx_only_veh_exited] = sc_fitting.i_VEH_AGENT
                # - totals
                n_ped_first = np.count_nonzero(first_exiter == sc_fitting.i_PED_AGENT)
                n_veh_first = np.count_nonzero(first_exiter == sc_fitting.i_VEH_AGENT)
                n_noone_first = np.count_nonzero(np.isnan(first_exiter))
                # - plot
                ax = outc_axs[i_ret_model, i_scenario * 2]
                ax.bar(0, n_ped_first)
                ax.bar(0, n_noone_first, bottom=n_ped_first)
                ax.bar(0, n_veh_first, bottom=n_ped_first+n_noone_first)
                if i_ret_model == 0:
                    ax.set_title(scenario.name)
                if i_scenario == 0:
                    ax.set_ylabel(prob_fit.name, fontsize=8, 
                                  rotation=ylabel_rotation)
                
                # interaction duration
                # durations = np.maximum(ped_exit_t[bidx_both_exited],
                #                        veh_exit_t[bidx_both_exited])
                ax = outc_axs[i_ret_model, i_scenario * 2 + 1]
                ped_exit_t[np.isnan(ped_exit_t)] = 15
                veh_exit_t[np.isnan(veh_exit_t)] = 15
                ax.hist(ped_exit_t, bins = np.arange(16), color='b', alpha=0.5)
                ax.hist(veh_exit_t, bins = np.arange(16), color='g', alpha=0.5)
                
        
        
    # the dict with all of the results may be useful
    return prob_fits
    

if __name__ == '__main__':
    # run the analysis on the "pure" probabilistic fits
    
    do(sc_fitting.PROB_FIT_FILE_NAME_FMT, sc_fitting.RETAINED_PROB_FNAME)