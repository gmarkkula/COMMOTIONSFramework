
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 19:37:11 2021

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
import pickle
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
                               'params_dict', 'main_crit_dict', 'sec_crit_dict'])

# constants
DO_TIME_SERIES_PLOTS = False
DO_PARAMS_PLOTS = False
DO_RETAINED_PARAMS_PLOT = False
DO_CRIT_PLOT = False
N_MAIN_CRIT_FOR_PLOT = 4
MODELS_TO_ANALYSE = 'all' # ('oVAoBEooBEvoAI',)
ASSUMPTIONS_TO_NOT_ANALYSE = 'none'
SPEEDUP_FRACT = 1.01
SURPLUS_DEC_THRESH = 0.5 # m/s^2
HESITATION_SPEED_FRACT = 0.95
VEH_SPEED_AT_PED_START_THRESH = 0.5 # m/s
CRITERION_GROUPS = ('Main criteria', 'Secondary criteria')
i_MAIN = 0
i_SEC = 1
N_CRIT_GROUPS = len(CRITERION_GROUPS)
CRITERIA = (('Vehicle asserting priority', 'Vehicle short-stopping', 
             'Pedestrian hesitation in deceleration scenario', 
             'Pedestrian starting before vehicle at full stop'),
            ('Pedestrian hesitation in constant-speed scenario',)
            )
assert(N_CRIT_GROUPS == len(CRITERIA))
CRIT_METRICS = (('$\overline{v}_\mathrm{v}/v_\mathrm{v,free}$ (-)', 
                 '$\overline{d}/d_\mathrm{stop}$ (-)',
                 '$\overline{v}_\mathrm{p}/v_\mathrm{p,free}$ (-)', 
                 '$v_\mathrm{v}(t_\mathrm{cross})$ (m/s)'), 
                ('$\overline{v}_\mathrm{p}/v_\mathrm{p,free}$ (-)',))
N_CRITERIA = 0
for grp_criteria in CRITERIA:
    N_CRITERIA += len(grp_criteria)


PED_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_PED_AGENT]
VEH_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_VEH_AGENT]
N_MAIN_CRIT_FOR_RETAINING = 3
PARAMS_JITTER = 0.015


def do():
    # find pickle files from deterministic fitting
    det_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                                (sc_fitting.DET_FIT_FILE_NAME_FMT % '*'))
    det_fit_files.sort()
    print(det_fit_files)
    
    # need to prepare for criterion plot?
    if DO_CRIT_PLOT:
        crit_fig, crit_axs = plt.subplots(nrows=sc_plot.N_BASE_MODELS, ncols=N_CRITERIA,
                                          sharex='col', sharey=True, 
                                          figsize=(15, 10), tight_layout=True)
    
    # loop through the deterministic fitting results files
    det_fits = {}
    retained_models = []
    for det_fit_file in det_fit_files:
        print()
        det_fit = parameter_search.load(det_fit_file, verbose=True)
        if ((not(MODELS_TO_ANALYSE == 'all') and not (det_fit.name in MODELS_TO_ANALYSE))
            or ASSUMPTIONS_TO_NOT_ANALYSE in det_fit.name):
            print(f'Skipping model {det_fit.name}.')
            continue
        det_fits[det_fit.name] = det_fit
        n_parameterisations = det_fit.results.metrics_matrix.shape[0]
        print(f'Analysing model {det_fit.name},'
              f' {n_parameterisations} parameterisations...')
        
        # split model name into base and variant
        i_model_base, i_model_variant = sc_plot.split_model_name(det_fit.name)
        
        # calculate criterion vectors
        criteria_matrices = []
        i_crit_glob = -1
        for i_crit_grp in range(N_CRIT_GROUPS):
            print(f'\t{CRITERION_GROUPS[i_crit_grp]}:')
            criteria_matrices.append(
                np.full((len(CRITERIA[i_crit_grp]), n_parameterisations), False))
            for i_crit, crit in enumerate(CRITERIA[i_crit_grp]):
                i_crit_glob +=1
                
                # criterion-specific calculations
                if crit == 'Vehicle asserting priority':
                    veh_av_speed = det_fit.get_metric_results('VehPrioAssert_veh_av_speed')
                    crit_metric = veh_av_speed / VEH_FREE_SPEED
                    crit_thresh = SPEEDUP_FRACT
                    crit_greater_than = True
                    # crit_met_all = veh_av_speed > SPEEDUP_FRACT * VEH_FREE_SPEED
                    
                elif crit == 'Vehicle short-stopping':
                    veh_av_surplus_dec = det_fit.get_metric_results(
                        'VehShortStop_veh_av_surpl_dec')
                    crit_metric = veh_av_surplus_dec
                    crit_thresh = SURPLUS_DEC_THRESH
                    crit_greater_than = True
                    # crit_met_all = veh_av_surplus_dec > SURPLUS_DEC_THRESH
                    
                elif crit == 'Pedestrian hesitation in constant-speed scenario':
                    ped_av_speed = det_fit.get_metric_results('PedHesitateVehConst_ped_av_speed')
                    crit_metric = ped_av_speed / PED_FREE_SPEED
                    crit_thresh = HESITATION_SPEED_FRACT
                    crit_greater_than = False
                    # crit_met_all = ped_av_speed < HESITATION_SPEED_FRACT * PED_FREE_SPEED
                
                elif crit == 'Pedestrian hesitation in deceleration scenario':
                    ped_av_speed = det_fit.get_metric_results('PedHesitateVehYield_ped_av_speed')
                    crit_metric = ped_av_speed / PED_FREE_SPEED
                    crit_thresh = HESITATION_SPEED_FRACT
                    crit_greater_than = False
                    # crit_met_all = ped_av_speed < HESITATION_SPEED_FRACT * PED_FREE_SPEED
                    
                elif crit == 'Pedestrian starting before vehicle at full stop':
                    veh_speed_at_ped_start = det_fit.get_metric_results(
                        'PedCrossVehYield_veh_speed_at_ped_start')
                    crit_metric = veh_speed_at_ped_start
                    crit_thresh = VEH_SPEED_AT_PED_START_THRESH
                    crit_greater_than = True
                    # crit_met_all = veh_speed_at_ped_start > VEH_SPEED_AT_PED_START_THRESH
                
                else:
                    raise Exception(f'Unexpected criterion "{crit}".')
                    
                # apply criterion across all kinematic variants
                if crit_greater_than:
                    crit_met_all = crit_metric > crit_thresh
                else:
                    crit_met_all = crit_metric < crit_thresh
                # criterion met for model if met for at least one of the kinematic variants
                crit_met = np.any(crit_met_all, axis=1)
                criteria_matrices[i_crit_grp][i_crit, :] = crit_met
                # print some info
                n_crit_met = np.count_nonzero(crit_met)
                print(f'\t\t{crit}: Found {n_crit_met}'
                      f' ({100 * n_crit_met / n_parameterisations:.1f} %) parameterisations.') 
                
                if DO_CRIT_PLOT:
                    ax = crit_axs[i_model_base, i_crit_glob]
                    # use the kinematic variation with max/min metric value,
                    # depending on direction of criterion (and disregard NaNs)
                    if crit_greater_than:
                        ecdf = ECDF(np.nanmax(crit_metric, axis=1))
                    else:
                        ecdf = ECDF(np.nanmin(crit_metric, axis=1))
                    ax.step(ecdf.x, ecdf.y, sc_plot.MVAR_LINESPECS[i_model_variant], 
                            color=sc_plot.MVAR_COLORS[i_model_variant],
                            lw=sc_plot.MVAR_LWS[i_model_variant])
                    if i_model_variant == sc_plot.N_MODEL_VARIANTS-1:
                        if i_model_base == 0:
                            ax.set_title(crit, fontsize=8)
                            if i_crit_glob == 0:
                                ax.legend(sc_plot.MODEL_VARIANTS, fontsize=8)
                        elif i_model_base == sc_plot.N_BASE_MODELS-1:
                            ax.set_xlabel(CRIT_METRICS[i_crit_grp][i_crit])
                        if i_crit_glob == 0:
                            ax.set_ylabel(sc_plot.BASE_MODELS[i_model_base], 
                                          fontsize=8)
                        ax.axvline(crit_thresh, ls=':', color='gray')
                        
            # end i_crit, crit for loop
        # end i_crit_grp for loop
        
        # - look across multiple criteria
        main_criteria_matrix = criteria_matrices[i_MAIN]
        # -- main criteria
        all_main_criteria_met = np.all(main_criteria_matrix, axis=0)
        n_all_main_criteria_met = np.count_nonzero(all_main_criteria_met)
        print(f'\tAll main criteria met: Found {n_all_main_criteria_met}'
              f' ({100 * n_all_main_criteria_met / n_parameterisations:.1f} %)'
              ' parameterisations.')  
        n_main_criteria_met = np.sum(main_criteria_matrix, axis=0)
        n_max_main_criteria_met = np.max(n_main_criteria_met)
        met_max_main_criteria = n_main_criteria_met == n_max_main_criteria_met
        n_met_max_main_criteria = np.count_nonzero(met_max_main_criteria)
        print(f'\tMax no of main criteria met was {n_max_main_criteria_met},'
              f' for {n_met_max_main_criteria} parameterisations.')
        # -- secondary criteria
        sec_criteria_matrix = criteria_matrices[i_SEC]
        n_sec_criteria_met = np.sum(sec_criteria_matrix, axis=0)
        n_sec_criteria_met_among_best = n_sec_criteria_met[met_max_main_criteria]
        n_max_sec_crit_met_among_best = np.max(n_sec_criteria_met_among_best)
        n_met_max_sec_crit_among_best = np.count_nonzero(
            n_sec_criteria_met_among_best == n_max_sec_crit_met_among_best)
        print('\t\tOut of these, the max number of secondary criteria met was'
              f' {n_max_sec_crit_met_among_best}, for {n_met_max_sec_crit_among_best}'
              ' parameterisations.')
        # -- NaNs
        print(f'\tNaNs in main crit: {np.sum(np.isnan(main_criteria_matrix), axis=1)}'
              f'; sec crit: {np.sum(np.isnan(sec_criteria_matrix), axis=1)}')
        # -- store these analysis results as object attributes
        det_fit.main_criteria_matrix = main_criteria_matrix
        det_fit.n_main_criteria_met = n_main_criteria_met
        det_fit.sec_criteria_matrix = sec_criteria_matrix
        
        # get the parameter ranges, for possible retaining and/or plotting below
        param_ranges = []
        for i_param in range(det_fit.n_params):
            param_ranges.append((np.amin(det_fit.results.params_matrix[:, i_param]),
                                 np.amax(det_fit.results.params_matrix[:, i_param])))
        
        # did the model meet all main criteria at least somewhere, even if not in
        # a single parameterisation? 
        main_crit_met_somewhere = np.amax(main_criteria_matrix, axis=1)
        all_main_crit_met_somewhere = np.all(main_crit_met_somewhere)
        if all_main_crit_met_somewhere:
            # yes, so retain this model for further analysis
            retained_params = (n_main_criteria_met >= N_MAIN_CRIT_FOR_RETAINING)
            retained_models.append(sc_fitting.ModelWithParams(
                model=det_fit.name, param_names=copy.copy(det_fit.param_names), 
                param_ranges=param_ranges,
                params_array=np.copy(det_fit.results.params_matrix[retained_params])))
        
        # pick a maximally sucessful parameterisations, and provide simulation 
        # plots if requested
        i_parameterisation = np.nonzero(met_max_main_criteria 
                                        & (n_sec_criteria_met
                                           == n_max_sec_crit_met_among_best))[0][0]
        params_array = det_fit.results.params_matrix[i_parameterisation, :]
        params_dict = det_fit.get_params_dict(params_array)
        main_crit_dict = {crit : main_criteria_matrix[i_crit, i_parameterisation] 
                     for i_crit, crit in enumerate(CRITERIA[i_MAIN])}
        sec_crit_dict = {crit : sec_criteria_matrix[i_crit, i_parameterisation] 
                     for i_crit, crit in enumerate(CRITERIA[i_SEC])}
        det_fit.example = ExampleParameterisation(
            i_parameterisation=i_parameterisation, params_array=params_array,
            params_dict=params_dict, main_crit_dict=main_crit_dict, 
            sec_crit_dict=sec_crit_dict)
        if np.sum(main_crit_met_somewhere) >= N_MAIN_CRIT_FOR_PLOT:
            if DO_TIME_SERIES_PLOTS:
                print('\tLooking at one of the parameterisations meeting'
                      f' {n_main_criteria_met[i_parameterisation]} criteria:')
                print(f'\t\t{params_dict}')
                print(f'\t\t{main_crit_dict}')
                print(f'\t\t{sec_crit_dict}')
                det_fit.set_params(params_dict)
                for scenario in det_fit.scenarios.values():
                    print(f'\n\n\t\t\tScenario "{scenario.name}"')
                    sc_simulations = det_fit.simulate_scenario(
                        scenario, apply_stop_criteria=False)
                    be_plots = 'oBE' in det_fit.name
                    for sim in sc_simulations:
                        sim.do_plots(kinem_states=True, 
                                     veh_stop_dec=(scenario.name == 'VehShortStop'), 
                                     beh_probs=be_plots)
                        sc_fitting.get_metrics_for_scenario(scenario, sim, verbose=True)
            if DO_PARAMS_PLOTS:
                #sc_fitting.do_crit_params_plot(det_fit, main_criteria_matrix, log=True)
                print(f'\tParameterisations meeting at least {N_MAIN_CRIT_FOR_PLOT} criteria:')
                sc_fitting.do_params_plot(
                    det_fit.param_names, det_fit.results.params_matrix[
                        n_main_criteria_met >= N_MAIN_CRIT_FOR_PLOT], 
                    param_ranges, log=True, jitter=PARAMS_JITTER)
                                
                            
                
            
        
    # provide info on retained models
    print('\n\n*** Retained models (meeting all main criteria at least somewhere in parameter space) ***')
    for ret_model in retained_models:
        n_ret_params = ret_model.params_array.shape[0]
        n_total = det_fits[ret_model.model].n_parameterisations
        print(f'\nModel {ret_model.model}\nRetaining {n_ret_params}'
              f' out of {n_total}'
              f' ({100 * n_ret_params / n_total:.1f} %) parameterisations meeting'
              f' at least {N_MAIN_CRIT_FOR_RETAINING} main criteria, across:')
        print(ret_model.param_names)
        if DO_RETAINED_PARAMS_PLOT:
            sc_fitting.do_params_plot(ret_model.param_names, ret_model.params_array, 
                                      ret_model.param_ranges, log=True, jitter=PARAMS_JITTER)
        print('\n***********************')
        
    
    # save the retained models
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/' + sc_fitting.RETAINED_DET_FNAME,
              'wb') as file_obj:
        pickle.dump(retained_models, file_obj)
        
    
    # return the full dict of analysed deterministic models
    return det_fits
        
    
if __name__ == '__main__':
    det_fits = do()