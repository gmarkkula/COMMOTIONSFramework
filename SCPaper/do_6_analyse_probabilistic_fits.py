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
import pickle
import copy
import numpy as np
import collections
import parameter_search
import sc_fitting


ExampleParameterisation = collections.namedtuple(
    'ExampleParameterisation',['i_parameterisation', 'params_array', 
                                'params_dict', 'crit_dict'])

# constants
DO_TIME_SERIES_PLOTS = True
N_CRIT_FOR_TS_PLOT = 4
DO_PARAMS_PLOTS = False
DO_RETAINED_PARAMS_PLOT = True
N_CRIT_FOR_PARAMS_PLOT = 4
N_CRIT_FOR_RETAINING = 4
MODELS_TO_ANALYSE = 'all' # ('oVAoBEooBEvoAI',)
ASSUMPTIONS_TO_NOT_ANALYSE = 'none'
CRITERIA = ('Collision-free encounter', 
            'Collision-free encounter with pedestrian priority', 
            'Collision-free pedestrian lead situation', 
            'Pedestrian hesitation in constant-speed scenario')
PED_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_PED_AGENT]
VEH_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_VEH_AGENT]
PARAMS_JITTER = 0.015
#N_MAIN_CRIT_FOR_RETAINING = 3


def do(prob_fit_file_name_fmt, retained_fits_file_name):

    # find pickle files from probabilistic fitting
    prob_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                                (prob_fit_file_name_fmt % '*'))
    prob_fit_files.sort()
    print(prob_fit_files)
    
    
    # loop through the fitting results files
    prob_fits = {}
    retained_models = []
    for prob_fit_file in prob_fit_files:
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
                
            elif crit == 'Pedestrian hesitation in constant-speed scenario':
                ped_av_speed = prob_fit.get_metric_results('PedHesitateVehConst_ped_av_speed_to_CS')
                crit_met_all = ((ped_av_speed < 0.95 * PED_FREE_SPEED)
                                | np.isnan(ped_av_speed))
                # criterion met for parameterisation if met for enough of the repetitions
                crit_met = np.sum(crit_met_all, axis=1) >= 4
            
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
        if n_max_criteria_met >= N_CRIT_FOR_TS_PLOT and DO_TIME_SERIES_PLOTS:
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
        if n_max_criteria_met >= N_CRIT_FOR_PARAMS_PLOT and DO_PARAMS_PLOTS:
            sc_fitting.do_crit_params_plot(prob_fit, criteria_matrix, log=True)
    
        
    # provide info on retained models
    print('\n\n*** Retained models ***')
    for ret_model in retained_models:
        n_ret_params = ret_model.params_array.shape[0]
        n_total = prob_fits[ret_model.model].n_parameterisations
        print(f'\nModel {ret_model.model}\nRetaining {n_ret_params}'
              f' out of {n_total}'
              f' ({100 * n_ret_params / n_total:.1f} %) parameterisations, across:')
        print(ret_model.param_names)
        if DO_RETAINED_PARAMS_PLOT:
            sc_fitting.do_params_plot(ret_model.param_names, ret_model.params_array, 
                                      ret_model.param_ranges, log=True, jitter=PARAMS_JITTER)
        print('\n***********************')
        
    
    # save the retained models
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/' + retained_fits_file_name, 
              'wb') as file_obj:
        pickle.dump(retained_models, file_obj)
    

if __name__ == '__main__':
    # run the analysis on the "pure" probabilistic fits
    do(sc_fitting.PROB_FIT_FILE_NAME_FMT, sc_fitting.RETAINED_PROB_FNAME)