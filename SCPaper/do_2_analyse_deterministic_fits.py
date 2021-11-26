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
import collections
import parameter_search
import sc_fitting

ExampleParameterisation = collections.namedtuple(
    'ExampleParameterisation',['i_parameterisation', 'params_array', 
                               'params_dict', 'main_crit_dict', 'sec_crit_dict'])

# constants
DO_PLOTS = False
N_MAIN_CRIT_FOR_PLOT = 3
MODELS_TO_ANALYSE = 'all' # ('oVAoBEooBEvoAI',)
ASSUMPTIONS_TO_NOT_ANALYSE = 'none'
SPEEDUP_FRACT = 1.01
SURPLUS_DEC_THRESH = 1 # m/s^2
HESITATION_SPEED_FRACT = 0.8
VEH_SPEED_AT_PED_START_THRESH = 0.5 # m/s
MAIN_CRITERIA = ('veh_prio_assert', 'veh_short_stop', 
                 'ped_hesitate_veh_yield', 'ped_cross_veh_yield')
SEC_CRITERIA = ('ped_hesitate_veh_const',)
PED_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_PED_AGENT]
VEH_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_VEH_AGENT]
N_MAIN_CRIT_FOR_RETAINING = 3


# find pickle files from deterministic fitting
det_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                            (sc_fitting.DET_FIT_FILE_NAME_FMT % '*'))
det_fit_files.sort()
print(det_fit_files)


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
    
    # calculate criterion vectors
    
    print('\tMain criteria:')
    
    # - "veh_prio_assert": vehicle asserting priority
    veh_av_speed = det_fit.get_metric_results('VehPrioAssert_veh_av_speed')
    veh_prio_assert = veh_av_speed > SPEEDUP_FRACT * VEH_FREE_SPEED
    n_veh_prio_assert = np.count_nonzero(veh_prio_assert)
    print(f'\t\tVehicle asserting priority: Found {n_veh_prio_assert}'
          f' ({100 * n_veh_prio_assert / n_parameterisations:.1f} %) parameterisations.') 
    
    # - "veh_short_stop": vehicle short-stopping
    veh_av_surplus_dec = det_fit.get_metric_results(
        'VehShortStop_veh_av_surpl_dec')
    veh_short_stop = veh_av_surplus_dec > SURPLUS_DEC_THRESH
    n_veh_short_stop = np.count_nonzero(veh_short_stop)
    print(f'\t\tVehicle short-stopping: Found {n_veh_short_stop}'
          f' ({100 * n_veh_short_stop / n_parameterisations:.1f} %) parameterisations.') 
    
    # - "ped_hesitate_veh_yield": pedestrian decelerating for a yielding vehicle
    ped_av_speed = det_fit.get_metric_results('PedHesitateVehYield_ped_av_speed')
    ped_hesitate_veh_yield = ped_av_speed < HESITATION_SPEED_FRACT * PED_FREE_SPEED
    n_ped_hesitate_veh_yield = np.count_nonzero(ped_hesitate_veh_yield)
    print(f'\t\tPedestrian hesitation in deceleration scenario:'
          f' Found {n_ped_hesitate_veh_yield}'
          f' ({100 * n_ped_hesitate_veh_yield / n_parameterisations:.1f} %) parameterisations.') 
    
    # - "ped_cross_veh_yield": pedestrian beginning to cross before yielding vehicle at full stop
    veh_speed_at_ped_start = det_fit.get_metric_results(
        'PedCrossVehYield_veh_speed_at_ped_start')
    ped_cross_veh_yield = veh_speed_at_ped_start > VEH_SPEED_AT_PED_START_THRESH
    n_ped_cross_veh_yield = np.count_nonzero(ped_cross_veh_yield)
    print(f'\t\tPedestrian starting before vehicle at full stop:'
          f' Found {n_ped_cross_veh_yield}'
          f' ({100 * n_ped_cross_veh_yield / n_parameterisations:.1f} %) parameterisations.') 
    
    print('\tSecondary criteria:')
    
    # - "ped_hesitate_veh_const": pedestrian decelerating for a constant speed vehicle
    ped_av_speed = det_fit.get_metric_results('PedHesitateVehConst_ped_av_speed')
    ped_hesitate_veh_const = ped_av_speed < HESITATION_SPEED_FRACT * PED_FREE_SPEED
    n_ped_hesitate_veh_const = np.count_nonzero(ped_hesitate_veh_const)
    print(f'\t\tPedestrian hesitation in constant-speed scenario:'
          f' Found {n_ped_hesitate_veh_const}'
          f' ({100 * n_ped_hesitate_veh_const / n_parameterisations:.1f} %) parameterisations.') 
 
    
    # - look across multiple criteria
    # -- main criteria
    main_criteria_matrix = np.array((veh_prio_assert, 
                                     veh_short_stop,
                                     ped_hesitate_veh_yield, 
                                     ped_cross_veh_yield))
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
    sec_criteria_matrix = np.array((ped_hesitate_veh_const,))
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
    
    # did the model meet all main criteria at least somewhere, even if not in
    # a single parameterisation?
    main_crit_met_somewhere = np.amax(main_criteria_matrix, axis=1)
    all_main_crit_met_somewhere = np.all(main_crit_met_somewhere)
    if all_main_crit_met_somewhere:
        retained_params = (n_main_criteria_met == N_MAIN_CRIT_FOR_RETAINING)
        retained_models.append(sc_fitting.ModelWithParams(
            model=det_fit.name, param_names=copy.copy(det_fit.param_names), 
            params_array=np.copy(det_fit.results.params_matrix[retained_params])))
    
    # pick a maximally sucessful parameterisations, and provide simulation 
    # plots if requested
    i_parameterisation = np.nonzero(met_max_main_criteria 
                                    & (n_sec_criteria_met
                                       == n_max_sec_crit_met_among_best))[0][0]
    params_array = det_fit.results.params_matrix[i_parameterisation, :]
    params_dict = det_fit.get_params_dict(params_array)
    main_crit_dict = {crit : main_criteria_matrix[i_crit, i_parameterisation] 
                 for i_crit, crit in enumerate(MAIN_CRITERIA)}
    sec_crit_dict = {crit : sec_criteria_matrix[i_crit, i_parameterisation] 
                 for i_crit, crit in enumerate(SEC_CRITERIA)}
    det_fit.example = ExampleParameterisation(
        i_parameterisation=i_parameterisation, params_array=params_array,
        params_dict=params_dict, main_crit_dict=main_crit_dict, 
        sec_crit_dict=sec_crit_dict)
    if DO_PLOTS and (np.sum(main_crit_met_somewhere) >= N_MAIN_CRIT_FOR_PLOT):
        print('\tLooking at one of the parameterisations meeting'
              f' {n_main_criteria_met[i_parameterisation]} criteria:')
        print(f'\t\t{params_dict}')
        print(f'\t\t{main_crit_dict}')
        print(f'\t\t{sec_crit_dict}')
        det_fit.set_params(params_dict)
        for scenario in det_fit.scenarios.values():
            print(f'\n\n\t\t\tScenario "{scenario.name}"')
            sc_simulation = det_fit.simulate_scenario(scenario)
            be_plots = 'oBE' in det_fit.name
            sc_simulation.do_plots(kinem_states=True, 
                                   beh_probs=be_plots)
        
    
# provide info on retained models
print('\n\n*** Retained models ***')
for ret_model in retained_models:
    print(f'\nModel {ret_model.model}\nRetaining {ret_model.params_array.shape[0]}'
          ' parameterisations, across:')
    print(ret_model.param_names)
    print('\n***********************')
    

# save the retained models
with open(sc_fitting.FIT_RESULTS_FOLDER + '/RetainedDetModels.pkl', 'wb') as file_obj:
    pickle.dump(retained_models, file_obj)
    
    