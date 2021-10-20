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
import numpy as np
import parameter_search
import sc_fitting

# constants
DO_PLOTS = True
MODELS_TO_ANALYSE = 'all' # ('oVAoBEo',)
ASSUMPTIONS_TO_NOT_ANALYSE = 'none'
SPEEDUP_FRACT = 1.01
SURPLUS_DEC_THRESH = 2 # m/s^2
HESITATION_SPEED_FRACT = 0.8
VEH_SPEED_AT_PED_START_THRESH = 0.1 # m/s
MAIN_CRITERIA = ('veh_assert_prio', 'veh_short_stop', 
                 'ped_hesitate_dec', 'ped_start_bef_veh_stop')
SEC_CRITERIA = ('ped_hesitate_const', 'ped_fast_crossing')
PED_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_PED_AGENT]
VEH_FREE_SPEED = sc_fitting.AGENT_FREE_SPEEDS[sc_fitting.i_VEH_AGENT]


# find pickle files from deterministic fitting
det_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                            (sc_fitting.DET_FIT_FILE_NAME_FMT % '*'))
det_fit_files.sort()
print(det_fit_files)


# loop through the deterministic fitting results files
det_fits = {}
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
    
    # - "ActVehStatPed": vehicle asserting priority
    veh_1st = det_fit.get_metric_results('ActVehStatPed_veh_1st') == 1
    veh_min_speed_before = det_fit.get_metric_results(
        'ActVehStatPed_veh_min_speed_before')
    veh_never_slowing = veh_min_speed_before >= VEH_FREE_SPEED
    veh_mean_speed_early_before = det_fit.get_metric_results(
        'ActVehStatPed_veh_mean_speed_early_before')
    veh_on_av_faster = veh_mean_speed_early_before > SPEEDUP_FRACT * VEH_FREE_SPEED
    veh_assert_prio = veh_1st & veh_never_slowing & veh_on_av_faster
    n_veh_assert_prio = np.count_nonzero(veh_assert_prio)
    print(f'\t\tVehicle asserting priority: Found {n_veh_assert_prio}'
          f' ({100 * n_veh_assert_prio / n_parameterisations:.1f} %) parameterisations.') 
    
    # - "ActVehPrioPed": vehicle short-stopping
    veh_max_surplus_dec_before = det_fit.get_metric_results(
        'ActVehStatPedPrio_veh_max_surplus_dec_before')
    veh_short_stop = veh_max_surplus_dec_before > SURPLUS_DEC_THRESH
    n_veh_short_stop = np.count_nonzero(veh_short_stop)
    print(f'\t\tVehicle short-stopping: Found {n_veh_short_stop}'
          f' ({100 * n_veh_short_stop / n_parameterisations:.1f} %) parameterisations.') 
    
    # - "ActPedPrioEncounter": pedestrian decelerating, then crossing before 
    # -                        vehicle has come to a full stop 
    ped_min_speed_before = det_fit.get_metric_results(
        'ActPedPrioEncounter_ped_min_speed_before')
    ped_hesitate_dec = ped_min_speed_before < HESITATION_SPEED_FRACT * PED_FREE_SPEED
    n_ped_hesitate_dec = np.count_nonzero(ped_hesitate_dec)
    print(f'\t\tPedestrian hesitation in deceleration scenario:'
          f' Found {n_ped_hesitate_dec}'
          f' ({100 * n_ped_hesitate_dec / n_parameterisations:.1f} %) parameterisations.') 
    veh_speed_at_ped_start = det_fit.get_metric_results(
        'ActPedPrioEncounter_veh_speed_at_ped_start')
    ped_start_bef_veh_stop = veh_speed_at_ped_start > VEH_SPEED_AT_PED_START_THRESH
    n_ped_start_bef_veh_stop = np.count_nonzero(ped_start_bef_veh_stop)
    print(f'\t\tPedestrian starting before vehicle at full stop:'
          f' Found {n_ped_start_bef_veh_stop}'
          f' ({100 * n_ped_start_bef_veh_stop / n_parameterisations:.1f} %) parameterisations.') 
    
    print('\tSecondary criteria:')
    
    # - "ActPedLeading": pedestrian decelerating, then crossing before vehicle 
    # -                  at higher than free speed
    ped_1st = det_fit.get_metric_results('ActPedLeading_ped_1st') == 1
    ped_min_speed_before = det_fit.get_metric_results(
        'ActPedLeading_ped_min_speed_before')
    ped_hesitate_const = (ped_1st & (ped_min_speed_before 
                                     < HESITATION_SPEED_FRACT * PED_FREE_SPEED))
    n_ped_hesitate_const = np.count_nonzero(ped_hesitate_const)
    print(f'\t\tPedestrian hesitation in constant-speed scenario:'
          f' Found {n_ped_hesitate_const}'
          f' ({100 * n_ped_hesitate_const / n_parameterisations:.1f} %) parameterisations.') 
    ped_max_speed_after = det_fit.get_metric_results(
        'ActPedLeading_ped_max_speed_after')
    ped_fast_crossing = ped_1st & (ped_max_speed_after > PED_FREE_SPEED)
    n_ped_fast_crossing = np.count_nonzero(ped_fast_crossing)
    print(f'\t\tPedestrian crossing fast in front of constant-speed vehicle:'
          f' Found {n_ped_fast_crossing}'
          f' ({100 * n_ped_fast_crossing / n_parameterisations:.1f} %) parameterisations.') 
    
    # - look across multiple criteria
    # -- main criteria
    main_criteria_matrix = np.array((veh_assert_prio, veh_short_stop, 
                                    ped_hesitate_dec, ped_start_bef_veh_stop))
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
    sec_criteria_matrix = np.array((ped_hesitate_const, ped_fast_crossing))
    n_sec_criteria_met = np.sum(sec_criteria_matrix, axis=0)
    n_sec_criteria_met_among_best = n_sec_criteria_met[met_max_main_criteria]
    n_max_sec_crit_met_among_best = np.max(n_sec_criteria_met_among_best)
    n_met_max_sec_crit_among_best = np.count_nonzero(
        n_sec_criteria_met_among_best == n_max_sec_crit_met_among_best)
    print('\t\tOut of these, the max number of secondary criteria met was'
          f' {n_max_sec_crit_met_among_best}, for {n_met_max_sec_crit_among_best}'
          ' parameterisations.')
    # -- store these analysis results as object attributes
    det_fit.main_criteria_matrix = main_criteria_matrix
    det_fit.n_main_criteria_met = n_main_criteria_met
    det_fit.sec_criteria_matrix = sec_criteria_matrix
    
    # pick a maximally sucessful parameterisations, and provide simulation plots
    if DO_PLOTS:
        i_parameterisation = np.nonzero(met_max_main_criteria)[0][0]
        params_array = det_fit.results.params_matrix[i_parameterisation, :]
        params_dict = det_fit.get_params_dict(params_array)
        main_crit_dict = {crit : main_criteria_matrix[i_crit, i_parameterisation] 
                     for i_crit, crit in enumerate(MAIN_CRITERIA)}
        sec_crit_dict = {crit : sec_criteria_matrix[i_crit, i_parameterisation] 
                     for i_crit, crit in enumerate(SEC_CRITERIA)}
        print('\tLooking at one of the parameterisations meeting'
              f' {n_max_main_criteria_met} criteria:')
        print(f'\t\t{params_dict}')
        print(f'\t\t{main_crit_dict}')
        print(f'\t\t{sec_crit_dict}')
        det_fit.set_params(params_array)
        for scenario in sc_fitting.DET1S_SCENARIOS.values():
            print(f'\n\n\t\t\tScenario "{scenario.name}"')
            sc_simulation = det_fit.simulate_scenario(scenario)
            be_plots = 'oBE' in det_fit.name
            sc_simulation.do_plots(kinem_states=True, 
                                   beh_probs=be_plots)
        
    
    
    