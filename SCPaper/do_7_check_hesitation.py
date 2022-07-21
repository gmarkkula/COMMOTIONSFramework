# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:58:10 2021

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
import pickle
import numpy as np
import itertools
import multiprocessing as mp
from dataclasses import dataclass
import sc_fitting

SCENARIO_TTCAS = (-3, 8)
N_SIMS = 50


def run_one_sim(model_name, params_dict, car_ttca):
    scenario = sc_fitting.SCPaperScenario('PedHesitateVehConst', 
                                        initial_ttcas=(3, car_ttca), 
                                        veh_const_speed=True,
                                        stop_criteria = sc_fitting.IN_CS_STOP,
                                        metric_names = ('ped_av_speed_to_CS',),
                                        time_step = sc_fitting.PROB_SIM_TIME_STEP,
                                        end_time = sc_fitting.PROB_SIM_END_TIME) 
    sim = sc_fitting.construct_model_and_simulate_scenario(
        model_name, params_dict, scenario)
    metrics = sc_fitting.get_metrics_for_scenario(scenario, sim, verbose=False)
    return metrics['PedHesitateVehConst_ped_av_speed_to_CS']

@dataclass
class HesitationCheck:
    params_dict: dict
    mean_vs: list
    ci_radii: list
    passed: bool


if __name__ == '__main__':
    
    print('Starting pool of workers...')
    pool = mp.Pool(mp.cpu_count()-1)

    # load the retained models
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/' + sc_fitting.RETAINED_PROB_FNAME,
              'rb') as file_obj:
        retained_models = pickle.load(file_obj)
        
    # loop through the retained models
    hesitation_check_results = {}
    for retained in retained_models:
        print(f'Model {retained.model}; {retained.param_names}:')
        
        # loop through the retained parameterisations and compare the before-
        # conflict-space speeds between the PedHesitateVehConst scenario and
        # a scenario where the car has already passed the pedestrian (TTCA = - 3 s)
        n_params = retained.params_array.shape[0]
        retained.hesitation_verified = np.full(n_params, False)
        hesitation_check_results[retained.model] = []
        for i_param in range(n_params):
            params_dict = dict(zip(retained.param_names, retained.params_array[i_param, :]))
            print(f'\t{params_dict}...')
            
            # loop through the two scenarios, and get confidence intervals for
            # the before-conflict-space average speeds in each scenario
            mean_vs = []
            ci_radii = []
            for i_scen, car_ttca in enumerate(SCENARIO_TTCAS):
                sim_iter = ((retained.model, params_dict, car_ttca) 
                            for i in range(N_SIMS))
                vs_list = list(pool.starmap(run_one_sim, sim_iter))   
                vs = np.array(vs_list)
                vs = vs[~np.isnan(vs)]
                if len(vs) < N_SIMS:
                    print(f'\t\tRemoved {N_SIMS - len(vs)} simulations where ped. did not enter conflict space.')
                mean_v = np.mean(vs)
                ci_radius = 1.96 * np.std(vs)/math.sqrt(N_SIMS)
                mean_vs.append(mean_v)
                ci_radii.append(ci_radius)
                print(f'\t\tTTCA = {car_ttca} s: Av: {mean_v:.3f} m/s;'
                      f' CI: [{mean_v - ci_radius:.3f}, {mean_v + ci_radius:.3f}] m/s')
            
            # separation between confidence intervals?
            baseline_lower = mean_vs[0] - ci_radii[0]
            test_upper = mean_vs[1] + ci_radii[1]
            passed = test_upper < baseline_lower
            if passed:
                print('\t\t\tPassed.')
                retained.hesitation_verified[i_param] = True
                
            # save hesitation check results
            hesitation_check_results[retained.model].append(
                HesitationCheck(params_dict, mean_vs, ci_radii, passed))  
            
                
    # print overall summary
    print('\n********** Summary **********\n')
    for retained in retained_models:
        print(f'Model {retained.model}; {retained.param_names}:')
        n_params = retained.params_array.shape[0]
        n_verified = np.sum(retained.hesitation_verified)
        print(f'\tPedestrian hesitation verified in {n_verified} out of'
              f' {n_params} parameterisations ({100 * n_verified/n_params:.1f} %).\n')


    # save the results
    sc_fitting.save_results(hesitation_check_results, 
                            sc_fitting.HESITATION_CHECK_FNAME)
    
            