# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 07:47:47 2021

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
import pickle
import multiprocessing as mp
#import numpy as np
import sc_scenario
import sc_fitting

N_SCENARIO_REPS = 6


def run_fit(model_str, param_arrays):
    default_params_k = sc_fitting.get_default_params_k(model_str)
    assumptions = sc_scenario.get_assumptions_dict_from_string(model_str)
    if os.name == 'nt':
        # running on my own computer
        n_workers = mp.cpu_count()-1
        verbosity = 2
    else:
        # running on cluster, so use all cores, and limit status output
        n_workers = mp.cpu_count()
        verbosity = 1
    this_fit = sc_fitting.SCPaperParameterSearch(
        model_str, sc_fitting.HIKER_SCENARIOS, assumptions, 
        sc_fitting.DEFAULT_PARAMS, default_params_k, 
        param_arrays, list_search=True,
        n_repetitions=N_SCENARIO_REPS, parallel=True,
        n_workers=n_workers, verbosity=verbosity, 
        file_name_format=sc_fitting.HIKER_FIT_FILE_NAME_FMT)
    

if __name__ == '__main__':
    
    # load the retained models
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/'
              + sc_fitting.RETAINED_COMB_FNAME, 'rb') as file_obj:
        comb_models = pickle.load(file_obj)

    for comb_model in comb_models:
        
        # build a dict of the parameter values to test
        param_arrays_dict = {}
        for i_param, param_name in enumerate(comb_model.param_names):
            param_arrays_dict[param_name] = comb_model.params_array[:, i_param]
        
        # run fit across these parameterisations
        run_fit(comb_model.model, param_arrays_dict)
        
