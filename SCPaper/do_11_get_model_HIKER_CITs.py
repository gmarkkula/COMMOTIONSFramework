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
import numpy as np
import sc_scenario
import parameter_search
import sc_fitting

REQ_ASSUMPTION = 'oDA' # a string to require in model name, or None

parameter_search.STATUS_REP_HEADER_LEN = 50 # long model names here...

N_SCENARIO_REPS = 6
MAX_PARAMETERISATIONS = 500


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
    
    # run a dummy probablistic simulation to prevent problems with 
    # parallelisation on ARC4 (see 2022-01-19 diary notes)
    sc_fitting.run_dummy_prob_sim()
    
    # load the retained models
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/'
              + sc_fitting.RETAINED_COMB_FNAME, 'rb') as file_obj:
        comb_models = pickle.load(file_obj)
        
    # initialise random number generator
    rng = np.random.default_rng()

    n_total = 0
    for comb_model in comb_models:
        
        if ((not(REQ_ASSUMPTION is None)) 
            and (not(REQ_ASSUMPTION in comb_model.model))):
            continue
        
        # - subsample the matrix of parameterisations if needed
        n_parameterisations = comb_model.params_array.shape[0]
        if n_parameterisations > MAX_PARAMETERISATIONS:
            idx_included = rng.choice(n_parameterisations, 
                                      size=MAX_PARAMETERISATIONS, 
                                      replace=False)
            params_matrix = comb_model.params_array[idx_included, :]
        else:
            params_matrix = comb_model.params_array
            
        n_total += params_matrix.shape[0]
        print(f'Model {comb_model.model}: Testing'
              f' {params_matrix.shape[0]} parameterisations...')
        
        # build a dict of the parameter values to test
        param_arrays_dict = {}
        for i_param, param_name in enumerate(comb_model.param_names):
            param_arrays_dict[param_name] = params_matrix[:, i_param]
        
        # run fit across these parameterisations
        run_fit(comb_model.model, param_arrays_dict)
        
    print(f'Total parameterisations tested: {n_total}.')
    