# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 07:54:11 2021

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
import sc_fitting


INCL_DET_MODELS = ('oVAoBEvoAI',)
EXCL_PROB_MODELS = ('oVAoEAoAN',)


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
        model_str, sc_fitting.PROB_FIT_SCENARIOS, assumptions, 
        sc_fitting.DEFAULT_PARAMS, default_params_k, 
        param_arrays, list_search=True,
        n_repetitions=sc_fitting.N_PROB_SCEN_REPETITIONS, parallel=True,
        n_workers=n_workers, verbosity=verbosity, 
        file_name_format=sc_fitting.COMB_FIT_FILE_NAME_FMT)


if __name__ == '__main__':
    
    # load the retained models
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/'
              + sc_fitting.RETAINED_DET_FNAME, 'rb') as file_obj:
        det_models = pickle.load(file_obj)
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/' 
              + sc_fitting.RETAINED_PROB_FNAME, 'rb') as file_obj:
        prob_models = pickle.load(file_obj)
    
    # loop through and fit all combinations of retained deterministic and 
    # probabilistic models
    for det_model in det_models:
        if not det_model.model in INCL_DET_MODELS:
            continue
        for prob_model in prob_models:
            if prob_model.model in EXCL_PROB_MODELS:
                continue
            
            # get combined model name
            assert(prob_model.model[0:3] == 'oVA')
            model_name = det_model.model + prob_model.model[3:]
            
            # get the combined list of parameterisations
            # - first construct a big matrix with deterministic parameters to 
            # - the left, repeating each deterministic row of parameters once
            # - for each probabilistic set of parameters
            n_det_params = len(det_model.param_names)
            n_prob_params = len(prob_model.param_names)
            n_det_parameterisations = det_model.params_array.shape[0]
            n_prob_parameterisations = prob_model.params_array.shape[0]
            params_matrix = np.repeat(det_model.params_array, 
                                      n_prob_parameterisations, axis=0)
            params_matrix = np.append(params_matrix, np.tile(
                prob_model.params_array, (n_det_parameterisations, 1)), axis=1)
            # - then generate the dict needed for the SCPaperParameterSearch
            # - constructor
            param_names = det_model.param_names + prob_model.param_names
            param_arrays_dict = {}
            for i_param, param_name in enumerate(param_names):
                param_arrays_dict[param_name] = params_matrix[:, i_param]
                
            print(f'Fitting model {model_name} across {n_det_parameterisations}'
                  f' x {n_prob_parameterisations} = {params_matrix.shape[0]}'
                  ' combined parameterisations...')
            
            # run fit across these parameterisations
            run_fit(model_name, param_arrays_dict)
            