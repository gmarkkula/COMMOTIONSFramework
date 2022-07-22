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
import parameter_search
import sc_fitting

parameter_search.STATUS_REP_HEADER_LEN = 50 # long model names here...


INCL_DET_MODELS = 'all' # ('oVAaoVAloBEvoAI', 'oVAoVAloBEvoAI') # either 'all' or a tuple of names of models to include
EXCL_PROB_MODELS = ('oVAoAN', 'oVAoEAoAN', 'oVAoDAoAN')
REQ_PROB_ASSUMPTION = 'oDA' # a string to require, or None
MAX_PARAMETERISATIONS = 5000


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
    
    # run a dummy probablistic simulation to prevent problems with 
    # parallelisation on ARC4 (see 2022-01-19 diary notes)
    sc_fitting.run_dummy_prob_sim()
    
    # load the retained models
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/'
              + sc_fitting.RETAINED_DET_FNAME, 'rb') as file_obj:
        det_models = pickle.load(file_obj)
    with open(sc_fitting.FIT_RESULTS_FOLDER + '/' 
              + sc_fitting.RETAINED_PROB_FNAME, 'rb') as file_obj:
        prob_models = pickle.load(file_obj)
        
    # initialise random number generator
    rng = np.random.default_rng()
    
    # check for any specification of determinsistic model to include from
    # command line
    if len(sys.argv) > 1:
        INCL_DET_MODELS = (sys.argv[1],)
        print(f'Found command line option "{sys.argv[1]}"')
    print(f'Included deterministic models are: {INCL_DET_MODELS}')
    
    # loop through and fit all combinations of retained deterministic and 
    # probabilistic models
    for det_model in det_models:
        
        # this deterministic model included?
        if INCL_DET_MODELS != 'all':
            assert(type(INCL_DET_MODELS) is tuple)
            if not det_model.model in INCL_DET_MODELS:
                continue
        for prob_model in prob_models:
            
            # this probabilistic model excluded?
            if prob_model.model in EXCL_PROB_MODELS:
                continue
            if ((not(REQ_PROB_ASSUMPTION is None)) 
                and (not(REQ_PROB_ASSUMPTION in prob_model.model))):
                continue
            
            # get combined model name
            assert(prob_model.model[0:3] == 'oVA')
            model_name = det_model.model + prob_model.model[3:]
            
            # get the combined list of parameterisations
            # - get number of parameters and retained parameterisations
            n_det_params = len(det_model.param_names)
            n_prob_params = len(prob_model.param_names)
            n_det_parameterisations = det_model.params_array.shape[0]
            n_prob_parameterisations = prob_model.params_array.shape[0]
            n_comb_parameterisations = (n_det_parameterisations 
                                        * n_prob_parameterisations)
            # - first construct a big matrix with deterministic parameters to 
            # - the left, repeating each deterministic row of parameters once
            # - for each probabilistic set of parameters
            params_matrix = np.repeat(det_model.params_array, 
                                      n_prob_parameterisations, axis=0)
            params_matrix = np.append(params_matrix, np.tile(
                prob_model.params_array, (n_det_parameterisations, 1)), axis=1)
            assert(params_matrix.shape[0] == n_comb_parameterisations)
            # - subsample the matrix of parameterisations if needed
            if n_comb_parameterisations > MAX_PARAMETERISATIONS:
                idx_included = rng.choice(n_comb_parameterisations, 
                                          size=MAX_PARAMETERISATIONS, 
                                          replace=False)
                params_matrix = params_matrix[idx_included, :]
            # - then generate the dict needed for the SCPaperParameterSearch
            # - constructor
            param_names = det_model.param_names + prob_model.param_names
            param_arrays_dict = {}
            for i_param, param_name in enumerate(param_names):
                param_arrays_dict[param_name] = params_matrix[:, i_param]
                
            print(f'Fitting model {model_name} across {n_det_parameterisations}'
                  f' x {n_prob_parameterisations} --> {params_matrix.shape[0]}'
                  ' combined parameterisations...')
            
            # run fit across these parameterisations
            run_fit(model_name, param_arrays_dict)
            