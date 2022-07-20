# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 08:45:31 2021

@author: tragma
"""
import os 
import sys
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR, __ = os.path.split(THIS_FILE_DIR)
if not PARENT_DIR in sys.path:
    sys.path.append(PARENT_DIR)

# other imports
import math
import numpy as np
import sc_scenario
import sc_fitting
import multiprocessing as mp


# set constants

# - models 
BASE_MODELS = ('oVA', 'oVAoEA', 'oVAoDA')  
MODEL_VARIANTS = ('oAN', 'oSNc', 'oSNv', 'oSNcoPF', 'oSNvoPF')


# - free parameter values
N_VALS_PER_PARAM = 10
PARAM_ARRAYS = {}
PARAM_ARRAYS['T'] = np.logspace(np.log10(0.1), np.log10(0.5), N_VALS_PER_PARAM)
PARAM_ARRAYS['DeltaV_th_rel'] = np.logspace(np.log10(0.001), np.log10(0.1), N_VALS_PER_PARAM)
PARAM_ARRAYS['sigma_V'] = np.logspace(np.log10(0.001), np.log10(1), N_VALS_PER_PARAM)
PARAM_ARRAYS['tau_theta'] = np.logspace(np.log10(0.005), np.log10(5), N_VALS_PER_PARAM)
PARAM_ARRAYS['tau_d'] = np.logspace(np.log10(0.5), np.log10(500), N_VALS_PER_PARAM)
PARAM_ARRAYS['xi_th'] = np.logspace(np.log10(5e-6), np.log10(5e-3), N_VALS_PER_PARAM)



def run_fit(model_str):
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
        sc_fitting.DEFAULT_PARAMS, default_params_k, PARAM_ARRAYS, 
        n_repetitions=sc_fitting.N_PROB_SCEN_REPETITIONS, parallel=True,
        n_workers=n_workers, verbosity=verbosity, 
        file_name_format=sc_fitting.PROB_FIT_FILE_NAME_FMT)
    

if __name__ == "__main__":
    for base_model in BASE_MODELS:
        for model_variant in MODEL_VARIANTS:
            run_fit(base_model + model_variant)
    print('Done!')