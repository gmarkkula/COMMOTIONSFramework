# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 06:14:22 2021

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
import numpy as np
import sc_scenario
import sc_fitting
import multiprocessing as mp


# set constants

# - models 
BASE_MODELS = ('', 'oVA', 'oVAa', 'oVAaoBEc', 'oVAoVAl', 'oVAaoVAl')  
MODEL_VARIANTS = ('', 'oBEo', 'oBEv', 'oBEooBEv', 'oBEvoAI', 'oBEooBEvoAI')


# - free parameter values
PARAM_ARRAYS = {}
N_GRID_SIDE = 10
PARAM_ARRAYS['k_c'] = np.logspace(np.log10(0.2), np.log10(2), N_GRID_SIDE)
PARAM_ARRAYS['k_sc'] = np.logspace(np.log10(0.02), np.log10(0.2), N_GRID_SIDE)
PARAM_ARRAYS['thetaDot_1'] = np.logspace(np.log10(0.001), np.log10(0.1), N_GRID_SIDE) 
PARAM_ARRAYS['T_delta'] = np.logspace(np.log10(10), np.log10(100), N_GRID_SIDE)
PARAM_ARRAYS['beta_V'] = np.logspace(np.log10(1), np.log10(200), N_GRID_SIDE)
PARAM_ARRAYS['T_Of'] = np.logspace(np.log10(0.5), np.log10(10), N_GRID_SIDE)
PARAM_ARRAYS['sigma_O'] = np.logspace(np.log10(0.02), np.log10(2), N_GRID_SIDE)




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
        model_str, sc_fitting.ONE_AG_SCENARIOS, assumptions, 
        sc_fitting.DEFAULT_PARAMS, default_params_k, PARAM_ARRAYS, 
        n_repetitions=sc_fitting.N_ONE_AG_SCEN_VARIATIONS, parallel=True,
        n_workers=n_workers, verbosity=verbosity)
    

if __name__ == "__main__":
    for base_model in reversed(BASE_MODELS):
        for model_variant in reversed(MODEL_VARIANTS):
            run_fit(base_model + model_variant)
    print('Done!')
    