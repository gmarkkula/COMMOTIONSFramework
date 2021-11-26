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
BASE_MODELS = ('', 'oVA', 'oVAa', 'oVAoVAl', 'oVAaoVAl', 'oVAaoBEc')  
MODEL_VARIANTS = ('', 'oBEo', 'oBEv', 'oBEooBEv', 'oBEvoAI', 'oBEooBEvoAI')


# - free parameter values
PARAM_ARRAYS = {}
# PARAM_ARRAYS['T'] = (0.2, 0.4, 0.6, 0.8)
# PARAM_ARRAYS['k_c'] = np.logspace(np.log10(0.2), np.log10(2), 4)
# PARAM_ARRAYS['k_sc'] = np.logspace(np.log10(0.02), np.log10(0.2), 4)
# PARAM_ARRAYS['thetaDot_1'] = [0.05, 0.1, 0.2, 0.4]
# PARAM_ARRAYS['T_delta'] = (15, 30, 60)
# PARAM_ARRAYS['beta_V'] = (5, 15, 45, 135)
# PARAM_ARRAYS['T_Of'] = (0.5, 1, 2, math.inf)
# PARAM_ARRAYS['sigma_O'] = (0.02, 0.1, 0.5, 2.5)
PARAM_ARRAYS['k_c'] = np.logspace(np.log10(0.2), np.log10(2), 10)
PARAM_ARRAYS['k_sc'] = np.logspace(np.log10(0.02), np.log10(0.2), 10)
PARAM_ARRAYS['thetaDot_1'] = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
PARAM_ARRAYS['T_delta'] = (10, 20, 40, 60, 90)
PARAM_ARRAYS['beta_V'] = (1, 3, 5, 9, 15, 27, 45, 81, 135, 243)
PARAM_ARRAYS['T_Of'] = (0.5, 1, 2, 4, math.inf)
PARAM_ARRAYS['sigma_O'] = (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5)




def run_fit(model_str):
    if 'oVA' in model_str:
        default_params_k = sc_fitting.DEFAULT_PARAMS_K_VA
    else:
        default_params_k = sc_fitting.DEFAULT_PARAMS_K_NVA
    assumptions = sc_scenario.get_assumptions_dict_from_string(model_str)
    this_fit = sc_fitting.SCPaperParameterSearch(
        model_str, sc_fitting.ONE_AG_SCENARIOS, assumptions, 
        sc_fitting.DEFAULT_PARAMS, default_params_k, PARAM_ARRAYS, verbosity=2)
    

if __name__ == "__main__":
    # get full list of models to fit
    models_to_fit = []
    for base_model in BASE_MODELS:
        for model_variant in MODEL_VARIANTS:
            models_to_fit.append(base_model + model_variant)
    # parallelise the model fits
    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(run_fit, models_to_fit)
    input('Done! Press [Enter] to exit...')
    