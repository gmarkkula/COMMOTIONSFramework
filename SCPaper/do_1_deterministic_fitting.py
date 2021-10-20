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
import commotions
import sc_scenario
from sc_scenario_helper import CtrlType
import sc_fitting
import multiprocessing as mp


# set constants

# - models 
BASE_MODELS = ('', 'oVA', 'oVAa')
MODEL_VARIANTS = ('', 'oBEo', 'oBEv', 'oBEooBEv', 'oBEvoAI', 'oBEooBEvoAI')


# - model parameter constants
# -- fixed parameter values
# --- fixed value function gains, affordance-based
DEFAULT_PARAMS_K_VA = {}
for ctrl_type in CtrlType:
    DEFAULT_PARAMS_K_VA[ctrl_type] = commotions.Parameters()
    DEFAULT_PARAMS_K_VA[ctrl_type]._da = 0.5
# --- fixed value function gains, non-affordance-based
DEFAULT_PARAMS_K_NVA = {}
for ctrl_type in CtrlType:
    DEFAULT_PARAMS_K_NVA[ctrl_type] = commotions.Parameters()
    DEFAULT_PARAMS_K_NVA[ctrl_type]._e = 0
    # these two are only really applicable to acceleration-controlling agents, 
    # but there is no harm in adding them here for both agents
    DEFAULT_PARAMS_K_NVA[ctrl_type]._da = 0.01
    DEFAULT_PARAMS_K_NVA[ctrl_type]._sg = 0
# --- other fixed parameters
DEFAULT_PARAMS = commotions.Parameters()
DEFAULT_PARAMS.T_P = 0.5
DEFAULT_PARAMS.T_s = 0.5
DEFAULT_PARAMS.D_s = 0.5
DEFAULT_PARAMS.beta_O = 1 
DEFAULT_PARAMS.T_O1 = 0.05 
#DEFAULT_PARAMS.DeltaV_th_rel = 0.001 
DEFAULT_PARAMS.DeltaT = 0.5 
DEFAULT_PARAMS.V_0_rel = 4 
DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]) 
# -- free parameter values
PARAM_ARRAYS = {}
PARAM_ARRAYS['T'] = (0.2, 0.4, 0.6, 0.8)
PARAM_ARRAYS['k_c'] = np.logspace(np.log10(0.2), np.log10(2), 4)
PARAM_ARRAYS['k_sc'] = np.logspace(np.log10(0.02), np.log10(0.2), 4)
PARAM_ARRAYS['T_delta'] = (15, 30, 60)
PARAM_ARRAYS['beta_V'] = (5, 15, 45, 135)
PARAM_ARRAYS['T_Of'] = (0.5, 1, 2, math.inf)
PARAM_ARRAYS['sigma_O'] = (0.02, 0.1, 0.5, 2.5)
# PARAM_ARRAYS['k_c'] = np.logspace(np.log10(0.2), np.log10(2), 10)
# PARAM_ARRAYS['k_sc'] = np.logspace(np.log10(0.02), np.log10(0.2), 10)
# PARAM_ARRAYS['T_delta'] = (10, 20, 40, 60, 90)
# PARAM_ARRAYS['beta_V'] = (1, 3, 5, 9, 15, 27, 45, 81, 135, 243)
# PARAM_ARRAYS['T_Of'] = (0.5, 1, 2, 4, math.inf)
# PARAM_ARRAYS['sigma_O'] = (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5)




def run_fit(model_str):
    if 'oVA' in model_str:
        default_params_k = DEFAULT_PARAMS_K_VA
    else:
        default_params_k = DEFAULT_PARAMS_K_NVA
    assumptions = sc_scenario.get_assumptions_dict_from_string(model_str)
    this_fit = sc_fitting.SCPaperDeterministicOneSidedFitting(
        model_str, assumptions, DEFAULT_PARAMS, default_params_k,
        PARAM_ARRAYS, verbosity=2)
    

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
    