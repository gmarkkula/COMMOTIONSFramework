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
BASE_MODEL = ''
MODELS_TO_RUN = ('', 'oBEo', 'oBEv', 'oBEooBEv', 'oBEvoAI', 'oBEooBEvoAI')


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
DEFAULT_PARAMS.T_P = 0.4
DEFAULT_PARAMS.beta_O = 1 
DEFAULT_PARAMS.T_O1 = 0.05 
#DEFAULT_PARAMS.DeltaV_th_rel = 0.001 
DEFAULT_PARAMS.DeltaT = 0.4 
DEFAULT_PARAMS.V_0_rel = 4 
DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.5, 0, 0.5, 1]) 
# -- free parameter values
PARAM_ARRAYS = {}
#PARAM_ARRAYS['T_P'] = (0.4, 0.8)
PARAM_ARRAYS['k_c'] = np.logspace(np.log10(0.2), np.log10(2), 4)
PARAM_ARRAYS['k_sc'] = np.logspace(np.log10(0.02), np.log10(0.2), 4)
PARAM_ARRAYS['T_delta'] = (15, 30, 60)
PARAM_ARRAYS['T'] = (0.2, 0.4, 0.6, 0.8)
PARAM_ARRAYS['beta_V'] = (5, 15, 45, 135)
PARAM_ARRAYS['T_Of'] = (0.5, 1, 2, math.inf)
PARAM_ARRAYS['sigma_O'] = (0.02, 0.1, 0.5, 2.5)




def run_fit(model_str):
    full_model_str = BASE_MODEL + model_str
    if 'oVA' in full_model_str:
        default_params_k = DEFAULT_PARAMS_K_VA
    else:
        default_params_k = DEFAULT_PARAMS_K_NVA
    assumptions = sc_scenario.get_assumptions_dict_from_string(full_model_str)
    this_fit = sc_fitting.SCPaperDeterministicOneSidedFitting(
        full_model_str, assumptions, DEFAULT_PARAMS, default_params_k,
        PARAM_ARRAYS, verbosity=2)
    

if __name__ == "__main__":
    #run_fit('oBEo')
    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(run_fit, MODELS_TO_RUN)
    input('Done! Press [Enter] to exit...')
    