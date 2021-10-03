# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 06:14:22 2021

@author: tragma
"""

import sys
sys.path.append('C:\\GITHUB\\COMMOTIONSFramework\\')

import math
import numpy as np
import commotions
import sc_scenario
from sc_scenario_helper import CtrlType
import sc_fitting
import multiprocessing as mp


# set constants

# - models 
BASE_MODEL = 'oVAoEA'
MODELS_TO_RUN = ('', 'oBEo', 'oBEv', 'oBEooBEv', 'oBEvoAI', 'oBEooBEvoAI')


# - model parameter constants
# -- fixed parameter values
DEFAULT_PARAMS = commotions.Parameters()
DEFAULT_PARAMS.beta_O = 1 
DEFAULT_PARAMS.T_O1 = 0.05 
DEFAULT_PARAMS.DeltaV_th_rel = 0.001 
DEFAULT_PARAMS.DeltaT = 0.4 
DEFAULT_PARAMS.V_0_rel = 4 
DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.5, 0, 0.5, 1]) 
DEFAULT_PARAMS_K = {}
for ctrl_type in CtrlType:
    DEFAULT_PARAMS_K[ctrl_type] = commotions.Parameters()
    DEFAULT_PARAMS_K[ctrl_type]._da = 0.5
# -- free parameter values
PARAM_ARRAYS = {}
PARAM_ARRAYS['T_P'] = (0.4, 0.8)
PARAM_ARRAYS['T_delta'] = (15, 30, 60)
PARAM_ARRAYS['T'] = (0.2, 0.4, 0.6, 0.8)
PARAM_ARRAYS['beta_V'] = (15, 30, 60, 120)
PARAM_ARRAYS['T_Of'] = (0.5, 1, 2, math.inf)
PARAM_ARRAYS['sigma_O'] = (0.02, 0.1, 0.5, 2.5)




def run_fit(model_str):
    full_model_str = BASE_MODEL + model_str
    assumptions = sc_scenario.get_assumptions_dict_from_string(full_model_str)
    this_fit = sc_fitting.SCPaperDeterministicOneSidedFitting(
        full_model_str, assumptions, DEFAULT_PARAMS, DEFAULT_PARAMS_K,
        PARAM_ARRAYS, verbosity=2)
    

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(run_fit, MODELS_TO_RUN)
    input('Done! Press [Enter] to exit...')
    