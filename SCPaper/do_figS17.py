# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 07:09:48 2022

@author: tragma
"""
import numpy as np
import matplotlib.pyplot as plt
import sc_fitting
import do_1_deterministic_fitting
import do_5_probabilistic_fitting 

MODEL_NAME = 'oVAoBEvoAIoEAoSNvoPF'
PARAMS_JITTER = 0.015


# load retained parameterisations from combined fits
ret_comb_models = sc_fitting.load_results(sc_fitting.RETAINED_COMB_FNAME)
for ret_model in ret_comb_models:
    if ret_model.model == MODEL_NAME:
        break
if ret_model.model != MODEL_NAME:
    raise Exception(f'Could not find model "{MODEL_NAME}" among retained combined models.')


# find parameterisations rejected in interaction/DSS/HIKER tests


# plot showing original param ranges from deterministic/probabilistic fits
param_ranges = []
for param_name in ret_model.param_names:
    if param_name in do_1_deterministic_fitting.PARAM_ARRAYS:
        param_values = do_1_deterministic_fitting.PARAM_ARRAYS[param_name]
    elif param_name in do_5_probabilistic_fitting.PARAM_ARRAYS:
        param_values = do_5_probabilistic_fitting.PARAM_ARRAYS[param_name]
    else:
        raise Exception(f'Could not find parameter {param_name}.')
    param_ranges.append((np.amin(param_values), np.amax(param_values)))

# plot
plt.close('all')
sc_fitting.do_params_plot(ret_model.param_names, 
                          ret_model.params_array, 
                          param_ranges, 
                          log=True, jitter=PARAMS_JITTER)