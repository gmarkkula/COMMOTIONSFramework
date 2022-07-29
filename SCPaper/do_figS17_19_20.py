# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 07:09:48 2022

@author: tragma
"""
import numpy as np
import matplotlib.pyplot as plt
import sc_fitting
import sc_plot
import do_1_deterministic_fitting
import do_5_probabilistic_fitting 

DO_RETAINED_COMB_PLOT = True
DO_EXCL_INCTRL_EXP_PLOT = True

SAVE_FIGS = False

MODEL_NAME = 'oVAoBEvoAIoEAoSNvoPF'
PARAMS_JITTER = 0.015


# load retained parameterisations from combined fits
ret_comb_models = sc_fitting.load_results(sc_fitting.RETAINED_COMB_FNAME)
for ret_model in ret_comb_models:
    if ret_model.model == MODEL_NAME:
        break
if ret_model.model != MODEL_NAME:
    raise Exception(f'Could not find model "{MODEL_NAME}" among retained combined models.')


# use original param ranges from deterministic/probabilistic fits for plots
param_ranges = []
for param_name in ret_model.param_names:
    if param_name in do_1_deterministic_fitting.PARAM_ARRAYS:
        param_values = do_1_deterministic_fitting.PARAM_ARRAYS[param_name]
    elif param_name in do_5_probabilistic_fitting.PARAM_ARRAYS:
        param_values = do_5_probabilistic_fitting.PARAM_ARRAYS[param_name]
    else:
        raise Exception(f'Could not find parameter {param_name}.')
    param_ranges.append((np.amin(param_values), np.amax(param_values)))


if DO_RETAINED_COMB_PLOT:
    # plot parameterisations retained from combined tests
    plt.close('all')
    params_array = ret_model.tested_params_array
    param_subsets = (np.arange(params_array.shape[0]),
                     ret_model.idx_retained)
    sc_fitting.do_params_plot(ret_model.param_names, 
                              params_array, 
                              param_ranges, 
                              param_subsets=param_subsets,
                              color=('lightgray', 'g'),
                              log=True, jitter=PARAMS_JITTER,
                              model_name=MODEL_NAME)
    if SAVE_FIGS:
        file_name = sc_plot.FIGS_FOLDER + 'figS17.png'
        print(f'Saving {file_name}...')
        plt.savefig(file_name, bbox_inches='tight', dpi=sc_plot.DPI)  


if DO_EXCL_INCTRL_EXP_PLOT:
    # plot parameterisations sampled when testing on controlled exp data, and the ones
    # exhibiting non-progressing behaviour
    FIRST_FIG_NO = 19
    for i_excl, file_name in enumerate((sc_fitting.EXCL_HIKER_FNAME, 
                                        sc_fitting.EXCL_DSS_FNAME)):
        excl_params = sc_fitting.load_results(file_name)
        params_array = excl_params[MODEL_NAME]['params_array']
        n_non_progress = excl_params[MODEL_NAME]['n_non_progress']
        param_subsets = (np.arange(params_array.shape[0]), 
                         n_non_progress >= 1,
                         n_non_progress >= 5)
        sc_fitting.do_params_plot(ret_model.param_names, 
                                  params_array, 
                                  param_ranges, 
                                  param_subsets=param_subsets,
                                  color = ('lightgray', 'deepskyblue', 'k'),
                                  log=True, jitter=PARAMS_JITTER,
                                  do_alpha=False, model_name=MODEL_NAME)
        if SAVE_FIGS:
            fig_no = FIRST_FIG_NO + i_excl
            file_name = sc_plot.FIGS_FOLDER + f'figS{fig_no}.png'
            print(f'Saving {file_name}...')
            plt.savefig(file_name, bbox_inches='tight', dpi=sc_plot.DPI)  
