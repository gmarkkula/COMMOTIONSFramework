# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:32:29 2022

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
import os, contextlib
import warnings
import math
import multiprocessing as mp
import numpy as np
import sc_scenario
import sc_fitting
import do_2_analyse_deterministic_fits
from do_2_analyse_deterministic_fits import CRITERIA, i_MAIN


MAX_PARAMETERISATIONS = 1000
if not math.isinf(MAX_PARAMETERISATIONS):
    warnings.warn(f'Limiting the number of parameterisations per model to {MAX_PARAMETERISATIONS}.')
    rng = np.random.default_rng()


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
        model_str, {'_' : sc_fitting.ALT_SHORTSTOP_SCENARIO}, assumptions, 
        sc_fitting.DEFAULT_PARAMS, default_params_k, 
        param_arrays, list_search=True,
        n_repetitions=sc_fitting.N_ONE_AG_SCEN_VARIATIONS, parallel=True,
        n_workers=n_workers, verbosity=verbosity,
        file_name_format=sc_fitting.ALT_SHORTSTOP_FIT_FILE_NAME_FMT,
        overwrite_existing=False)


if __name__ == '__main__':

    # do the criterion analysis of the deterministic fits (without any output)
    do_2_analyse_deterministic_fits.DO_TIME_SERIES_PLOTS = False
    do_2_analyse_deterministic_fits.DO_PARAMS_PLOTS = False
    do_2_analyse_deterministic_fits.DO_RETAINED_PARAMS_PLOT = False
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            det_fits = do_2_analyse_deterministic_fits.do()
            
    # loop through the models and retest any parameterisations which were found
    # to achieve short-stopping in the analyses so far
    idx_short_stop = CRITERIA[i_MAIN].index('Vehicle short-stopping')
    for model_str, det_fit in det_fits.items():
        idx_did_short_stop = det_fit.main_criteria_matrix[idx_short_stop, :]
        n_did_short_stop = len(np.nonzero(idx_did_short_stop)[0])
        print(f'\nModel {model_str}: Found {n_did_short_stop} parameterisations'
              ' achieving short-stopping in the "exaggerated early deceleration" sense.')
        if n_did_short_stop > 0:
            print('\tChecking these in the "exaggerated final stopping margin" sense:\n')
            params_matrix = det_fit.results.params_matrix[idx_did_short_stop, :]
            if n_did_short_stop <= MAX_PARAMETERISATIONS:
                idx_included = np.arange(n_did_short_stop)
            else:
                print(f'(Limiting to a random sample of {MAX_PARAMETERISATIONS} parameterisations.)')
                idx_included = rng.choice(n_did_short_stop, 
                                          size=MAX_PARAMETERISATIONS, replace=False)
            param_arrays_dict = {}
            for i_param, param_name in enumerate(det_fit.param_names):
                param_arrays_dict[param_name] = params_matrix[idx_included, i_param]
            run_fit(model_str, param_arrays_dict)
            
            
            
            