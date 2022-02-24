# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 09:58:07 2022

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
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parameter_search
import sc_fitting


MAX_NON_CROSS_TRIALS = np.inf
if not np.isinf(MAX_NON_CROSS_TRIALS):
    print(f'NB: Excluding parameterisations with more than {MAX_NON_CROSS_TRIALS}'
          ' non-crossing trials.\n')


# find pickle files from the HIKER fits
hiker_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                            (sc_fitting.HIKER_FIT_FILE_NAME_FMT % '*'))
hiker_fit_files.sort()
print(hiker_fit_files)


# loop through the fitting results files
n_total_params_tested = 0
for hiker_fit_file in hiker_fit_files:
    print()
    hiker_fit = parameter_search.load(hiker_fit_file, verbose=True)
    print(f'\tParameterisations tested: {hiker_fit.n_parameterisations}')
    n_total_params_tested += hiker_fit.n_parameterisations
    
    # go through
    i_included_params = []
    n_non_cross_trials = []
    for i_param in range(hiker_fit.n_parameterisations):
        n_non_cross_trials.append(np.count_nonzero(np.isnan(hiker_fit.results.metrics_matrix[i_param, :, :])))
        if n_non_cross_trials[-1] <= MAX_NON_CROSS_TRIALS:
            i_included_params.append(i_param)
    n_non_cross_trials = np.array(n_non_cross_trials)
    idx_max_non_cross_trials = np.nonzero(n_non_cross_trials == np.amax(n_non_cross_trials))[0]
    print(f'Example parameterisations with max for model of'
          f' {np.amax(n_non_cross_trials)} non-cross trials:')
    params_array = hiker_fit.results.params_matrix[idx_max_non_cross_trials[0]]
    print(hiker_fit.get_params_dict(params_array))
    
    
    if len(i_included_params) > 0:
        print(f'Plotting for {len(i_included_params)} included parameterisations:')
        # reorganise data for plotting/analysis
        model_cits = {}
        for i_speed, veh_speed_mph in enumerate(sc_fitting.HIKER_VEH_SPEEDS_MPH):
            for i_gap, veh_time_gap in enumerate(sc_fitting.HIKER_VEH_TIME_GAPS):
                for i_yield, veh_yielding in enumerate((False, True)):
                    scen_name = sc_fitting.get_hiker_scen_name(veh_speed_mph, 
                                                               veh_time_gap, 
                                                               veh_yielding)
                    i_scen = hiker_fit.metric_names.index(
                        scen_name + '_' + sc_fitting.HIKER_CIT_METRIC_NAME)
                    model_cits[scen_name] = pd.DataFrame(columns=('i_param', 
                                                                  'crossing_time'))
                    for i_param in i_included_params:
                        param_cits = pd.DataFrame(
                            {'i_param': np.full(hiker_fit.n_repetitions, i_param),
                             'crossing_time': hiker_fit.results.metrics_matrix[
                                 i_param, i_scen, :]})
                        model_cits[scen_name] = model_cits[scen_name].append(param_cits)
        
        
        sc_fitting.do_hiker_cit_cdf_plot(model_cits, fig_name=hiker_fit.name)

print(f'*** Total number of parameterisations tested: {n_total_params_tested}')                 
                
                