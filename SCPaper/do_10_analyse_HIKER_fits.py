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
import parameter_search
import sc_fitting

# find pickle files from the HIKER fits
hiker_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                            (sc_fitting.HIKER_FIT_FILE_NAME_FMT % '*'))
hiker_fit_files.sort()
print(hiker_fit_files)


# loop through the fitting results files
for hiker_fit_file in hiker_fit_files:
    print()
    hiker_fit = parameter_search.load(hiker_fit_file, verbose=True)
    
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
                for i_param in range(hiker_fit.n_parameterisations):
                    param_cits = pd.DataFrame(
                        {'i_param': np.full(hiker_fit.n_repetitions, i_param),
                         'crossing_time': hiker_fit.results.metrics_matrix[
                             i_param, i_scen, :]})
                    model_cits[scen_name] = model_cits[scen_name].append(param_cits)
    
    
    sc_fitting.do_hiker_cit_cdf_plot(model_cits)
                    
                
                