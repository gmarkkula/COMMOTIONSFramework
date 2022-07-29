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
import sc_plot

DO_PARAMS_PLOT = False
DO_CIT_CDF_PLOTS = True # supplementary figure
PLOT_ALL_CIT_CDFS = False
CIT_CDF_PLOT_MODELS_BASES = ('oVAoBEvoAI', 'oVAoBEooBEvoAI', 'oVAaoVAloBEvoAI')
CIT_CDF_PLOT_MODELS_VARIANTS = ('oSNv', 'oEAoSNvoPF', 'oEAoSNv', 'oDAoSNvoPF', 'oDAoSNv')
CIT_CDF_PLOT_MODELS = []
for base in CIT_CDF_PLOT_MODELS_BASES:
    for variant in CIT_CDF_PLOT_MODELS_VARIANTS:
        CIT_CDF_PLOT_MODELS.append(base + variant)
CIT_CDF_LEGEND_MODEL = 'oVAoBEvoAIoSNv'
CIT_CDF_FIG_NO = 18
SAVE_FIGS = False
SAVE_CIT_DATA_FOR_MODELS = ('oVAoBEvoAIoEAoSNvoPF', 'oVAoBEvoAIoDAoSNvoPF')

CIT_CDF_MAX_NON_CROSS_TRIALS = np.inf
if not np.isinf(CIT_CDF_MAX_NON_CROSS_TRIALS):
    print('NB: Excluding parameterisations with more than'
          f'{CIT_CDF_MAX_NON_CROSS_TRIALS} non-crossing trials from CIT CDFs.\n')

plt.close('all')

# find pickle files from the HIKER fits
hiker_fit_files = glob.glob(sc_fitting.FIT_RESULTS_FOLDER + 
                            (sc_fitting.HIKER_FIT_FILE_NAME_FMT % '*'))
hiker_fit_files.sort()
print(hiker_fit_files)


# loop through the fitting results files
n_total_params_tested = 0
excl_params = {}
for hiker_fit_file in hiker_fit_files:
    print()
    hiker_fit = parameter_search.load(hiker_fit_file, verbose=True)
    print(f'\tParameterisations tested: {hiker_fit.n_parameterisations}')
    n_total_params_tested += hiker_fit.n_parameterisations
    
    idx_yielding_cits = [i for i, name in enumerate(hiker_fit.metric_names) if 'y' in name]
    yielding_cits = hiker_fit.results.metrics_matrix[:, idx_yielding_cits, :]
    n_noncross = np.count_nonzero(np.isnan(yielding_cits), axis=(1, 2))
    noncross_yield_params = np.any(np.isnan(yielding_cits), axis=(1, 2))
    excl_params[hiker_fit.name] = {}
    excl_params[hiker_fit.name]['params_array'] = hiker_fit.results.params_matrix
    excl_params[hiker_fit.name]['n_non_progress'] = n_noncross
    excl_params[hiker_fit.name]['rejected'] = noncross_yield_params
    cross_yield_params = ~noncross_yield_params
    n_cross_yield_params = np.count_nonzero(cross_yield_params)
    print(f'\tFound {n_cross_yield_params}'
          f' ({100 * n_cross_yield_params / hiker_fit.n_parameterisations:.1f} %)'
          ' parameterisations which always crossed in vehicle yielding scenarios.')
    
    if DO_PARAMS_PLOT:
        sc_fitting.do_params_plot(hiker_fit.param_names, 
                                  hiker_fit.results.params_matrix, 
                                  jitter=0.01, color=('r', 'g'), 
                                  param_subsets=(noncross_yield_params, 
                                                 cross_yield_params))
    
    do_this_cit_cdf_plot = DO_CIT_CDF_PLOTS and (
        (hiker_fit.name in CIT_CDF_PLOT_MODELS) or PLOT_ALL_CIT_CDFS)
    if do_this_cit_cdf_plot or (hiker_fit.name in SAVE_CIT_DATA_FOR_MODELS):
        i_included_params = np.nonzero(cross_yield_params)[0]
        if len(i_included_params) == 0:
            print('\t*** No parameterisations to plot ***')
        else:
            print(f'\tPreparing CIT plotting data for {len(i_included_params)} included parameterisations:')
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
            if do_this_cit_cdf_plot:
                sc_fitting.do_hiker_cit_cdf_plot(model_cits, 
                                                 fig_name=hiker_fit.name,
                                                 legend=(hiker_fit.name 
                                                         == CIT_CDF_LEGEND_MODEL),
                                                 show_name_in_fig=True,
                                                 finalise=False)
                if SAVE_FIGS:
                    file_name = f'figS{CIT_CDF_FIG_NO}_{hiker_fit.name}.pdf'
                    file_path = sc_plot.FIGS_FOLDER + file_name
                    print(f'Saving {file_path}...')
                    plt.savefig(file_path, bbox_inches='tight')
                plt.show()
            if hiker_fit.name in SAVE_CIT_DATA_FOR_MODELS:
                sc_fitting.save_results(
                    model_cits, sc_fitting.MODEL_CIT_FNAME_FMT % hiker_fit.name)

print(f'*** Total number of parameterisations tested: {n_total_params_tested}')                 
                
sc_fitting.save_results(excl_params, sc_fitting.EXCL_HIKER_FNAME)           