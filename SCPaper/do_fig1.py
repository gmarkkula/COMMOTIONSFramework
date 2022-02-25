# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 07:43:03 2022

@author: tragma
"""

# note to self: might want to consider: https://github.com/btel/svg_utils
# from matplotlib.offsetbox import OffsetImage

import os, contextlib
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats
import sc_fitting
import sc_plot
import do_2_analyse_deterministic_fits
from do_2_analyse_deterministic_fits import get_max_crit_parameterisations, get_best_scen_var_for_paramet


OVERWRITE_SAVED_SIM_RESULTS = False
PLOT_ILLUSTRATIONS = True
PLOT_METRICS = True
PLOT_METRIC_VALUES = True
SAVE_PDF = False

DPI = sc_plot.DPI / 3

SCENARIOS = sc_fitting.ONE_AG_SCENARIOS
SCENARIO_NAMES = SCENARIOS.keys()
N_COLS = len(SCENARIO_NAMES)
N_ROWS = 5
SCENARIO_CRITERIA = ('Priority assertion', 'Short-stopping', 'Gap acceptance hesitation',
          'Yield acceptance hesitation', 'Early yield acceptance')
SCENARIO_METRIC_NAMES = ('$\overline{v}_\mathrm{v}/v_\mathrm{v,free}$ (-)', 
                         '$\overline{d}/d_\mathrm{stop}$ (-)',
                         '$\overline{v}_\mathrm{p}/v_\mathrm{p,free}$ (-)',
                         '$\overline{v}_\mathrm{p}/v_\mathrm{p,free}$ (-)', 
                         '$v_\mathrm{v}(t_\mathrm{cross})$ (m/s)')
SCENARIO_METRIC_XLIMS = ((0.95, 1.02), (-1, 4), (0.7, 1.05), (0, 1.5), (-2, 15))
N_KDE_POINTS = 200

DET_SIM_RESULTS_FNAME = 'fig_1_DetFitSimResults.pkl'

rng = default_rng(0)

plt.close('all')
fig, axs = plt.subplots(num=1, nrows = N_ROWS, ncols = N_COLS, 
                        figsize=(sc_plot.FULL_WIDTH, 0.8*sc_plot.FULL_WIDTH),
                        dpi=DPI, tight_layout=True)

if PLOT_ILLUSTRATIONS:
    print('Plotting illustrations...')
    for i_scenario, scenario in enumerate(SCENARIO_NAMES):
        ax = axs[0, i_scenario]
        scen_im = plt.imread(sc_plot.FIGS_FOLDER + scenario + '.png')
        ax.imshow(scen_im)
        ax.axis('off')
        ax.set_title(SCENARIO_CRITERIA[i_scenario] + '\n')
  
print('Running do_2...')       
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull):
        do_2_analyse_deterministic_fits.DO_PLOTS = False
        do_2_analyse_deterministic_fits.SAVE_RETAINED_MODELS = False
        det_fits = do_2_analyse_deterministic_fits.do()
        
        
if PLOT_METRICS:
    
    print('Plotting metric distributions...')
    
    CRIT_BASES = ('', 'oVA')
    BASE_DASHES = ((10,2), (1,0))
    CRIT_VARIANTS = ('', 'oBEo', 'oBEvoAI')
    VARIANT_COLORS = (sc_plot.COLORS['base variant black'],
                      sc_plot.COLORS['oBEo variant yellow'],
                      'green')
    
    def transf_kde(x):
        kde_y = kde(x) * yscale
        kde_y[kde_y < 1e-6] = np.nan
        return base_y + kde_y ** 0.3
    
    for i_scenario in range(len(SCENARIO_NAMES)):
        xlims = SCENARIO_METRIC_XLIMS[i_scenario]
        xrange = xlims[1] - xlims[0]
        yscale = xrange/70
        for i_base, base in enumerate(CRIT_BASES):
            ax = axs[i_base + 1, i_scenario]
            for i_variant, variant in enumerate(CRIT_VARIANTS):
                base_y = len(CRIT_VARIANTS) - i_variant
                color = VARIANT_COLORS[i_variant]
                model_name = base + variant
                det_fit = det_fits[model_name]
                crit_details = det_fit.crit_details[SCENARIO_CRITERIA[i_scenario]]
                # if crit_details.crit_greater_than:
                #     plot_values = np.nanmax(crit_details.metric_values, axis=1)
                # else:
                #     plot_values = np.nanmin(crit_details.metric_values, axis=1)
                plot_values = crit_details.metric_values.reshape(-1, 1)
                plot_values = plot_values[~np.isnan(plot_values)]
                plot_values_jittered = plot_values + (xrange/100) * rng.uniform(-1, 1, len(plot_values))
                kde = stats.gaussian_kde(plot_values_jittered, bw_method = (xrange/150)/np.std(plot_values_jittered)) 
                kde_x = np.linspace(xlims[0], xlims[1], N_KDE_POINTS)
                kde_y = transf_kde(kde_x)
                if PLOT_METRIC_VALUES:
                    plot_values_unique = np.unique(plot_values)
                    ax.vlines(plot_values_unique, base_y + 0.02, transf_kde(plot_values_unique), 
                              color=sc_plot.lighten_color(color, 0.7), lw=0.7)
                ax.plot(kde_x, kde_y, dashes = BASE_DASHES[i_base], color=color)
                ax.set_xlim(xlims[0], xlims[1])
                ax.set_ylim(0.5, len(CRIT_VARIANTS) + 1)
                ax.get_yaxis().set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if i_base == 0:
                    ax.get_xaxis().set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                else:
                    ax.set_xlabel(SCENARIO_METRIC_NAMES[i_scenario])
            
            
#for i_scenario, scenario in enumerate(scenario_names)

# print('Plotting deterministic model time series examples...')

# DET_MODEL_NAMES = ('', 'oBEo', 'oVAoBEvoAI')   
         
# # get simulation results, by loading existing, or looping through models and simulating
# if sc_fitting.results_exist(DET_SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_SIM_RESULTS:
#     sims = sc_fitting.load_results(DET_SIM_RESULTS_FNAME)
# else:
#     sims = {}
#     for model_name in DET_MODEL_NAMES:
#         det_fit = det_fits[model_name]
            
#         # get parameterisation to simulate


#         # - loop through scenarios and run simulations
#         sims[model_name] = {}
#         for scenario in SCENARIOS.values():
#             sim_iter = ((model_name, i, n_paramets, params_dicts[i], scenario) 
#                         for i in range(n_paramets))
#             if PARALLEL:
#                 sims[model_name][scenario.name] = list(
#                     pool.starmap(run_one_sim, sim_iter)) 
#             else:
#                 sims[model_name][scenario.name] = list(
#                     itertools.starmap(run_one_sim, sim_iter)) 
#             # also store the parameterisation indices
#             sims[model_name]['idx_paramets'] = idx_paramets
#             # sim = sc_fitting.construct_model_and_simulate_scenario(
#             #     model_name, params_dict, scenario, apply_stop_criteria=False,
#             #     zero_acc_after_exit=False)
#             #sims[model_name][scenario_name].append(sim)
#     # save simulation results
#     sc_fitting.save_results(sims, SIM_RESULTS_FNAME)

# if SAVE_PDF:
#     file_name = sc_plot.FIGS_FOLDER + 'fig1.pdf'
#     print(f'Saving {file_name}...')
#     plt.savefig(file_name, bbox_inches='tight')