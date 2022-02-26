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
from do_2_analyse_deterministic_fits import (get_max_crit_parameterisations, 
                                             get_best_parameterisations_for_crit,
                                             get_best_scen_var_for_paramet)


PLOT_ILLUSTRATIONS = True
PLOT_METRICS = True
PLOT_METRIC_VALUES = True
PLOT_DET_EXAMPLES = True
OVERWRITE_SAVED_DET_SIM_RESULTS = False
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
fig, fig_axs = plt.subplots(num=1, nrows = N_ROWS, ncols = N_COLS, 
                        figsize=(sc_plot.FULL_WIDTH, 0.8*sc_plot.FULL_WIDTH),
                        dpi=DPI, tight_layout=True)

if PLOT_ILLUSTRATIONS:
    print('Plotting illustrations...')
    for i_scenario, scenario in enumerate(SCENARIO_NAMES):
        ax = fig_axs[0, i_scenario]
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
        
    
CRIT_BASES = ('', 'oVA')
BASE_DASHES = ((10,2), (1,0))
CRIT_VARIANTS = ('', 'oBEo', 'oBEvoAI')
VARIANT_COLORS = (sc_plot.COLORS['base variant black'],
                  sc_plot.COLORS['oBEo variant yellow'],
                  'green')
        
if PLOT_METRICS:
    
    print('Plotting metric distributions...')
    
    def transf_kde(x):
        kde_y = kde(x) * yscale
        kde_y[kde_y < 1e-6] = np.nan
        return base_y + kde_y ** 0.3
    
    for i_scenario in range(len(SCENARIO_NAMES)):
        xlims = SCENARIO_METRIC_XLIMS[i_scenario]
        xrange = xlims[1] - xlims[0]
        yscale = xrange/70
        for i_base, base in enumerate(CRIT_BASES):
            ax = fig_axs[i_base + 1, i_scenario]
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

def run_one_det_sim(model_name, params_dict, scenario, i_var):
    print(f'Simulating {model_name} with '
          f' {params_dict} in variant #{i_var+1} scenario {scenario.name}...')
    return sc_fitting.construct_model_and_simulate_scenario(
        model_name, params_dict, scenario, i_variation=i_var, 
        apply_stop_criteria=False, zero_acc_after_exit=False)

print('Plotting deterministic model time series examples...')

# the models to show simulations for
DET_MODEL_NAMES = ('', 'oBEo', 'oVAoBEvoAI')
DET_MODEL_BASES = (0, 0, 1)
DET_MODEL_VARIANTS = (0, 1, 2)
# criterion to use for selecting a parameterisation to show for each model   
MODEL_FOCUS_CRITS = ('Yield acceptance hesitation', 'Gap acceptance hesitation', 
                     'Priority assertion')
DEFAULT_FOCUS_MODEL = 'oVAoBEvoAI'
         
# get simulation results, by loading existing, or looping through models and simulating
if sc_fitting.results_exist(DET_SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_DET_SIM_RESULTS:
    det_sims = sc_fitting.load_results(DET_SIM_RESULTS_FNAME)
else:
    
    # first, figure out which model parameterisation to use for each model
    idx_plot_params = []
    for i_model, model_name in enumerate(DET_MODEL_NAMES):
        det_fit = det_fits[model_name]
        idx_max_crit_params = get_max_crit_parameterisations(det_fit)
        idx_best_params_for_focus_crit = get_best_parameterisations_for_crit(
            det_fit, MODEL_FOCUS_CRITS[i_model], 
            idx_params_subset=idx_max_crit_params)
        idx_plot_params.append(idx_best_params_for_focus_crit[0])
        
    # then, figure out which kinematic variant to use for each scenario
    i_plot_kinem_vars = []
    for i_scenario, scenario_name in enumerate(SCENARIO_NAMES):
        scen_crit = SCENARIO_CRITERIA[i_scenario]
        if scen_crit in MODEL_FOCUS_CRITS:
            focus_model_name = DET_MODEL_NAMES[MODEL_FOCUS_CRITS.index(scen_crit)]
        else:
            focus_model_name = DEFAULT_FOCUS_MODEL
        det_fit = det_fits[focus_model_name]
        i_focus_model = DET_MODEL_NAMES.index(focus_model_name)
        i_plot_kinem_vars.append(
            get_best_scen_var_for_paramet(det_fit, idx_plot_params[i_focus_model], 
                                          scenario_name, verbose=True))
        
    # now simulate the models
    det_sims = {}
    for i_model, model_name in enumerate(DET_MODEL_NAMES):
        det_fit = det_fits[model_name]
        params_array = det_fit.results.params_matrix[idx_plot_params[i_model]]
        params_dict = det_fit.get_params_dict(params_array)

        # - loop through scenarios and run simulations
        det_sims[model_name] = {}
        for i_scenario, scenario in enumerate(SCENARIOS.values()):
            i_var = i_plot_kinem_vars[i_scenario]
            det_sims[model_name][scenario.name] = run_one_det_sim(
                model_name, params_dict, scenario, i_var)
    # save simulation results
    sc_fitting.save_results(det_sims, DET_SIM_RESULTS_FNAME)
    

# plot
PED_V_LIMS = (-.5, 2.5)
V_LIMS = ((12.5, 14.5), (-1, 17), PED_V_LIMS, PED_V_LIMS, PED_V_LIMS)
T_MAXS = (3, 7.3, 7.3, 7.3, 6)
for i_model, model_name in enumerate(DET_MODEL_NAMES):
    det_fit = det_fits[model_name]
    for i_scenario, scenario_name in enumerate(SCENARIO_NAMES):
        axs = fig_axs[3:5, i_scenario]
        sim = det_sims[model_name][scenario_name]
        
        # get the active agent and set colours
        act_agent = None
        i_act_agent = None
        for i_agent, agent in enumerate(sim.agents):
            if agent.const_acc == None:
                act_agent = agent
                i_act_agent = i_agent
                break
        act_agent.plot_color = VARIANT_COLORS[DET_MODEL_VARIANTS[i_model]]
        act_agent.other_agent.plot_color = sc_plot.COLORS['passive agent grey']
        if i_model == 0:
            i_plot_agents = (1-i_act_agent, i_act_agent)
            agent_alpha = (1, 1)
        else:
            i_plot_agents = (i_act_agent,)
            agent_alpha = (1,)
            
        # plot
        sim.do_kinem_states_plot(np.insert(axs, 0, None), veh_stop_dec=False, 
                                 agent_alpha=agent_alpha,
                                 i_plot_agents=i_plot_agents,
                                 axis_labels=(i_scenario==0),
                                 plot_fill=True)
        axs[0].set_xlim(0, T_MAXS[i_scenario])
        axs[0].set_ylim(V_LIMS[i_scenario][0], V_LIMS[i_scenario][1])
        axs[1].set_xlim(0, T_MAXS[i_scenario])
        axs[1].set_ylim(-4, 17)
        axs[1].set_xlabel('Time (s)')
        

plt.show()


if SAVE_PDF:
    file_name = sc_plot.FIGS_FOLDER + 'fig1.pdf'
    print(f'Saving {file_name}...')
    plt.savefig(file_name, bbox_inches='tight')