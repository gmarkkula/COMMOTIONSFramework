# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:33:39 2022

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
import itertools
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import sc_scenario
import sc_fitting
import sc_plot
import do_2_analyse_deterministic_fits
from do_2_analyse_deterministic_fits import get_max_crit_parameterisations 
from do_2_analyse_deterministic_fits import get_best_scen_var_for_paramet 

OVERWRITE_SAVED_SIM_RESULTS = False
PARALLEL = True

MODEL_NAMES = ('', 'oBEo', 'oBEvoAI', 'oVAoBEvoAI')
#MODEL_NAMES = ('oVAoBEvoAI',)
N_MAX_PARAMETS_PER_MODEL = 100
SCENARIOS = sc_fitting.ONE_AG_SCENARIOS
SIM_RESULTS_FNAME = 'fig_DetFitSimResults.pkl'
ALPHA = 0.1


def run_one_sim(model_name, i_paramet, n_paramets, params_dict, scenario):
    print(f'Simulating {model_name} with parameterisation'
          f' #{i_paramet+1}/{n_paramets}:'
          f' {params_dict} scenario {scenario.name}...')
    sim_vars = []
    for i_var in range(scenario.n_variations):
        print(f'\tVariation #{i_var+1}/{scenario.n_variations}...')
        sim_vars.append(sc_fitting.construct_model_and_simulate_scenario(
            model_name, params_dict, scenario, i_variation=i_var, 
            apply_stop_criteria=False, zero_acc_after_exit=False))
    return sim_vars


if __name__ == '__main__':
    
    # run do_2... without any output
    do_2_analyse_deterministic_fits.DO_PLOTS = False 
    do_2_analyse_deterministic_fits.SPEEDUP_FRACT = 1.005
    
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            det_fits = do_2_analyse_deterministic_fits.do()

    # initialise random number generator
    rng = np.random.default_rng(seed=0)    
    
    # get simulation results, by loading existing, or looping through models and simulating
    if sc_fitting.results_exist(SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_SIM_RESULTS:
        sims = sc_fitting.load_results(SIM_RESULTS_FNAME)
    else:
        print('Starting pool of workers...')
        if PARALLEL:
            pool = mp.Pool(mp.cpu_count()-1)
        sims = {}
        for model_name in MODEL_NAMES:
            det_fit = det_fits[model_name]
                
            # get parameterisations to simulate
            n_paramets_total = det_fit.results.params_matrix.shape[0]
            if n_paramets_total > N_MAX_PARAMETS_PER_MODEL:
                idx_paramets = rng.choice(n_paramets_total, 
                                          size=N_MAX_PARAMETS_PER_MODEL,
                                          replace=False)
                idx_paramets.sort()
            else:
                idx_paramets = np.arange(n_paramets_total)
            print(idx_paramets)
            n_paramets = len(idx_paramets)
            # - get list of parameterisation dicts
            params_dicts = []
            for idx_paramet in idx_paramets:
                params_array = det_fit.results.params_matrix[idx_paramet, :]
                params_dicts.append(dict(
                    zip(det_fit.param_names, params_array)))
            # - loop through scenarios and run simulations
            sims[model_name] = {}
            for scenario in SCENARIOS.values():
                sim_iter = ((model_name, i, n_paramets, params_dicts[i], scenario) 
                            for i in range(n_paramets))
                if PARALLEL:
                    sims[model_name][scenario.name] = list(
                        pool.starmap(run_one_sim, sim_iter)) 
                else:
                    sims[model_name][scenario.name] = list(
                        itertools.starmap(run_one_sim, sim_iter)) 
                # also store the parameterisation indices
                sims[model_name]['idx_paramets'] = idx_paramets
                # sim = sc_fitting.construct_model_and_simulate_scenario(
                #     model_name, params_dict, scenario, apply_stop_criteria=False,
                #     zero_acc_after_exit=False)
                #sims[model_name][scenario_name].append(sim)
        # save simulation results
        sc_fitting.save_results(sims, SIM_RESULTS_FNAME)

# In[1]

    # plot
    PED_V_LIMS = (-.5, 2.5)
    V_LIMS = ((12.5, 14.5), (-1, 20), PED_V_LIMS, PED_V_LIMS, PED_V_LIMS)
    T_MAXS = (3, 7.5, 8, 8, 6)
    plt.close('all')
    for model_name in MODEL_NAMES:
        det_fit = det_fits[model_name]
        fig_width = 2 + 2 * len(SCENARIOS)
        fig, fig_axs = plt.subplots(nrows=2, ncols=len(SCENARIOS), 
                                figsize=(fig_width, 4),
                                sharex='col', tight_layout=True,
                                num='_' + model_name + '_')
        idx_best_paramets = get_max_crit_parameterisations(det_fit)
        for i_scenario, scenario_name in enumerate(SCENARIOS.keys()):
            axs = fig_axs[:, i_scenario]
            axs = np.insert(axs, 0, None)
            i_var = get_best_scen_var_for_paramet(det_fit, idx_best_paramets[0], 
                                                  scenario_name, verbose=False)
            for i_sim_vars, sim_vars in enumerate(sims[model_name][scenario_name]):
                sim = sim_vars[i_var]
                
                # get the active agent and set colours
                act_agent = None
                i_act_agent = None
                for i_agent, agent in enumerate(sim.agents):
                    if agent.const_acc == None:
                        act_agent = agent
                        i_act_agent = i_agent
                        break
                act_agent.plot_color = sc_plot.COLORS['active agent blue']
                act_agent.other_agent.plot_color = sc_plot.COLORS['passive agent grey']
                if i_sim_vars == 0:
                    i_plot_agents = (1-i_act_agent, i_act_agent)
                    agent_alpha = (1, ALPHA)
                else:
                    i_plot_agents = (i_act_agent,)
                    agent_alpha = (ALPHA,)
                sim.do_kinem_states_plot(axs, veh_stop_dec=False, 
                                         agent_alpha=agent_alpha,
                                         i_plot_agents=i_plot_agents,
                                         axis_labels=(i_scenario==0),
                                         plot_fill=False)
                # if i_sim_vars == 20:
                #     break
            axs[1].set_xlim(0, T_MAXS[i_scenario])
            axs[1].set_ylim(V_LIMS[i_scenario][0], V_LIMS[i_scenario][1])
            axs[2].set_ylim(-4, 15)
            axs[2].set_xlabel('Time (s)')
                
            
            
            