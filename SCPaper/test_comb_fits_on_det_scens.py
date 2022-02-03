# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 09:25:16 2022

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
import contextlib
import itertools
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import sc_fitting
import sc_plot
import do_2_analyse_deterministic_fits
from do_2_analyse_deterministic_fits import get_max_crit_parameterisations 
from do_2_analyse_deterministic_fits import get_best_scen_var_for_paramet 


OVERWRITE_SAVED_SIM_RESULTS = False
PARALLEL = True

DET_MODEL_NAMES = ('oVAoBEvoAI',)
PROB_MODEL_NAMES = ('oVAoEAoSNvoPF',)
N_PARAMETS_PER_MODEL_AND_SCEN = 50
SCENARIOS = sc_fitting.ONE_AG_SCENARIOS
SIM_RESULTS_FNAME = 'test_CombFitOnDetSimResults.pkl'
ALPHA = 0.1


def run_one_sim(model_name, i_paramet, n_paramets, params_dict, scenario):
    print(f'Simulating {model_name} with parameterisation'
          f' #{i_paramet+1}/{n_paramets}:'
          f' {params_dict} in scenario {scenario.name}...')
    sim_vars = []
    for i_var in range(scenario.n_variations):
        print(f'\tVariation #{i_var+1}/{scenario.n_variations}...')
        sim_vars.append(sc_fitting.construct_model_and_simulate_scenario(
            model_name, params_dict, scenario, i_variation=i_var,
            apply_stop_criteria=False, zero_acc_after_exit=False))
    return sim_vars


def get_ret_models_as_dict(ret_models_list):
    ret_models_dict = {}
    for ret_model in ret_models_list:
        ret_models_dict[ret_model.model] = ret_model
    return ret_models_dict


if __name__ == '__main__':
    
    # run do_2... without any output
    do_2_analyse_deterministic_fits.SAVE_RETAINED_MODELS = False
    do_2_analyse_deterministic_fits.DO_PLOTS = False 
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            det_fits = do_2_analyse_deterministic_fits.do()

    # initialise random number generator
    rng = np.random.default_rng(seed=0)
    
    # load info on retained models - and get as dicts instead
    # - deterministic
    ret_det_models_tmp = sc_fitting.load_results(sc_fitting.RETAINED_DET_FNAME)
    ret_det_models = get_ret_models_as_dict(ret_det_models_tmp)
    del ret_det_models_tmp
    # - probabilistic
    ret_prob_models_tmp = sc_fitting.load_results(sc_fitting.RETAINED_PROB_FNAME)
    ret_prob_models = get_ret_models_as_dict(ret_prob_models_tmp)
    del ret_prob_models_tmp
        
    # get simulation results, by loading existing, or looping through models and simulating
    if sc_fitting.results_exist(SIM_RESULTS_FNAME) and not OVERWRITE_SAVED_SIM_RESULTS:
        sims = sc_fitting.load_results(SIM_RESULTS_FNAME)
    else:
        if PARALLEL:
            print('Starting pool of workers...')
            pool = mp.Pool(mp.cpu_count()-1)
        sims = {}
        for det_model_name in DET_MODEL_NAMES:
            det_model = ret_det_models[det_model_name]
            det_fit = det_fits[det_model_name]
            for prob_model_name in PROB_MODEL_NAMES:
                prob_model = ret_prob_models[prob_model_name]
                
                # get combined model name
                assert(prob_model.model[0:3] == 'oVA')
                model_name = det_model.model + prob_model.model[3:]
                
                # get the combined list of parameterisations
                # - get the deterministic parameterisations from cases with 
                # - all criteria met
                idx_det_all = np.all(det_fit.main_criteria_matrix, axis=0)
                det_params_array = det_fit.results.params_matrix[idx_det_all, :]
                # - get number of parameters and retained parameterisations
                n_det_params = len(det_model.param_names)
                n_prob_params = len(prob_model.param_names)
                n_det_parameterisations = det_params_array.shape[0]
                n_prob_parameterisations = prob_model.params_array.shape[0]
                n_comb_parameterisations = (n_det_parameterisations 
                                            * n_prob_parameterisations)
                # - first construct a big matrix with deterministic parameters to 
                # - the left, repeating each deterministic row of parameters once
                # - for each probabilistic set of parameters
                params_matrix = np.repeat(det_params_array, 
                                          n_prob_parameterisations, axis=0)
                params_matrix = np.append(params_matrix, np.tile(
                    prob_model.params_array, (n_det_parameterisations, 1)), axis=1)
                assert(params_matrix.shape[0] == n_comb_parameterisations)
                # - subsample the matrix of parameterisations if needed
                if n_comb_parameterisations > N_PARAMETS_PER_MODEL_AND_SCEN:
                    idx_included = rng.choice(n_comb_parameterisations, 
                                              size=N_PARAMETS_PER_MODEL_AND_SCEN, 
                                              replace=False)
                    params_matrix = params_matrix[idx_included, :]
                n_act_comb_paramets = params_matrix.shape[0]
                # - generate list of parameter dicts
                param_names = det_model.param_names + prob_model.param_names
                params_dicts = []
                for i_sim in range(n_act_comb_paramets):
                    params_array = params_matrix[i_sim, :]
                    params_dict = dict(zip(param_names, params_array))
                    params_dicts.append(params_dict)
                
                # loop through scenarios and run simulations
                sims[model_name] = {}
                for scenario in SCENARIOS.values():
                    scenario.time_step = sc_fitting.PROB_SIM_TIME_STEP
                    sim_iter = ((model_name, i, n_act_comb_paramets, 
                                 params_dicts[i], scenario) 
                                for i in range(n_act_comb_paramets))
                    if PARALLEL:
                        sims[model_name][scenario.name] = list(
                            pool.starmap(run_one_sim, sim_iter))   
                    else:
                        sims[model_name][scenario.name] = list(
                            itertools.starmap(run_one_sim, sim_iter)) 
                        
        # save simulation results
        sc_fitting.save_results(sims, SIM_RESULTS_FNAME)

# In[1]


    # plot
    PED_V_LIMS = (-.5, 4)
    V_LIMS = ((-1, 15), (-1, 15), PED_V_LIMS, PED_V_LIMS, PED_V_LIMS)
    T_MAXS = (6, 7.5, 9, 9, 9)
    plt.close('all')
    for model_name in sims.keys():
        #n_paramets = len(sims[model_name]['idx_paramets'])
        #n_paramets_total = det_fits[model_name].results.params_matrix.shape[0]
        print(f'\n*** Model "{model_name}" ***')
        det_fit = det_fits[DET_MODEL_NAMES[0]]
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
            axs[1].set_title(scenario_name)
            axs[1].set_xlim(0, T_MAXS[i_scenario])
            axs[1].set_ylim(V_LIMS[i_scenario][0], V_LIMS[i_scenario][1])
            axs[2].set_ylim(-4, 15)
            axs[2].set_xlabel('Time (s)')
        
        plt.show()