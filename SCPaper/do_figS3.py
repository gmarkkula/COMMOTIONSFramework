# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:31:16 2022

@author: tragma
"""

import matplotlib.pyplot as plt
import sc_fitting
import sc_plot

SAVE_PDF = True
if SAVE_PDF:
    SCALE_DPI = 1
else:
    SCALE_DPI = 0.5

SCENARIO = sc_fitting.SCPaperScenario(name='startup', initial_ttcas=(100, 100), end_time = 25)
SCENARIO.initial_speeds = (0, 0)
PARAMS_DICT = {'k_c': 0, 'k_sc': 0}
MODELS = ('', 'oVA')

plt.close('all')
fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', 
                        figsize=(0.7*sc_plot.FULL_WIDTH, 0.4*sc_plot.FULL_WIDTH),
                        dpi=sc_plot.DPI * SCALE_DPI)
plt.subplots_adjust(wspace=0.5)

for i_model, model_name in enumerate(MODELS):
    sim = sc_fitting.construct_model_and_simulate_scenario(model_name=model_name, params_dict=PARAMS_DICT, 
                                                           scenario=SCENARIO)
    for i_agent in range(2):
        agent = sim.agents[i_agent]
        
        ax = axs[0, i_agent]
        ax.plot(sim.time_stamps, agent.trajectory.long_acc)
        if i_agent == sc_fitting.i_PED_AGENT:
            ax.set_xlim(-.1, 1.6)
            agent_name = 'Pedestrian'
        else:
            agent_name = 'Driver'
        ax.set_ylabel(agent_name + ' acceleration (m/s²)')
        sc_plot.leave_only_yaxis(ax)
        
        ax = axs[1, i_agent]
        if i_model == 0:
            ax.axhline(sc_fitting.AGENT_FREE_SPEEDS[i_agent], ls='--', c='lightgray')
        ax.plot(sim.time_stamps, agent.trajectory.long_speed)
        ax.set_ylabel(agent_name + ' speed (m/s)')
        sc_plot.leave_only_yaxis(ax)
        sc_plot.add_linked_time_axis(ax)
        
axs[0, 1].legend(('Short-term payoffs', 'Affordance-based'), 
                 title='Value estimation:', frameon=False)

if SAVE_PDF:
    file_name = sc_plot.FIGS_FOLDER + 'figS3.pdf'
    print(f'Saving {file_name}...')
    plt.savefig(file_name, bbox_inches='tight')