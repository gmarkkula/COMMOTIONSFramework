# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 07:43:03 2022

@author: tragma
"""

# note to self: might want to consider: https://github.com/btel/svg_utils

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage

import sc_fitting
import sc_plot

DPI = sc_plot.DPI / 3

N_COLS = len(sc_fitting.ONE_AG_SCENARIOS)
N_ROWS = 5
TITLES = ('Priority assertion', 'Short-stopping', 'Gap acceptance hesitation',
          'Yield acceptance hesitation', 'Early yield acceptance')

fig, axs = plt.subplots(num=1, nrows = N_ROWS, ncols = N_COLS, 
                        figsize=(sc_plot.FULL_WIDTH, 0.8*sc_plot.FULL_WIDTH), dpi=DPI)

for i_scenario, scenario in enumerate(sc_fitting.ONE_AG_SCENARIOS.keys()):
    ax = axs[0, i_scenario]
    scen_im = plt.imread(sc_plot.FIGS_FOLDER + scenario + '.png')
    ax.imshow(scen_im)
    ax.axis('off')
    ax.set_title(TITLES[i_scenario] + '\n')
    
    
plt.savefig(sc_plot.FIGS_FOLDER + 'fig1.pdf', bbox_inches='tight')