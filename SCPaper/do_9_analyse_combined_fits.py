# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 07:59:33 2021

@author: tragma
"""
import sc_fitting
import do_6_analyse_probabilistic_fits

do_6_analyse_probabilistic_fits.DO_PLOTS = True
do_6_analyse_probabilistic_fits.DO_TIME_SERIES_PLOTS = False
do_6_analyse_probabilistic_fits.DO_PARAMS_PLOTS = False
do_6_analyse_probabilistic_fits.DO_RETAINED_PARAMS_PLOT = True
do_6_analyse_probabilistic_fits.DO_CRIT_PLOT = True # supplementary figure
do_6_analyse_probabilistic_fits.CRIT_PLOT_MODELS = (
    'oVAoBEvoAIoSNv', 
    'oVAoBEvoAIoEAoSNc', 
    'oVAoBEvoAIoDAoSNc', 
    'oVAoBEvoAIoEAoSNv', 
    'oVAoBEvoAIoDAoSNv', 
    'oVAoBEvoAIoEAoSNvoPF',
    'oVAoBEvoAIoDAoSNvoPF',
    'oVAoBEooBEvoAIoEAoSNvoPF',
    'oVAoBEooBEvoAIoDAoSNvoPF',
    'oVAaoVAloBEvoAIoEAoSNc', 
    'oVAaoVAloBEvoAIoDAoSNc',
    'oVAaoVAloBEvoAIoEAoSNvoPF',
    'oVAaoVAloBEvoAIoDAoSNvoPF')
do_6_analyse_probabilistic_fits.CRIT_PLOT_FIG_NO = 16
do_6_analyse_probabilistic_fits.DO_OUTCOME_PLOT = True
do_6_analyse_probabilistic_fits.SAVE_FIGS = False
do_6_analyse_probabilistic_fits.RET_PARAMS_PLOTS_TO_SAVE = ()

# same analysis as do_6.., just using different input and output files 
comb_fits = do_6_analyse_probabilistic_fits.do(sc_fitting.COMB_FIT_FILE_NAME_FMT, 
                                               sc_fitting.RETAINED_COMB_FNAME,
                                               ylabel_rotation=83)