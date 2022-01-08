# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 07:59:33 2021

@author: tragma
"""
import sc_fitting
import do_6_analyse_probabilistic_fits

do_6_analyse_probabilistic_fits.DO_TIME_SERIES_PLOTS = True
do_6_analyse_probabilistic_fits.DO_PARAMS_PLOTS = False
do_6_analyse_probabilistic_fits.N_CRIT_FOR_TS_PLOT = 3

# same analysis as do_6.., just using different input and output files 
do_6_analyse_probabilistic_fits.do(sc_fitting.COMB_FIT_FILE_NAME_FMT, 
                                   sc_fitting.RETAINED_COMB_FNAME)