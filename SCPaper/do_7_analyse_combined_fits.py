# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 07:59:33 2021

@author: tragma
"""
import sc_fitting
import do_4_analyse_probabilistic_fits

do_4_analyse_probabilistic_fits.DO_TIME_SERIES_PLOTS = True
do_4_analyse_probabilistic_fits.DO_PARAMS_PLOTS = True

do_4_analyse_probabilistic_fits.do(sc_fitting.COMB_FIT_FILE_NAME_FMT, 
                                   sc_fitting.RETAINED_COMB_FNAME)