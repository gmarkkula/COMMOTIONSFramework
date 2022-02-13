# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 07:22:42 2022

@author: tragma
"""
import os
import matplotlib.pyplot as plt

# expecting a figures subfolder in the folder where this file is located
SCPAPER_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
FIGS_FOLDER = SCPAPER_PATH + 'figs/'

FULL_WIDTH = 14 # inches
DPI = 300

plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':8})

BASE_MODELS = ('', 'oVA', 'oVAa', 'oVAaoBEc', 'oVAoVAl', 'oVAaoVAl')  
N_BASE_MODELS = len(BASE_MODELS)
MODEL_VARIANTS = ('', 'oBEo', 'oBEooBEv', 'oBEooBEvoAI', 'oBEv', 'oBEvoAI')
N_MODEL_VARIANTS = len(MODEL_VARIANTS)
MVAR_LINESPECS = ('-', '--', '-.', '-.', ':', ':')
MVAR_COLORS = ('gray', 'orange', 'green', 'green', 'blue', 'blue')
MVAR_LWS = (1, 1, 1, 2, 1, 2)


def get_rgb_tuple(r, g, b):
    return (r/255, g/255, b/255)

COLORS = {}
COLORS['active agent blue'] = get_rgb_tuple(47, 156, 255)
COLORS['passive agent grey'] = get_rgb_tuple(190, 190, 190)
COLORS['other passes second green'] = get_rgb_tuple(76, 162, 123)
COLORS['other passes first red'] = get_rgb_tuple(120, 0, 37)


def split_model_name(full_name):
    for base_name in reversed(BASE_MODELS):
        if base_name == '':
            model_base = ''
            model_variant = full_name
        else:
            if full_name[:len(base_name)] == base_name:
                model_base = base_name
                model_variant = full_name[len(base_name):]
                break
    i_model_base = BASE_MODELS.index(model_base)
    i_model_variant = MODEL_VARIANTS.index(model_variant)
    return i_model_base, i_model_variant