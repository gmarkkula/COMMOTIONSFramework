# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 07:22:42 2022

@author: tragma
"""
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# expecting a figures subfolder in the folder where this file is located
SCPAPER_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
FIGS_FOLDER = SCPAPER_PATH + 'figs/'

FULL_WIDTH = 8 # inches
DPI = 300

DEFAULT_FONT_SIZE = 7
PANEL_LABEL_FONT_SIZE = 11
AXIS_LW = 0.5
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':DEFAULT_FONT_SIZE})
plt.rc('axes', linewidth=AXIS_LW)
plt.rc('xtick.major', width=AXIS_LW)
plt.rc('ytick.major', width=AXIS_LW)

BASE_MODELS = ('', 'oVA', 'oVAa', 'oVAaoBEc', 'oVAoVAl', 'oVAaoVAl')  
N_BASE_MODELS = len(BASE_MODELS)
MODEL_VARIANTS = ('', 'oBEo', 'oBEooBEv', 'oBEooBEvoAI', 'oBEv', 'oBEvoAI')
N_MODEL_VARIANTS = len(MODEL_VARIANTS)
MVAR_LINESPECS = ('-', '--', '-.', '-.', ':', ':')
MVAR_COLORS = ('gray', 'orange', 'green', 'green', 'blue', 'blue')
MVAR_LWS = (1, 1, 1, 2, 1, 2)


def get_rgb_tuple(r, g, b):
    return (r/255, g/255, b/255)

def mix_colors(rgb_tuple1, rgb_tuple2, pos=0.5):
    rgb1 = np.array(rgb_tuple1)
    rgb2 = np.array(rgb_tuple2)
    w1 = 1-pos
    w2 = pos
    rgb_mix = w1 * rgb1 + w2 * rgb2 
    return tuple(rgb_mix)
    
def lighten_color(color, light_factor):
    return mix_colors(colors.to_rgb(color), colors.to_rgb('white'), light_factor)
    
    
    

COLORS = {}
COLORS['active agent blue'] = get_rgb_tuple(47, 156, 255)
COLORS['passive agent grey'] = get_rgb_tuple(190, 190, 190)
COLORS['base variant black'] = get_rgb_tuple(0, 8, 20)
COLORS['oBEo variant yellow'] = get_rgb_tuple(255, 195, 0)
COLORS['oBEvoAI variant blue'] = get_rgb_tuple(0, 53, 102)
COLORS['Passing first green'] = get_rgb_tuple(76, 162, 123)
COLORS['Passing second red'] = get_rgb_tuple(120, 0, 37)

DISPLAY_PARAM_NAMES = {}
DISPLAY_PARAM_NAMES['k_c'] = r'$k_\mathrm{c}$'
DISPLAY_PARAM_NAMES['k_sc'] = r'$k_\mathrm{sc}$'
DISPLAY_PARAM_NAMES['T_delta'] = r'$T_\delta$'
DISPLAY_PARAM_NAMES['thetaDot_1'] = r'$\dot{\theta}_1$'
DISPLAY_PARAM_NAMES['beta_V'] = r'$\beta_\mathrm{V}$'
DISPLAY_PARAM_NAMES['T_Of'] = r'$T_\mathrm{Of}$'
DISPLAY_PARAM_NAMES['sigma_O'] = r'$\sigma_\mathrm{O}$'
DISPLAY_PARAM_NAMES['T'] = r'$T$'
DISPLAY_PARAM_NAMES['DeltaV_th_rel'] = r'$\Delta V_\mathrm{th,rel}$'
DISPLAY_PARAM_NAMES['tau_theta'] = r'$\sigma_\mathrm{v}$'


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


def get_display_param_name(param_name):
    if param_name in DISPLAY_PARAM_NAMES.keys():
        return DISPLAY_PARAM_NAMES[param_name]
    else:
        return param_name
    
def get_display_params_str(params_dict):
    params_str = '('
    for i_param, param_name in enumerate(params_dict.keys()):
        if i_param > 0:
            params_str += ', '
        params_str += f'{get_display_param_name(param_name)} = {params_dict[param_name]:.4g}'
    params_str +=')'
    return params_str

def add_panel_label(label, xy):
    plt.annotate(label, xy=xy, xycoords='figure fraction', 
                 fontsize=PANEL_LABEL_FONT_SIZE,
                 fontweight='bold')
    
    
def leave_only_yaxis(ax):
    ax.get_xaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
def add_linked_time_axis(linked_ax, nudge_down=0.02, label='Time (s)'):
    t_ax = linked_ax.get_figure().add_subplot(sharex=linked_ax)
    linked_pos = linked_ax.get_position()
    t_ax_y = linked_pos.y0 - nudge_down
    t_ax.set_position([linked_pos.x0, t_ax_y, linked_pos.width, 0.01])
    t_ax.get_yaxis().set_visible(False)
    t_ax.spines['left'].set_visible(False)
    t_ax.spines['right'].set_visible(False)
    t_ax.spines['top'].set_visible(False)
    t_ax.set_ylabel('__')
    t_ax.set_xlabel(label)
    return t_ax