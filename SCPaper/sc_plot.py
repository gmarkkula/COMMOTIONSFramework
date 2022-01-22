# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 07:22:42 2022

@author: tragma
"""


BASE_MODELS = ('', 'oVA', 'oVAa', 'oVAaoBEc', 'oVAoVAl', 'oVAaoVAl')  
N_BASE_MODELS = len(BASE_MODELS)
MODEL_VARIANTS = ('', 'oBEo', 'oBEooBEv', 'oBEooBEvoAI', 'oBEv', 'oBEvoAI')
N_MODEL_VARIANTS = len(MODEL_VARIANTS)
MVAR_LINESPECS = ('-', '--', '-.', '-.', ':', ':')
MVAR_COLORS = ('gray', 'orange', 'green', 'green', 'blue', 'blue')
MVAR_LWS = (1, 1, 1, 2, 1, 2)



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