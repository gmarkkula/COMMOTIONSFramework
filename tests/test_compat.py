# -*- coding: utf-8 -*-
"""
Compatibility tests: check that the code base imports and runs on current
versions of Python / NumPy / pandas / matplotlib, and that the stored results
and data pickles (created with older library versions) can still be loaded.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'SCPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)

import glob
import importlib
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

SCPAPER_DIR = os.path.join(REPO, 'SCPaper')
RESULTS_PKLS = sorted(glob.glob(os.path.join(SCPAPER_DIR, 'results', '*.pkl')))
DATA_PKLS = sorted(glob.glob(os.path.join(SCPAPER_DIR, 'data', '*.pkl')))
PY_FILES = sorted(glob.glob(os.path.join(REPO, '*.py'))
                  + glob.glob(os.path.join(SCPAPER_DIR, '*.py')))


# --- imports

@pytest.mark.parametrize('module_name', ('commotions', 'sc_scenario',
                                         'sc_scenario_helper',
                                         'sc_scenario_perception'))
def test_import_root_modules(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize('module_name', ('sc_fitting', 'sc_run',
                                         'parameter_search', 'sc_plot'))
def test_import_scpaper_modules(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize('file_path', PY_FILES,
                         ids=[os.path.basename(f) for f in PY_FILES])
def test_no_syntax_warnings(file_path):
    # e.g., invalid escape sequences, which are SyntaxWarnings in Python 3.12+
    with open(file_path, encoding='utf-8') as file_obj:
        source = file_obj.read()
    with warnings.catch_warnings():
        warnings.simplefilter('error', SyntaxWarning)
        compile(source, file_path, 'exec')


# --- loading of stored pickles

def test_found_pickles():
    assert len(RESULTS_PKLS) >= 170
    assert len(DATA_PKLS) == 1


@pytest.mark.parametrize('file_path', RESULTS_PKLS,
                         ids=[os.path.basename(f) for f in RESULTS_PKLS])
def test_load_results_pickle(file_path):
    import sc_fitting
    results = sc_fitting.load_results(os.path.basename(file_path), 
                                      verbose=False)
    assert results is not None


def test_load_legacy_pandas_cits():
    import sc_fitting
    cit_files = glob.glob(os.path.join(
        SCPAPER_DIR, 'results', sc_fitting.MODEL_CIT_FNAME_FMT % '*'))
    assert len(cit_files) > 0
    for cit_file in cit_files:
        cits = sc_fitting.load_results(os.path.basename(cit_file), 
                                       verbose=False)
        assert isinstance(cits, dict)
        for df in cits.values():
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ['i_param', 'crossing_time']


def test_load_hesitation_checks():
    import sc_fitting
    hes_checks = sc_fitting.load_results(sc_fitting.HESITATION_CHECK_FNAME,
                                         verbose=False)
    assert isinstance(hes_checks, dict) and len(hes_checks) > 0
    for model_checks in hes_checks.values():
        for hes_check in model_checks:
            assert isinstance(hes_check, sc_fitting.HesitationCheck)
            assert isinstance(hes_check.params_dict, dict)


def test_load_hiker_data():
    import sc_fitting
    hiker_data = sc_fitting.load_hiker_data()
    assert isinstance(hiker_data, dict) and len(hiker_data) > 0
    for df in hiker_data.values():
        assert isinstance(df, pd.DataFrame)
        assert 'crossing_time' in df.columns


def test_hiker_cit_cdf_plot_no_warnings():
    import sc_fitting
    hiker_data = sc_fitting.load_hiker_data()
    with warnings.catch_warnings():
        # e.g., matplotlib "color is redundantly defined" warnings
        warnings.simplefilter('error', UserWarning)
        sc_fitting.do_hiker_cit_cdf_plot(hiker_data, fig_name='test_compat',
                                         finalise=False)
    plt.close('all')


# --- parameter search

def test_tiny_grid_search():
    import parameter_search

    class TinyParameterSearch(parameter_search.ParameterSearch):
        def get_metrics_for_params(self, params):
            return {'m1': params['p1'] + params['p2'],
                    'm2': params['p1'] * params['p2']}

    search = TinyParameterSearch(param_names=('p1', 'p2'),
                                 metric_names=('m1', 'm2'), name='tiny',
                                 verbosity=0)
    with warnings.catch_warnings():
        # e.g., "RuntimeWarning: invalid value encountered in cast" from 
        # NaN-initialising an integer array
        warnings.simplefilter('error')
        search.search_grid({'p1': (0, 1, 2), 'p2': np.array((10., 20.))})
    params_matrix = search.results.params_matrix
    assert params_matrix.shape == (6, 2)
    # all combinations present
    assert (set(map(tuple, params_matrix)) 
            == {(p1, p2) for p1 in (0, 1, 2) for p2 in (10., 20.)})
    metrics_matrix = search.results.metrics_matrix
    assert np.array_equal(metrics_matrix[:, 0], params_matrix.sum(axis=1))
    assert np.array_equal(metrics_matrix[:, 1], params_matrix.prod(axis=1))


# --- simulation

def test_sc_run_simulation():
    import sc_fitting
    import sc_run
    model = sc_run.get_model_with_params('oVAoBEvoAI')
    assert isinstance(model, sc_fitting.ModelWithParams)
    assert model.n_parameterisations > 0
    sim = sc_run.run_simulation(model, initial_cp_distances=(3, 40),
                                initial_speeds=sc_fitting.AGENT_FREE_SPEEDS,
                                end_time=3, idx_parameterisation=0)
    assert len(sim.agents) == 2
    for agent in sim.agents:
        assert agent.trajectory.pos.shape[1] == len(sim.time_stamps)
        assert np.all(np.isfinite(agent.trajectory.pos))
        assert np.all(np.isfinite(agent.trajectory.long_speed))


# --- do_12 CIT reorganisation (previously using DataFrame.append)

def test_do_12_get_model_cits():
    import parameter_search
    import sc_fitting
    do_12 = importlib.import_module('do_12_analyse_HIKER_fits')
    model_name = 'oVAoBEvoAIoDAoSNvoPF'
    hiker_fit = parameter_search.load(
        sc_fitting.FIT_RESULTS_FOLDER 
        + sc_fitting.HIKER_FIT_FILE_NAME_FMT % model_name)
    # same parameterisation selection as in do_12
    idx_yielding_cits = [i for i, name in enumerate(hiker_fit.metric_names)
                         if 'y' in name]
    yielding_cits = hiker_fit.results.metrics_matrix[:, idx_yielding_cits, :]
    noncross_yield_params = np.any(np.isnan(yielding_cits), axis=(1, 2))
    i_included_params = np.nonzero(~noncross_yield_params)[0]
    assert len(i_included_params) > 0
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        model_cits = do_12.get_model_cits(hiker_fit, i_included_params)
    # compare to the stored output from the original (pandas 1.x) run 
    legacy_cits = sc_fitting.load_results(
        sc_fitting.MODEL_CIT_FNAME_FMT % model_name, verbose=False)
    assert set(model_cits.keys()) == set(legacy_cits.keys())
    n_rows = len(i_included_params) * hiker_fit.n_repetitions
    for scen_name, df in model_cits.items():
        legacy_df = legacy_cits[scen_name]
        assert list(df.columns) == ['i_param', 'crossing_time']
        assert np.issubdtype(df['i_param'].dtype, np.integer)
        assert np.issubdtype(df['crossing_time'].dtype, np.floating)
        assert len(df) == n_rows
        assert np.array_equal(df.index, legacy_df.index)
        assert np.array_equal(df['i_param'].values, 
                              legacy_df['i_param'].astype(int).values)
        assert np.array_equal(df['crossing_time'].values, 
                              legacy_df['crossing_time'].astype(float).values,
                              equal_nan=True)
    # no included parameterisations should give empty DataFrames
    empty_cits = do_12.get_model_cits(hiker_fit, [])
    for df in empty_cits.values():
        assert len(df) == 0
        assert list(df.columns) == ['i_param', 'crossing_time']
