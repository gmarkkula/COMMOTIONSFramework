import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'SCPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)

import math

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import sc_scenario
from sc_scenario import CtrlType, SimStopCriterion
import sc_analysis


TIME_STEP = 0.1
PED_COLL_DIST = 0.8 / 2 + 1.8 / 2 # ped length / 2 + veh width / 2 = 1.3 m
VEH_COLL_DIST = 4.2 / 2 + 0.8 / 2 # veh length / 2 + ped width / 2 = 2.5 m


def make_const_acc_sim(ped_d0, ped_v0, veh_d0, veh_v0, ped_acc=0, veh_acc=0,
                       end_time=6, stop_criteria=()):
    """ Run a simulation with a pedestrian (agent 0, 'P') and a vehicle 
        (agent 1, 'V') with constant accelerations, in the same geometry as
        in SCPaper/sc_fitting.py.
    """
    sim = sc_scenario.SCSimulation(
        ctrl_types=(CtrlType.SPEED, CtrlType.ACCELERATION),
        widths=(0.8, 1.8), lengths=(0.8, 4.2),
        goal_positions=np.array([[0, 5], [-50, 0]]),
        initial_positions=np.array([[0, -ped_d0], [veh_d0, 0]]),
        initial_speeds=np.array((ped_v0, veh_v0)),
        const_accs=(ped_acc, veh_acc), start_time=0, end_time=end_time,
        time_step=TIME_STEP, agent_names=('P', 'V'),
        stop_criteria=stop_criteria)
    sim.run()
    return sim


# vehicle at 10 m/s from 20.55 m: in CS for t in (1.805, 2.305) s, 
# i.e., sampled entry at 1.9 s and exit at 2.4 s
VEH_D0 = 20.55
VEH_V0 = 10
VEH_ENTRY = (VEH_D0 - VEH_COLL_DIST) / VEH_V0
VEH_EXIT = (VEH_D0 + VEH_COLL_DIST) / VEH_V0
# pedestrian at 1.5 m/s
PED_V0 = 1.5


@pytest.fixture(scope='module')
def no_coll_sim():
    # pedestrian from 6 m: in CS for t in (3.133, 4.867) s, i.e., sampled 
    # entry at 3.2 s and exit at 4.9 s
    return make_const_acc_sim(6, PED_V0, VEH_D0, VEH_V0)


@pytest.fixture(scope='module')
def coll_sim():
    # pedestrian from 4.2 m: in CS for t in (1.933, 3.667) s, i.e., sampled 
    # entry at 2.0 s and exit at 3.7 s - overlapping with vehicle
    return make_const_acc_sim(4.2, PED_V0, VEH_D0, VEH_V0)


def test_entry_exit_times(no_coll_sim):
    assert sc_analysis.get_cs_entry_exit_times(no_coll_sim, 0) \
        == pytest.approx((3.2, 4.9))
    assert sc_analysis.get_cs_entry_exit_times(no_coll_sim, 1) \
        == pytest.approx((1.9, 2.4))
    all_times = sc_analysis.get_cs_entry_exit_times(no_coll_sim)
    assert len(all_times) == 2
    assert all_times[0] == pytest.approx((3.2, 4.9))
    assert all_times[1] == pytest.approx((1.9, 2.4))
    # within one time step of the analytical values
    ped_entry = (6 - PED_COLL_DIST) / PED_V0
    ped_exit = (6 + PED_COLL_DIST) / PED_V0
    analytical = ((ped_entry, ped_exit), (VEH_ENTRY, VEH_EXIT))
    for i_agent in range(2):
        times = sc_analysis.get_cs_entry_exit_times(no_coll_sim, i_agent)
        for time, analytical_time in zip(times, analytical[i_agent]):
            assert 0 <= time - analytical_time < TIME_STEP


def test_lazy_pandas_import():
    import subprocess
    code = (
        "import sys, numpy as np, sc_scenario, sc_analysis; "
        "assert 'pandas' not in sys.modules; "
        "sim = sc_scenario.SCSimulation("
        "ctrl_types=(sc_scenario.CtrlType.SPEED, sc_scenario.CtrlType.ACCELERATION), "
        "widths=(0.8, 1.8), lengths=(0.8, 4.2), "
        "goal_positions=np.array([[0, 5], [-50, 0]]), "
        "initial_positions=np.array([[0, -6.0], [20.55, 0]]), "
        "initial_speeds=np.array((1.5, 10.0)), const_accs=(0, 0), "
        "start_time=0, end_time=1, time_step=0.1, agent_names=('P', 'V')); "
        "sim.run(); "
        "sim.summarise_interaction(); "
        "assert 'pandas' not in sys.modules; "
        "sim.to_dataframe(); "
        "assert 'pandas' in sys.modules"
    )
    subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True)


def test_pet_no_collision(no_coll_sim):
    pet, i_first = sc_analysis.get_pet(no_coll_sim, return_order=True)
    assert i_first == 1
    assert sc_analysis.get_first_agent_in_cs(no_coll_sim) == 1
    assert pet == pytest.approx(3.2 - 2.4)
    analytical_pet = (6 - PED_COLL_DIST) / PED_V0 - VEH_EXIT
    assert abs(pet - analytical_pet) < TIME_STEP
    assert no_coll_sim.get_pet() == pet
    assert not sc_analysis.had_collision(no_coll_sim)


def test_ttc_no_collision_course(no_coll_sim):
    ttcs = sc_analysis.get_ttc_series(no_coll_sim)
    assert ttcs.shape == no_coll_sim.time_stamps.shape
    assert np.all(np.isposinf(ttcs))
    assert sc_analysis.get_min_ttc(no_coll_sim) == math.inf


def test_min_distance_margin(no_coll_sim, coll_sim):
    # sampled minimum at t = 2.3 s: vehicle still in CS (margin 0), 
    # pedestrian at 6 - 2.3 * 1.5 = 2.55 m, i.e., margin 1.25 m
    assert sc_analysis.get_min_distance_margin(no_coll_sim) \
        == pytest.approx(1.25)
    assert sc_analysis.get_min_distance_margin(coll_sim) == 0


def test_collision(coll_sim):
    assert sc_analysis.had_collision(coll_sim)
    assert sc_analysis.get_pet(coll_sim) == 0
    assert sc_analysis.get_pet(coll_sim, return_order=True) == (0, 1)
    assert sc_analysis.get_min_ttc(coll_sim) == 0
    
    
def test_ttc_collision_course(coll_sim):
    ttcs = coll_sim.get_ttc_series()
    time_stamps = coll_sim.time_stamps
    ped_entry = (4.2 - PED_COLL_DIST) / PED_V0
    # before collision: time left until pedestrian enters the CS, while the 
    # vehicle is (projected to be) in it
    before = time_stamps < ped_entry
    assert np.allclose(ttcs[before], ped_entry - time_stamps[before], 
                       rtol=0, atol=1e-9)
    # during collision (both agents in CS)
    during = (time_stamps > ped_entry) & (time_stamps < VEH_EXIT)
    assert np.sum(during) == 4 # 2.0, 2.1, 2.2, 2.3 s
    assert np.all(ttcs[during] == 0)
    # after the vehicle has left the CS: no longer on collision course
    after = time_stamps > VEH_EXIT
    assert np.all(np.isposinf(ttcs[after]))
    

def test_ttc_consider_acc():
    # vehicle decelerating at 4 m/s^2 will stop 8.05 m before conflict point,
    # i.e., outside the CS
    sim = make_const_acc_sim(4.2, PED_V0, VEH_D0, VEH_V0, veh_acc=-4)
    ttcs_const_speed = sc_analysis.get_ttc_series(sim, consider_acc=False)
    ttcs_acc = sc_analysis.get_ttc_series(sim, consider_acc=True)
    assert ttcs_const_speed[0] == pytest.approx(
        (4.2 - PED_COLL_DIST) / PED_V0)
    assert np.all(np.isposinf(ttcs_acc))
    assert sc_analysis.get_min_ttc(sim, consider_acc=True) == math.inf
    assert sc_analysis.get_min_ttc(sim) < ttcs_const_speed[0]
    assert not sc_analysis.had_collision(sim)
    # vehicle never entered, so PET undefined, pedestrian first
    veh_entry, veh_exit = sc_analysis.get_cs_entry_exit_times(sim, 1)
    assert math.isnan(veh_entry) and math.isnan(veh_exit)
    pet, i_first = sc_analysis.get_pet(sim, return_order=True)
    assert math.isnan(pet)
    assert i_first == 0
    summary = sim.summarise_interaction()
    assert summary['first_passer'] == 'P'
    assert math.isnan(summary['pet'])
    assert summary['collision'] is False


def test_pet_undefined_if_no_entry():
    # both agents standing still, far from the CS
    sim = make_const_acc_sim(6, 0, VEH_D0, 1e-9, end_time=1)
    assert sc_analysis.get_pet(sim, return_order=True)[1] is None
    assert math.isnan(sc_analysis.get_pet(sim))
    assert sim.summarise_interaction()['first_passer'] is None


def test_summarise_interaction(no_coll_sim):
    summary = sc_analysis.summarise_interaction(no_coll_sim)
    assert summary['first_passer'] == 'V'
    assert summary['pet'] == pytest.approx(0.8)
    assert summary['min_ttc'] == math.inf
    assert summary['min_distance_margin'] == pytest.approx(1.25)
    assert summary['collision'] is False
    assert summary['P_entry_time'] == pytest.approx(3.2)
    assert summary['P_exit_time'] == pytest.approx(4.9)
    assert summary['V_entry_time'] == pytest.approx(1.9)
    assert summary['V_exit_time'] == pytest.approx(2.4)
    assert summary['actual_end_time'] == pytest.approx(5.9)


EXPECTED_COLUMNS = ['time']
for name in ('P', 'V'):
    EXPECTED_COLUMNS += [f'{name}_x', f'{name}_y', f'{name}_speed', 
                         f'{name}_acc', f'{name}_cp_dist', 
                         f'{name}_perc_oth_cp_dist', f'{name}_perc_oth_speed']
EXPECTED_COLUMNS += ['ttc']


def test_dataframe(coll_sim):
    df = coll_sim.to_dataframe()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == len(coll_sim.time_stamps)
    assert np.array_equal(df['time'], coll_sim.time_stamps)
    assert np.array_equal(df['V_cp_dist'], coll_sim.agents[1].signed_CP_dists)
    assert np.array_equal(df['P_y'], coll_sim.agents[0].trajectory.pos[1, :])
    assert np.array_equal(df['ttc'], coll_sim.get_ttc_series())
    df_no_perc = sc_analysis.sim_to_dataframe(coll_sim, 
                                              include_perception=False)
    assert list(df_no_perc.columns) == [col for col in EXPECTED_COLUMNS 
                                        if 'perc' not in col]
    

def test_dataframe_excludes_nan_tail():
    # stop the simulation when the vehicle enters the CS at 1.9 s
    sim = make_const_acc_sim(6, PED_V0, VEH_D0, VEH_V0, 
                             stop_criteria=(SimStopCriterion.AGENT_IN_CS,))
    assert sim.actual_end_time == pytest.approx(1.9)
    assert sc_analysis.get_n_simulated_steps(sim) == 20
    df = sim.to_dataframe()
    assert df.shape == (20, len(EXPECTED_COLUMNS))
    assert df['time'].iloc[-1] == pytest.approx(1.9)
    kinem_cols = [col for col in df.columns if 'perc' not in col]
    assert not df[kinem_cols].isna().any().any()
    ttcs = sim.get_ttc_series()
    assert len(ttcs) == len(sim.time_stamps)
    assert not np.any(np.isnan(ttcs[:20]))
    assert np.all(np.isnan(ttcs[20:]))
    # entry/exit only within the simulated period
    assert sc_analysis.get_cs_entry_exit_times(sim, 1)[0] \
        == pytest.approx(1.9)
    assert math.isnan(sc_analysis.get_cs_entry_exit_times(sim, 1)[1])
    assert all(math.isnan(t) for t in 
               sc_analysis.get_cs_entry_exit_times(sim, 0))
    assert math.isnan(sim.get_pet())


def test_csv_round_trip(coll_sim, tmp_path):
    path = tmp_path / 'sim.csv'
    df = coll_sim.to_csv(path)
    df_read = pd.read_csv(path)
    assert list(df_read.columns) == EXPECTED_COLUMNS
    pd.testing.assert_frame_equal(df, df_read, check_exact=False, rtol=1e-12)
    assert np.isposinf(df_read['ttc']).any() and (df_read['ttc'] == 0).any()
    path2 = tmp_path / 'sim2.csv'
    sc_analysis.export_csv(coll_sim, path2, include_perception=False, 
                           float_format='%.6f')
    df_read2 = pd.read_csv(path2)
    assert 'P_perc_oth_speed' not in df_read2.columns
    assert np.allclose(df_read2['V_x'], df['V_x'], atol=1e-6)


# --- population runner

STOCH_MODEL = 'oVAoBEvoAIoEAoSNvoPF'
POP_KWARGS = dict(initial_cp_distances=(3, 40), initial_speeds=(1.3, 50 / 3.6),
                  end_time=2, ped_prio=True)


@pytest.fixture(scope='module')
def stoch_model():
    import sc_run
    return sc_run.get_model_with_params(STOCH_MODEL)


def assert_sims_equal(sim1, sim2):
    assert np.array_equal(sim1.time_stamps, sim2.time_stamps)
    assert sim1.actual_end_time == sim2.actual_end_time
    for agent1, agent2 in zip(sim1.agents, sim2.agents):
        for arr1, arr2 in (
                (agent1.trajectory.pos, agent2.trajectory.pos),
                (agent1.trajectory.long_speed, agent2.trajectory.long_speed),
                (agent1.trajectory.long_acc, agent2.trajectory.long_acc),
                (agent1.signed_CP_dists, agent2.signed_CP_dists),
                (agent1.perception.states.x_perceived, 
                 agent2.perception.states.x_perceived),
                (agent1.states.est_action_vals, agent2.states.est_action_vals)):
            assert np.array_equal(arr1, arr2, equal_nan=True)


def sims_differ(sim1, sim2):
    return not np.array_equal(sim1.agents[0].perception.states.x_perceived,
                              sim2.agents[0].perception.states.x_perceived,
                              equal_nan=True)


def test_population_serial_vs_parallel(stoch_model):
    import sc_run
    sims_serial = sc_run.run_population(stoch_model, n_parameterisations=3, 
                                        base_seed=42, parallel=False, 
                                        **POP_KWARGS)
    sims_parallel = sc_run.run_population(stoch_model, n_parameterisations=3, 
                                          base_seed=42, parallel=True, 
                                          n_workers=2, **POP_KWARGS)
    assert len(sims_serial) == len(sims_parallel) == 3
    # same random subsample as drawn with default_rng(0), in drawn order
    expected_idxs = [int(i) for i in np.random.default_rng(0).choice(
        stoch_model.n_parameterisations, 3, replace=False)]
    assert [sim.idx_parameterisation for sim in sims_serial] == expected_idxs
    assert [sim.idx_parameterisation for sim in sims_parallel] == expected_idxs
    for sim_serial, sim_parallel in zip(sims_serial, sims_parallel):
        assert sim_serial.noise_seeds == sim_parallel.noise_seeds
        assert_sims_equal(sim_serial, sim_parallel)
    # results identical to run_simulation() with the same seeds
    sim = sc_run.run_simulation(stoch_model, 
                                idx_parameterisation=expected_idxs[1],
                                noise_seeds=sims_serial[1].noise_seeds,
                                **POP_KWARGS)
    assert_sims_equal(sim, sims_serial[1])
    # summary DataFrame
    df = sc_run.summarise_population(sims_parallel)
    assert len(df) == 3
    assert list(df['idx_parameterisation']) == expected_idxs
    for col in ['first_passer', 'pet', 'min_ttc', 'min_distance_margin', 
                'collision', 'P_entry_time', 'P_exit_time', 'V_entry_time', 
                'V_exit_time', 'actual_end_time'] + list(stoch_model.param_names):
        assert col in df.columns
    for i_sim, idx in enumerate(expected_idxs):
        assert np.array_equal(
            df.loc[i_sim, list(stoch_model.param_names)].to_numpy(dtype=float),
            stoch_model.params_array[idx, :])
    # internal states in DataFrame for a model-driven simulation
    df_sim = sims_serial[0].to_dataframe(include_internal=True)
    assert 'P_est_val_a+0' in df_sim.columns
    assert 'V_beh_prob_pass1st_given_a-1' in df_sim.columns
    assert df_sim.shape[1] == len(EXPECTED_COLUMNS) + 2 * (5 + 3 * 5)
    assert not df_sim['P_est_val_a+0'].isna().all()


def test_population_seeding(stoch_model):
    import sc_run
    idxs = [5, 5, 7]
    sims_a = sc_run.run_population(stoch_model, idx_parameterisations=idxs,
                                   base_seed=1, parallel=False, **POP_KWARGS)
    sims_b = sc_run.run_population(stoch_model, idx_parameterisations=idxs,
                                   base_seed=1, parallel=False, **POP_KWARGS)
    sims_c = sc_run.run_population(stoch_model, idx_parameterisations=idxs,
                                   base_seed=2, parallel=False, **POP_KWARGS)
    assert [sim.idx_parameterisation for sim in sims_a] == idxs
    for sim_a, sim_b, sim_c in zip(sims_a, sims_b, sims_c):
        assert_sims_equal(sim_a, sim_b)
        assert sims_differ(sim_a, sim_c)
    # repeated parameterisation gets different noise
    assert sims_a[0].noise_seeds != sims_a[1].noise_seeds
    assert sims_differ(sims_a[0], sims_a[1])
    # invalid inputs
    with pytest.raises(Exception):
        sc_run.run_population(stoch_model, n_parameterisations=2, 
                              idx_parameterisations=[0], **POP_KWARGS)
    with pytest.raises(Exception):
        sc_run.run_population(STOCH_MODEL, **POP_KWARGS)
