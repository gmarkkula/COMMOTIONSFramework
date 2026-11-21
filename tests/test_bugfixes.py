import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'SCPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)

import ast
import glob
import inspect
import math
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import commotions
import sc_scenario
import sc_scenario_helper
import sc_scenario_perception
import sc_fitting
import sc_plot
import sc_run

SCPAPER_DIR = os.path.join(REPO, 'SCPaper')
PED = sc_fitting.i_PED_AGENT
VEH = sc_fitting.i_VEH_AGENT
STOCH_MODEL = 'oVAoEAoSNvoPF'
STOCH_PARAMS = {'T': 0.2, 'DeltaV_th_rel': 0.01, 'tau_theta': 0.1}


# ---------------------------------------------------------------------------
# helpers

def _state(signed_CP_dist, long_speed, yaw_angle=0.0):
    state = commotions.KinematicState(pos=None, long_speed=long_speed,
                                      yaw_angle=yaw_angle)
    state.signed_CP_dist = signed_CP_dist
    return state


def _perception_sim(end_time=1, time_step=0.1):
    sim = commotions.Simulation(0, end_time, time_step)
    sim.conflict_point = np.zeros(2)
    return sim


def _const_speed_sc_sim(offset=np.zeros(2), stop_criteria=()):
    """ Two constant-speed agents (no decision making) in a standard SC 
        scenario geometry, translated by offset. """
    initial_pos = np.array(((0., -5.), (20., 0.))) + offset
    goals = np.array(((0., 5.), (-50., 0.))) + offset
    sim = sc_scenario.SCSimulation(
        ctrl_types=sc_fitting.AGENT_CTRL_TYPES,
        widths=sc_fitting.AGENT_WIDTHS, lengths=sc_fitting.AGENT_LENGTHS,
        goal_positions=goals, initial_positions=initial_pos,
        initial_speeds=np.array((1.3, 10.)), const_accs=(0, 0),
        end_time=4, time_step=0.1, stop_criteria=stop_criteria)
    sim.conflict_point = offset
    return sim


# ---------------------------------------------------------------------------
# 1-3: commotions helper functions

def test_quadratic_roots_linear_and_degenerate_cases():
    assert commotions.get_real_quadratic_roots(0, 2, -4) == (2.0,)
    assert commotions.get_real_quadratic_roots(0, 0, 1) == ()
    assert commotions.get_real_quadratic_roots(0, 0, 0) == ()
    # quadratic cases unchanged
    assert sorted(commotions.get_real_quadratic_roots(1, 0, -4)) == [-2.0, 2.0]
    assert commotions.get_real_quadratic_roots(1, 2, 1) == (-1.0,)
    assert commotions.get_real_quadratic_roots(1, 0, 1) == ()


def test_time_to_agent_collision_identical_velocities():
    s1 = commotions.KinematicState(pos=np.array((0., 0.)), long_speed=2,
                                   yaw_angle=0)
    s2 = commotions.KinematicState(pos=np.array((5., 0.)), long_speed=2,
                                   yaw_angle=0)
    assert commotions.get_time_to_agent_collision(s1, s2, 1) == math.inf
    # both stationary
    s1.long_speed = s2.long_speed = 0
    assert commotions.get_time_to_agent_collision(s1, s2, 1) == math.inf
    # but already colliding should still give zero
    assert commotions.get_time_to_agent_collision(s1, s2, 6) == 0


def test_time_to_agent_collision_single_root_in_past_or_future():
    # agent 2 passing agent 1 exactly at the collision distance (touching)
    s1 = commotions.KinematicState(pos=np.array((0., 0.)), long_speed=0,
                                   yaw_angle=0)
    # - touch happened in the past (agent 2 at x = 4, moving in +x, offset 1)
    s2 = commotions.KinematicState(pos=np.array((4., 1.)), long_speed=1,
                                   yaw_angle=0)
    assert commotions.get_time_to_agent_collision(s1, s2, 1) == math.inf
    # - touch in the future (agent 2 at x = -4, moving in +x)
    s2.pos = np.array((-4., 1.))
    assert commotions.get_time_to_agent_collision(s1, s2, 1) == \
        pytest.approx(4)


def test_linear_ramp_truncated_at_array_end():
    arr = np.zeros(5)
    commotions.add_linear_ramp_action_to_array(arr, 1.0, 4, 3)
    assert np.array_equal(arr, np.concatenate(
        (np.zeros(3), np.linspace(0, 1.0, 4)[:2])))
    # ramp starting beyond the end of the array does nothing
    arr = np.zeros(5)
    commotions.add_linear_ramp_action_to_array(arr, 1.0, 4, 7)
    assert np.all(arr == 0)


def test_linear_ramp_unchanged_when_not_truncated():
    arr = np.arange(10, dtype=float)
    expected = np.copy(arr)
    expected[2:6] += np.linspace(0, 0.7, 4)
    expected[6:] += 0.7
    commotions.add_linear_ramp_action_to_array(arr, 0.7, 4, 2)
    assert np.array_equal(arr, expected)


# ---------------------------------------------------------------------------
# 4: mutable default arguments and caller state mutation

def test_no_mutable_default_arguments():
    funcs = (commotions.KinematicState.__init__, 
             commotions.Trajectory.__init__,
             commotions.BaseAgent.__init__, 
             commotions.AgentWithGoal.__init__)
    for func in funcs:
        for param in inspect.signature(func).parameters.values():
            assert not isinstance(param.default, (
                np.ndarray, commotions.KinematicState, 
                commotions.ActionState)), (func, param)


def test_kinematic_state_default_pos_not_shared():
    s1 = commotions.KinematicState()
    s2 = commotions.KinematicState()
    assert np.array_equal(s1.pos, np.zeros(2))
    assert s1.pos is not s2.pos
    s1.pos += 1
    assert np.array_equal(s2.pos, np.zeros(2))
    # explicitly unspecified position is kept as None
    assert commotions.KinematicState(pos=None).pos is None


def test_trajectory_defaults():
    sim = commotions.Simulation(0, 1, 0.1)
    traj = commotions.Trajectory(sim)
    assert np.array_equal(traj.pos[:, 0], np.zeros(2))
    assert traj.long_speed[0] == 0 and traj.yaw_angle[0] == 0
    assert np.all(traj.long_acc == 0) and np.all(traj.yaw_rate == 0)


def test_base_agent_does_not_modify_caller_state():
    sim = commotions.Simulation(0, 1, 0.1)
    state = commotions.KinematicState(pos=None, long_speed=None, 
                                      yaw_angle=None)
    agent = commotions.BaseAgent('A', sim, state)
    assert state.pos is None
    assert state.long_speed is None
    assert state.yaw_angle is None
    assert np.array_equal(agent.trajectory.pos[:, 0], np.zeros(2))
    assert agent.trajectory.long_speed[0] == 0
    assert agent.trajectory.yaw_angle[0] == 0


def test_agent_with_goal_does_not_modify_caller_state():
    sim = commotions.Simulation(0, 1, 0.1)
    pos = np.array((0., -5.))
    state = commotions.KinematicState(pos=pos, long_speed=1, yaw_angle=None)
    agent = commotions.AgentWithGoal('A', sim, np.array((0., 5.)), state)
    assert state.yaw_angle is None
    assert state.pos is pos
    assert agent.trajectory.yaw_angle[0] == pytest.approx(math.pi / 2)
    # the default initial action states are independent zero actions
    agent2 = commotions.AgentWithGoal('B', sim, np.array((5., 0.)), None)
    assert np.all(agent.trajectory.long_acc == 0)
    assert np.all(agent2.trajectory.long_acc == 0)


# ---------------------------------------------------------------------------
# 5: perception aliasing

def test_noise_free_perception_truncation_does_not_affect_true_state():
    perc = sc_scenario_perception.Perception(_perception_sim())
    perc.update(0, _state(10, 1), _state(20, -1.5))
    assert perc.states.x_perceived is not perc.states.x_true
    assert perc.states.x_true[1, 0] == -1.5
    assert perc.states.x_perceived[1, 0] == 0
    assert perc.perc_oth_state.long_speed == 0
    assert perc.states.x_perceived[0, 0] == 20


def test_kalman_perception_truncation_does_not_affect_kalman_state():
    prior = sc_scenario_perception.KalmanPrior(
        cp_dist_mean=20, cp_dist_stddev=1, speed_mean=-2, speed_stddev=0.01)
    perc = sc_scenario_perception.Perception(
        _perception_sim(), pos_obs_noise_stddev=0.5, noise_seed=0,
        kalman_filter=True, prior=prior, spd_proc_noise_stddev=0.01,
        draw_from_estimate=False)
    assert perc.states.x_perceived is not perc.states.x_estimated
    perc.update(0, _state(10, 1), _state(20, 0))
    # the Kalman speed estimate is still negative ...
    assert perc.states.x_estimated[1, 0] < -1
    # ... but the perceived speed is truncated
    assert perc.states.x_perceived[1, 0] == 0
    assert perc.states.x_perceived[0, 0] == perc.states.x_estimated[0, 0]
    # and the next prediction is based on the untruncated estimate
    perc.update(1, _state(10, 1), _state(20, 0))
    assert perc.states.x_estimated[1, 1] < -1


# ---------------------------------------------------------------------------
# 6: early-stop state pollution

def test_early_stopped_simulation_has_no_spurious_last_values():
    scenario = sc_fitting.PROB_FIT_SCENARIOS['PedHesitateVehConst']
    sim_stop = sc_fitting.construct_model_and_simulate_scenario(
        STOCH_MODEL, STOCH_PARAMS, scenario, noise_seeds=(1, 2),
        apply_stop_criteria=True)
    sim_full = sc_fitting.construct_model_and_simulate_scenario(
        STOCH_MODEL, STOCH_PARAMS, scenario, noise_seeds=(1, 2),
        apply_stop_criteria=False)
    i_stop = sim_stop.state.i_time_step
    assert i_stop < sim_stop.settings.n_time_steps - 1, \
        'test assumes that the simulation stops early'
    for agent_stop, agent_full in zip(sim_stop.agents, sim_full.agents):
        for name in ('est_action_vals', 'action_evidence', 
                     'beh_activ_V_given_actions', 'beh_activ_O'):
            arr_stop = getattr(agent_stop.states, name)
            arr_full = getattr(agent_full.states, name)
            assert np.all(np.isnan(arr_stop[..., -1])), name
            # simulated time steps are unaffected
            assert np.array_equal(arr_stop[..., :i_stop+1],
                                  arr_full[..., :i_stop+1], equal_nan=True)
        perc_states = agent_stop.perception.states
        assert np.all(np.isnan(perc_states.x_estimated[:, -1]))
        assert np.all(np.isnan(perc_states.cov_matrix[:, :, -1]))
        assert np.array_equal(
            perc_states.x_estimated[:, :i_stop+1],
            agent_full.perception.states.x_estimated[:, :i_stop+1],
            equal_nan=True)
    # in the full simulation, the last time step was simulated
    assert not np.all(np.isnan(sim_full.agents[1].states.est_action_vals[:, -1]))


def test_perception_finalise():
    prior = sc_scenario_perception.KalmanPrior(20, 1, 1, 1)
    sim = _perception_sim()
    perc = sc_scenario_perception.Perception(
        sim, pos_obs_noise_stddev=0.5, noise_seed=0, kalman_filter=True,
        prior=prior)
    perc.update(0, _state(10, 1), _state(20, 1))
    assert not np.any(np.isnan(perc.states.x_estimated[:, -1]))
    perc.finalise()
    assert np.all(np.isnan(perc.states.x_estimated[:, -1]))
    assert np.all(np.isnan(perc.states.cov_matrix[:, :, -1]))
    assert not np.any(np.isnan(perc.states.x_estimated[:, 0]))
    # no change if the last time step has been reached
    perc = sc_scenario_perception.Perception(
        sim, pos_obs_noise_stddev=0.5, noise_seed=0, kalman_filter=True,
        prior=prior)
    for i in range(sim.settings.n_time_steps):
        perc.update(i, _state(10, 1), _state(20 - i * 0.1, 1))
    last = np.copy(perc.states.x_estimated[:, -1])
    perc.finalise()
    assert np.array_equal(perc.states.x_estimated[:, -1], last)


# ---------------------------------------------------------------------------
# 7: FIX_OTH_ACC_PHASE

def _get_oth_accs():
    params = commotions.Parameters()
    params.DeltaT = 0.5
    params.T_delta = 30
    params.V_ny = 0
    params.thetaDot_0 = 0.001
    params.thetaDot_1 = 0.1
    params.k = commotions.Parameters()
    sc_scenario_helper.set_val_gains_for_free_speed(params.k, 1.5)
    params.k._da = 0.5
    ego_image = sc_scenario_helper.SCAgentImage(
        ctrl_type=sc_scenario_helper.CtrlType.SPEED, params=params,
        v_free=1.5, g_free=sc_scenario_helper.get_agent_free_value_rate(params),
        V_free=sc_scenario_helper.get_agent_free_value(params),
        coll_dist=1.3, eff_width=0.8)
    oth_image = sc_scenario_helper.SCAgentImage(
        ctrl_type=sc_scenario_helper.CtrlType.ACCELERATION, params=None,
        v_free=None, g_free=None, V_free=None, coll_dist=2.5, eff_width=1.8)
    impls = {
        sc_scenario_helper.AccessOrder.EGOFIRST: 
            sc_scenario_helper.AccessOrderImplication(acc=0, T_acc=2, T_dw=0),
        sc_scenario_helper.AccessOrder.EGOSECOND: 
            sc_scenario_helper.AccessOrderImplication(
                acc=math.nan, T_acc=math.nan, T_dw=math.nan)}
    vals = sc_scenario_helper.get_access_order_values(
        ego_image, _state(5, 1.5), action_acc0=0, action_jerk=0,
        oth_image=oth_image, oth_curr_state=_state(40, 10), 
        oth_first_acc=-1, oth_cont_acc=2, access_ord_impls=impls, 
        consider_looming=True, return_details=True)
    details = vals[sc_scenario_helper.AccessOrder.EGOFIRST].details
    return details.oth_accs, details.idx_phase_starts


def test_oth_acc_phase_legacy_default(monkeypatch):
    assert sc_scenario_helper.FIX_OTH_ACC_PHASE is False
    oth_accs, _ = _get_oth_accs()
    assert oth_accs[0] == -1
    assert np.all(oth_accs[1:] == 2)


def test_oth_acc_phase_fixed(monkeypatch):
    monkeypatch.setattr(sc_scenario_helper, 'FIX_OTH_ACC_PHASE', True)
    oth_accs, idx_phase_starts = _get_oth_accs()
    i_end_action = idx_phase_starts[sc_scenario_helper.i_ACTION + 1]
    assert i_end_action > 1
    assert np.all(oth_accs[:i_end_action] == -1)
    assert np.all(oth_accs[i_end_action:] == 2)


# ---------------------------------------------------------------------------
# 8: FIX_FIRST_STEP_SPEED

def _noisy_unfiltered_perception():
    return sc_scenario_perception.Perception(
        _perception_sim(), pos_obs_noise_stddev=0.5, noise_seed=3)


def test_first_step_speed_legacy_default():
    assert sc_scenario_perception.FIX_FIRST_STEP_SPEED is False
    perc = _noisy_unfiltered_perception()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        perc.update(0, _state(10, 1), _state(20, 5))
    assert math.isnan(perc.states.x_noisy[1, 0]) or \
        perc.states.x_noisy[1, 0] == 0
    assert perc.perc_oth_state.long_speed == 0
    assert perc.states.x_perceived[1, 0] == 0


def test_first_step_speed_fixed(monkeypatch):
    monkeypatch.setattr(sc_scenario_perception, 'FIX_FIRST_STEP_SPEED', True)
    perc = _noisy_unfiltered_perception()
    perc.update(0, _state(10, 1), _state(20, 5))
    assert perc.perc_oth_state.long_speed == 5
    assert perc.states.x_perceived[1, 0] == 5
    # later time steps still use finite differences
    perc.update(1, _state(10, 1), _state(19.5, 5))
    expected = -((perc.states.x_noisy[0, 1] - perc.states.x_noisy[0, 0]) 
                 / 0.1)
    assert perc.states.x_perceived[1, 1] == max(0, expected)


# ---------------------------------------------------------------------------
# 9-10: SCSimulation

def test_cs_entry_uses_conflict_point():
    sim_origin = _const_speed_sc_sim()
    sim_origin.run()
    sim_offset = _const_speed_sc_sim(offset=np.array((100., 50.)))
    sim_offset.run()
    for ag_origin, ag_offset in zip(sim_origin.agents, sim_offset.agents):
        assert math.isfinite(ag_origin.ca_entry_time)
        assert ag_offset.ca_entry_time == ag_origin.ca_entry_time
        assert ag_offset.ca_entry_sample == ag_origin.ca_entry_sample
    assert sim_offset.first_passer.name == sim_origin.first_passer.name


def test_do_plots_action_probs_warns_instead_of_crashing():
    import matplotlib.pyplot as plt
    sim = _const_speed_sc_sim()
    sim.run()
    with pytest.warns(UserWarning, match='[Aa]ction probabilities'):
        sim.do_plots(action_probs=True)
    plt.close('all')


# ---------------------------------------------------------------------------
# 11: sc_run.run_simulation

def test_sc_run_standing_ped_distance_and_seeds_passed(monkeypatch):
    captured = {}
    def fake_construct(model_name, params_dict, scenario, **kwargs):
        captured['scenario'] = scenario
        captured['kwargs'] = kwargs
        return 'sim'
    monkeypatch.setattr(sc_fitting, 'construct_model_and_simulate_scenario',
                        fake_construct)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim = sc_run.run_simulation(
            'oVA', initial_cp_distances=(3, 40), 
            initial_speeds=(0, sc_fitting.AGENT_FREE_SPEEDS[VEH]),
            end_time=2, params_dict={}, noise_seeds=(5, 6))
    assert sim == 'sim'
    assert captured['kwargs']['noise_seeds'] == (5, 6)
    init_dists, _ = captured['scenario'].get_dists_and_accs()
    assert init_dists[PED] == pytest.approx(3)
    assert init_dists[VEH] == pytest.approx(40)


def test_sc_run_standing_ped_inside_conflict_space_raises():
    with pytest.raises(ValueError):
        sc_run.run_simulation(
            'oVA', initial_cp_distances=(1, 40), 
            initial_speeds=(0, sc_fitting.AGENT_FREE_SPEEDS[VEH]),
            end_time=2, params_dict={})


def test_sc_run_simulated_initial_distances():
    for ped_speed in (0, sc_fitting.AGENT_FREE_SPEEDS[PED]):
        sim = sc_run.run_simulation(
            'oVA', initial_cp_distances=(3, 40), 
            initial_speeds=(ped_speed, sc_fitting.AGENT_FREE_SPEEDS[VEH]),
            end_time=1, params_dict={})
        assert sim.agents[PED].signed_CP_dists[0] == pytest.approx(3)
        assert sim.agents[VEH].signed_CP_dists[0] == pytest.approx(40)
        assert sim.agents[PED].trajectory.long_speed[0] == ped_speed


def test_sc_run_noise_seeds_reproducible():
    def run(seeds):
        sim = sc_run.run_simulation(
            STOCH_MODEL, initial_cp_distances=(3, 40),
            initial_speeds=sc_fitting.AGENT_FREE_SPEEDS, end_time=2,
            params_dict=STOCH_PARAMS, noise_seeds=seeds)
        return np.concatenate([agent.perception.states.x_noisy[0, :]
                               for agent in sim.agents])
    run_a = run((1, 2))
    run_b = run((1, 2))
    run_c = run((3, 4))
    assert np.array_equal(run_a, run_b, equal_nan=True)
    assert not np.array_equal(run_a, run_c, equal_nan=True)


# ---------------------------------------------------------------------------
# 12: construct_model_and_simulate_scenario(report_time=True)

def test_construct_model_report_time(capsys):
    scenario = sc_fitting.SCPaperScenario('test', initial_ttcas=(3, 3), 
                                          end_time=0.5)
    sim = sc_fitting.construct_model_and_simulate_scenario(
        'oVA', {}, scenario, report_time=True)
    assert isinstance(sim, sc_scenario.SCSimulation)
    assert 'took' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 13-16 (and similar): static checks of the SCPaper scripts

def _scpaper_trees():
    for file_name in sorted(glob.glob(os.path.join(SCPAPER_DIR, '*.py'))):
        with open(file_name, encoding='utf-8') as f:
            source = f.read()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield os.path.basename(file_name), source, ast.parse(source)


def test_scpaper_scripts_compile_without_warnings():
    for file_name in glob.glob(os.path.join(SCPAPER_DIR, '*.py')):
        with open(file_name, encoding='utf-8') as f:
            source = f.read()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compile(source, file_name, 'exec')


def test_scpaper_module_references_exist():
    modules = {'sc_fitting': sc_fitting, 'sc_plot': sc_plot, 
               'sc_scenario': sc_scenario, 'sc_run': sc_run,
               'sc_scenario_helper': sc_scenario_helper,
               'commotions': commotions}
    missing = []
    for file_name, _, tree in _scpaper_trees():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) 
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id in modules
                and not hasattr(modules[node.value.id], node.attr)):
                missing.append(f'{file_name}:{node.lineno}:'
                               f' {node.value.id}.{node.attr}')
    assert missing == []


def test_scpaper_color_keys_exist():
    missing = []
    for file_name, _, tree in _scpaper_trees():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) 
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == 'COLORS'
                and isinstance(node.slice, ast.Constant)
                and node.slice.value not in sc_plot.COLORS):
                missing.append(f'{file_name}:{node.lineno}: {node.slice.value}')
    assert missing == []


def test_scpaper_criterion_names_exist():
    import do_2_analyse_deterministic_fits as do_2
    found = 0
    for file_name, _, tree in _scpaper_trees():
        for node in ast.walk(tree):
            # look for CRITERIA[i_XXX].index('...')
            if (isinstance(node, ast.Call) 
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'index'
                and isinstance(node.func.value, ast.Subscript)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == 'CRITERIA'
                and isinstance(node.func.value.slice, ast.Name)
                and file_name in ('do_2_analyse_deterministic_fits.py',
                                  'do_3_check_short_stopping.py')):
                i_grp = getattr(do_2, node.func.value.slice.id)
                crit_name = node.args[0].value
                assert crit_name in do_2.CRITERIA[i_grp], (file_name, crit_name)
                found += 1
    assert found >= 2


def test_scpaper_no_missing_f_string_prefix_in_exceptions():
    bad = []
    for file_name, _, tree in _scpaper_trees():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                and node.exc.args 
                and isinstance(node.exc.args[0], ast.Constant)
                and isinstance(node.exc.args[0].value, str)
                and re.search(r'\{[A-Za-z_]\w*\}', node.exc.args[0].value)):
                bad.append(f'{file_name}:{node.lineno}')
    assert bad == []


def test_scpaper_no_len_of_comparison():
    # e.g. assert(len(x > 0)), which is always True for non-empty x
    bad = []
    for file_name, _, tree in _scpaper_trees():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'len' and node.args
                and isinstance(node.args[0], ast.Compare)):
                bad.append(f'{file_name}:{node.lineno}')
    assert bad == []


def test_do_2_line_styles_are_tuples():
    _, source, tree = next(t for t in _scpaper_trees() 
                           if t[0] == 'do_2_analyse_deterministic_fits.py')
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) 
            and any(isinstance(t, ast.Name) and t.id == 'act_ls' 
                    for t in node.targets)):
            assert isinstance(node.value, ast.Tuple)
