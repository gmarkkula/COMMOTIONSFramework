import copy
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'SCPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)

import commotions  # noqa: E402
import sc_scenario  # noqa: E402
import sc_scenario_helper  # noqa: E402
from sc_scenario_helper import AccessOrder, AccessOrderImplication, CtrlType, SCAgentImage  # noqa: E402


def _ref_get_future_kinematic_state(initial_state, long_acc, yaw_rate,
                                    time_step, n_time_steps_to_advance,
                                    i_start_time_step=0, can_reverse=True,
                                    as_tuple=False):
    """Verbatim reference copy of original commotions.get_future_kinematic_state."""
    if type(long_acc) is np.ndarray:
        get_long_acc = lambda i: long_acc[i]  # noqa: E731
    else:
        get_long_acc = lambda i: long_acc  # noqa: E731
    if type(yaw_rate) is np.ndarray:
        get_yaw_rate = lambda i: yaw_rate[i]  # noqa: E731
    else:
        get_yaw_rate = lambda i: yaw_rate  # noqa: E731

    local_state = copy.copy(initial_state)
    for i in range(i_start_time_step + 1,
                   i_start_time_step + n_time_steps_to_advance + 1):
        min_speed = -math.inf if can_reverse else 0
        local_state.pos = (
            local_state.pos
            + max(min_speed, time_step * local_state.long_speed
                  + get_long_acc(i - 1) * (time_step ** 2) / 2)
            * np.array([math.cos(local_state.yaw_angle),
                        math.sin(local_state.yaw_angle)]))
        local_state.long_speed += get_long_acc(i - 1) * time_step
        local_state.long_speed = max(min_speed, local_state.long_speed)
        local_state.yaw_angle += get_yaw_rate(i - 1) * time_step

    if as_tuple:
        return (local_state.pos, local_state.long_speed, local_state.yaw_angle)
    return local_state


def test_get_future_kinematic_state_bit_identical():
    rng = np.random.default_rng(12345)
    for trial in range(200):
        pos0 = rng.normal(0, 20, size=2)
        spd0 = float(rng.uniform(-2.0, 15.0))
        if trial % 10 == 0:
            yaw0 = 0.0
        elif trial % 10 == 1:
            yaw0 = -0.0
        else:
            yaw0 = float(rng.uniform(-math.pi, math.pi))
        st = commotions.KinematicState(pos=pos0.copy(), long_speed=spd0, yaw_angle=yaw0)
        st.extra_tag = ('marker', trial)

        n_adv = int(rng.choice([0, 1, 5, 20]))
        i_start = int(rng.choice([0, 3, 12]))
        total = i_start + n_adv + 5
        long_acc = (rng.normal(0, 3, size=total) if trial % 2 == 0
                    else float(rng.normal(0, 3)))
        if trial % 3 == 0:
            yaw_rate = 0
        elif trial % 3 == 1:
            yaw_rate = float(rng.normal(0, 0.2))
        else:
            yaw_rate = rng.normal(0, 0.2, size=total)
        can_rev = bool(trial % 2)
        as_tup = bool(trial % 5 == 0)
        dt = float(rng.choice([0.025, 0.1]))

        got = commotions.get_future_kinematic_state(
            st, long_acc, yaw_rate, dt, n_adv, i_start, can_rev, as_tup)
        ref = _ref_get_future_kinematic_state(
            st, long_acc, yaw_rate, dt, n_adv, i_start, can_rev, as_tup)

        if as_tup:
            assert np.array_equal(got[0], ref[0])
            assert got[1] == ref[1]
            assert got[2] == ref[2]
        else:
            assert np.array_equal(got.pos, ref.pos)
            assert got.long_speed == ref.long_speed
            assert got.yaw_angle == ref.yaw_angle
            assert got.extra_tag == ('marker', trial)


def test_linear_ramp_cache_bit_identical():
    rng = np.random.default_rng(7)
    for mag in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.345):
        for n_steps in (1, 5, 20):
            base = rng.normal(size=40)
            a1 = base.copy()
            a2 = base.copy()
            commotions.add_linear_ramp_action_to_array(a1, mag, n_steps, 6)
            a2[6:6 + n_steps] += np.linspace(0, mag, n_steps)
            a2[6 + n_steps:] += mag
            assert np.array_equal(a1, a2)
            cached = commotions._LINEAR_RAMP_CACHE[(float(mag), int(n_steps))]
            assert not cached.flags.writeable


def test_horizon_cache_bit_identical():
    dt = sc_scenario_helper.ANTICIPATION_TIME_STEP
    for T_delta in (5.0, 10.0, 30.0, 40.0, 100.0):
        for end_t in (0.5, 1.325, 12.0, 75.0):
            ts_ref = np.arange(0, end_t, dt)
            df_ref = sc_scenario_helper.get_delay_discount(ts_ref, T_delta)
            n = math.ceil(end_t / dt)
            ts_got, df_got = sc_scenario_helper._get_horizon_arrays(T_delta, dt, n)
            assert np.array_equal(ts_got, ts_ref)
            assert np.array_equal(df_got, df_ref)


def test_fast_analytic_horizon_matches_discrete():
    assert sc_scenario_helper.FAST_ANALYTIC_HORIZON is False
    rng = np.random.default_rng(99)

    for ctrl_type in (CtrlType.SPEED, CtrlType.ACCELERATION):
        v_free = 1.5 if ctrl_type is CtrlType.SPEED else 12.0
        params = commotions.Parameters()
        params.DeltaT = 0.5
        params.T_delta = 30.0
        params.V_ny = -15.0
        params.thetaDot_0 = 0.001
        params.thetaDot_1 = 0.1
        params.k = commotions.Parameters()
        sc_scenario_helper.set_val_gains_for_free_speed(params.k, v_free)
        params.k._da = 0.5
        g_free = sc_scenario_helper.get_agent_free_value_rate(params)
        V_free = sc_scenario_helper.get_agent_free_value(params)
        ego_img = SCAgentImage(ctrl_type, params, v_free, g_free, V_free, 1.5, 1.0)
        oth_img = SCAgentImage(CtrlType.SPEED, params, 1.5, g_free, V_free, 1.5, 0.8)

        for _ in range(40):
            ego_st = commotions.KinematicState(
                pos=np.array([20.0, 0.0]),
                long_speed=float(rng.uniform(0.0, v_free * 1.2)),
                yaw_angle=math.pi)
            ego_st.signed_CP_dist = 20.0
            oth_st = commotions.KinematicState(
                pos=np.array([0.0, -5.0]), long_speed=1.2, yaw_angle=math.pi / 2)
            oth_st.signed_CP_dist = 5.0

            acc0 = float(rng.uniform(-3.0, 2.0))
            jerk = 0.0 if ctrl_type is CtrlType.SPEED else float(rng.uniform(-4.0, 4.0))
            T_acc_1 = float(rng.choice([0.0, 0.01, 1.2, 50.0, 250.0]))
            T_dw_2 = float(rng.choice([0.0, 2.3, 200.0]))
            impls = {
                AccessOrder.EGOFIRST: AccessOrderImplication(
                    acc=float(rng.uniform(-2.0, 2.0)), T_acc=T_acc_1, T_dw=0.0),
                AccessOrder.EGOSECOND: AccessOrderImplication(
                    acc=float(rng.uniform(-4.0, 0.5)),
                    T_acc=float(rng.choice([0.5, 3.0])), T_dw=T_dw_2),
            }

            sc_scenario_helper.FAST_ANALYTIC_HORIZON = False
            try:
                disc = sc_scenario_helper.get_access_order_values(
                    ego_img, ego_st, acc0, jerk, oth_img, oth_st,
                    0.0, 0.0, impls, consider_looming=False, return_details=False)
                sc_scenario_helper.FAST_ANALYTIC_HORIZON = True
                ana = sc_scenario_helper.get_access_order_values(
                    ego_img, ego_st, acc0, jerk, oth_img, oth_st,
                    0.0, 0.0, impls, consider_looming=False, return_details=False)
            finally:
                sc_scenario_helper.FAST_ANALYTIC_HORIZON = False

            for ao in AccessOrder:
                assert ana[ao].value == pytest.approx(disc[ao].value, rel=1e-12, abs=1e-12)


def test_nanmax2_semantics():
    pairs = [
        (1.0, 2.0), (2.0, 1.0), (-5.0, -math.inf), (math.nan, -0.5),
        (-0.5, math.nan), (math.nan, math.nan), (0.0, 0.0),
    ]
    for a, b in pairs:
        got = sc_scenario._nanmax2(a, b)
        with np.errstate(all='ignore'):
            ref = float(np.nanmax(np.array([a, b])))
        if math.isnan(ref):
            assert math.isnan(got)
        else:
            assert got == ref


def test_details_return_isolated_from_cache():
    assumptions = sc_scenario.get_assumptions_dict(
        False, oVA=True, oBEv=True, oAI=True)
    sim = sc_scenario.SCSimulation(
        ctrl_types=(CtrlType.SPEED, CtrlType.ACCELERATION),
        widths=(0.8, 1.8), lengths=(0.8, 4.2),
        goal_positions=np.array([[0.0, 5.0], [-50.0, 0.0]]),
        initial_positions=np.array([[0.0, -3.0], [35.0, 0.0]]),
        initial_speeds=np.array([1.3, 12.0]),
        end_time=2.0, time_step=0.1,
        optional_assumptions=assumptions,
        snapshot_times=(None, (0.5,)),
        detailed_snapshots=True)
    sim.run()
    import matplotlib.pyplot as plt
    plt.close('all')
    veh = sim.agents[1]
    deets = veh.snapshot_act_val_details[2, 1][AccessOrder.EGOSECOND].details
    assert deets is not None
    assert deets.time_stamps.flags.writeable
    orig_first = deets.time_stamps[0]
    deets.time_stamps[0] = 999.0
    ts_cached, _ = sc_scenario_helper._get_horizon_arrays(veh.params.T_delta, 0.025, 10)
    assert ts_cached[0] == orig_first
