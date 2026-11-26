import math
import warnings
import numpy as np
import pytest

import commotions
from commotions import (
    ActionState,
    AgentWithGoal,
    AgentWithIntermittentActions,
    AgentWithIntermittentGaze,
    BaseAgent,
    GazeState,
    KinematicState,
    ON_PHONE,
    ON_ROAD,
    Simulation,
    get_real_quadratic_roots,
    get_time_to_agent_collision,
)
from sc_scenario_perception import (
    GazeGatedLoomingPerception,
    KalmanPrior,
    Perception,
)


class TestZeroRelativeVelocityBugfix:
    """Tests for the a == 0 RuntimeWarning / ZeroDivisionError bugfix."""

    def test_quadratic_roots_a_zero_cases(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Constant non-zero equation: 0*t^2 + 0*t + 9 = 0 -> no real roots
            assert get_real_quadratic_roots(0.0, 0.0, 9.0) == ()
            assert get_real_quadratic_roots(
                np.float64(0.0), np.float64(0.0), np.float64(9.0)
            ) == ()
            # Trivial identity at origin: 0*t^2 + 0*t + 0 = 0
            assert get_real_quadratic_roots(0.0, 0.0, 0.0) == (0.0,)
            # Linear equation: 0*t^2 + 4*t - 10 = 0 -> t = 2.5
            roots = get_real_quadratic_roots(0.0, 4.0, -10.0)
            assert len(roots) == 1
            assert math.isclose(roots[0], 2.5)

    def test_time_to_collision_identical_velocity_vectors_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s1 = KinematicState(
                pos=np.array([0.0, 0.0], dtype=np.float64),
                long_speed=np.float64(1.4),
                yaw_angle=np.float64(0.3),
            )
            s2 = KinematicState(
                pos=np.array([5.0, 2.0], dtype=np.float64),
                long_speed=np.float64(1.4),
                yaw_angle=np.float64(0.3),
            )
            ttc = get_time_to_agent_collision(s1, s2, collision_distance=1.0)
            assert ttc == math.inf

            # Already overlapping with identical velocities -> immediate collision (0)
            s3 = KinematicState(
                pos=np.array([0.4, 0.0], dtype=np.float64),
                long_speed=np.float64(1.4),
                yaw_angle=np.float64(0.3),
            )
            assert get_time_to_agent_collision(s1, s3, collision_distance=1.0) == 0

    def test_time_to_collision_head_on_and_diverging(self):
        s1 = KinematicState(pos=np.array([0.0, 0.0]), long_speed=2.0, yaw_angle=0.0)
        s2 = KinematicState(
            pos=np.array([10.0, 0.0]), long_speed=3.0, yaw_angle=math.pi
        )
        # Distance 10 m, closing speed 5 m/s, collision radius 1.0 m -> (10 - 1) / 5 = 1.8 s
        ttc = get_time_to_agent_collision(s1, s2, collision_distance=1.0)
        assert math.isclose(ttc, 1.8, rel_tol=1e-9)

        # Diverging agents -> inf
        s2_div = KinematicState(
            pos=np.array([10.0, 0.0]), long_speed=3.0, yaw_angle=0.0
        )
        assert get_time_to_agent_collision(s1, s2_div, collision_distance=1.0) == math.inf


class TestCooperativeIntermittentGazeAgents:
    """Tests for cooperative multiple inheritance and Markov gaze dynamics."""

    def test_cooperative_mro_and_single_registration(self):
        mro = AgentWithIntermittentGaze.__mro__
        assert mro.index(AgentWithGoal) < mro.index(BaseAgent)
        assert mro.index(AgentWithIntermittentActions) < mro.index(BaseAgent)

        sim = Simulation(start_time=0.0, end_time=4.0, time_step=0.05)
        ped = AgentWithIntermittentGaze(
            name="ped_coop",
            simulation=sim,
            goal=np.array([0.0, 6.0]),
            initial_kinematic_state=KinematicState(
                pos=np.array([0.0, -4.0]), long_speed=1.2, yaw_angle=None
            ),
            action_interval=0.25,
            initial_gaze_state=ON_ROAD,
            rng_seed=42,
        )
        # BaseAgent.__init__ must execute exactly once via cooperative super()
        assert len(sim.agents) == 1
        assert sim.agents[0] is ped
        assert math.isclose(ped.trajectory.yaw_angle[0], math.pi / 2.0, abs_tol=1e-9)

        # Reverse inheritance order also cooperates cleanly
        class ReverseCooperativeAgent(AgentWithIntermittentActions, AgentWithGoal):
            pass

        rev = ReverseCooperativeAgent(
            name="rev_coop",
            simulation=sim,
            goal=np.array([5.0, 0.0]),
            initial_state=KinematicState(
                pos=np.array([0.0, 0.0]), long_speed=1.0, yaw_angle=None
            ),
            action_interval=0.2,
        )
        assert len(sim.agents) == 2
        assert np.allclose(rev.goal, np.array([5.0, 0.0]))
        assert math.isclose(rev.trajectory.yaw_angle[0], 0.0, abs_tol=1e-9)

    def test_intermittent_action_zero_order_hold(self):
        sim = Simulation(start_time=0.0, end_time=1.0, time_step=0.1)
        call_times = []

        def step_acc_callback(agent):
            t = agent.simulation.state.time
            call_times.append(round(t, 4))
            return ActionState(long_acc=0.5 + t, yaw_rate=0.0)

        agent = AgentWithIntermittentActions(
            name="zoh",
            simulation=sim,
            initial_state=KinematicState(pos=np.zeros(2), long_speed=1.0, yaw_angle=0.0),
            action_interval=0.3,
            first_action_time=0.0,
            action_callback=step_acc_callback,
        )
        sim.run()

        # Over [0.0, 0.9] with dt=0.1 and interval=0.3, updates occur at t=0.0, 0.3, 0.6, 0.9
        assert call_times == [0.0, 0.3, 0.6, 0.9]
        expected_indices = [0, 3, 6, 9]
        assert list(np.flatnonzero(agent.action_update_mask)) == expected_indices
        # Zero-order hold between t=0.0 and t=0.2
        assert np.allclose(agent.trajectory.long_acc[0:3], 0.5)
        assert np.allclose(agent.trajectory.long_acc[3:6], 0.8)

    def test_urgency_modulated_lookup_hazard_formula_and_simulation(self):
        sim = Simulation(start_time=0.0, end_time=8.0, time_step=0.05)
        sim.conflict_point = np.array([0.0, 0.0])

        ped = AgentWithIntermittentGaze(
            name="ped",
            simulation=sim,
            goal=np.array([0.0, 6.0]),
            initial_kinematic_state=KinematicState(
                pos=np.array([0.0, -5.0]), long_speed=1.2, yaw_angle=None
            ),
            action_interval=0.2,
            initial_gaze_state=ON_PHONE,
            lambda_down=0.4,
            lambda_up_0=0.30,
            kappa_P=0.25,
            kappa_curb=1.50,
            d_0=2.50,
            curb_pos=np.array([0.0, -1.5]),
            rng_seed=7,
        )

        # Verify exact hazard formula:
        # lambda_up = lambda_up_0 + kappa_P * Tr(P_t) + kappa_curb * exp(-d_curb / d_0)
        cov_trace = 12.0
        d_curb = 2.0
        expected_lambda = 0.30 + 0.25 * 12.0 + 1.50 * math.exp(-2.0 / 2.50)
        assert math.isclose(
            ped.compute_lookup_hazard_rate(cov_trace=cov_trace, d_curb=d_curb),
            expected_lambda,
            rel_tol=1e-12,
        )
        # Higher covariance trace and smaller curb distance both increase urgency
        assert ped.compute_lookup_hazard_rate(
            cov_trace=20.0, d_curb=2.0
        ) > ped.compute_lookup_hazard_rate(cov_trace=2.0, d_curb=2.0)
        assert ped.compute_lookup_hazard_rate(
            cov_trace=5.0, d_curb=0.5
        ) > ped.compute_lookup_hazard_rate(cov_trace=5.0, d_curb=6.0)

        sim.run()
        # Agent progresses smoothly toward goal with valid binary Markov states
        valid_states = {int(GazeState.ON_PHONE), int(GazeState.ON_ROAD)}
        assert set(np.unique(ped.gaze_state)).issubset(valid_states)
        assert ON_ROAD in ped.gaze_state  # Looked up from initial ON_PHONE state
        assert ped.trajectory.pos[1, -1] > 2.0
        assert not np.any(np.isnan(ped.lambda_up_history))


class TestGazeGatedLoomingPerception:
    """Tests for optical looming EKF, missing-observation gating, and eHMI Bayes."""

    def test_looming_equations_and_analytic_jacobian(self):
        sim = Simulation(start_time=0.0, end_time=1.0, time_step=0.05)
        perc = GazeGatedLoomingPerception(simulation=sim, oth_width=1.8, noise_seed=1)

        # Curb-standing ego (d_ego = 0): exact theta = W/D and theta_dot = W*V/D^2
        D = 20.0
        V = 12.0
        z, H = perc.compute_looming_observation_and_jacobian(
            oth_cp_dist=D, oth_speed=V, ego_cp_dist=0.0, ego_speed=0.0
        )
        assert math.isclose(z[0], 1.8 / D, rel_tol=1e-12)
        assert math.isclose(z[1], 1.8 * V / (D ** 2), rel_tol=1e-12)

        # Verify analytic Jacobian against central finite differences for general 2D state
        eps = 1e-6
        d0, v0, dego, vego = 18.5, 9.2, 3.4, 1.3
        _, H_analytic = perc.compute_looming_observation_and_jacobian(d0, v0, dego, vego)
        z_dp, _ = perc.compute_looming_observation_and_jacobian(d0 + eps, v0, dego, vego)
        z_dm, _ = perc.compute_looming_observation_and_jacobian(d0 - eps, v0, dego, vego)
        z_vp, _ = perc.compute_looming_observation_and_jacobian(d0, v0 + eps, dego, vego)
        z_vm, _ = perc.compute_looming_observation_and_jacobian(d0, v0 - eps, dego, vego)
        H_fd = np.column_stack(((z_dp - z_dm) / (2 * eps), (z_vp - z_vm) / (2 * eps)))
        assert np.allclose(H_analytic, H_fd, rtol=1e-6, atol=1e-8)

    def test_monotonic_uncertainty_growth_under_phone_distraction(self):
        sim = Simulation(start_time=0.0, end_time=4.0, time_step=0.05)
        sim.conflict_point = np.zeros(2)
        prior = KalmanPrior(
            cp_dist_mean=35.0, cp_dist_stddev=4.0, speed_mean=10.0, speed_stddev=2.0
        )
        perc_phone = GazeGatedLoomingPerception(
            simulation=sim,
            prior=prior,
            spd_proc_noise_stddev=0.25,
            pos_proc_noise_stddev=0.10,
            noise_seed=11,
        )
        perc_mixed = GazeGatedLoomingPerception(
            simulation=sim,
            prior=prior,
            spd_proc_noise_stddev=0.25,
            pos_proc_noise_stddev=0.10,
            noise_seed=11,
        )

        ego = KinematicState(pos=np.array([0.0, -2.0]), long_speed=0.0, yaw_angle=math.pi / 2)
        for k in range(sim.settings.n_time_steps):
            t = sim.time_stamps[k]
            veh = KinematicState(
                pos=np.array([35.0 - 10.0 * t, 0.0]),
                long_speed=10.0,
                yaw_angle=math.pi,
            )
            # Case A: continuous phone distraction (head pitch = 42 deg > 35 deg)
            perc_phone.update(k, ego, veh, head_pitch_deg=42.0)

            # Case B: foveated gaze for first 25 steps, then phone distraction (45 deg)
            pitch_b = 5.0 if k < 25 else 45.0
            perc_mixed.update(k, ego, veh, head_pitch_deg=pitch_b, eccentricity_deg=0.0)

        assert np.all(perc_phone.states.observation_gated)
        assert np.all(np.isnan(perc_phone.states.looming_noisy))
        # Strictly monotonic growth of Tr(P_t) from prior throughout continuous phone gaze
        prior_trace = float(np.trace(prior.cov_matrix))
        assert perc_phone.states.cov_trace[0] > prior_trace
        assert np.all(np.diff(perc_phone.states.cov_trace) > 0.0)

        # Strictly monotonic growth of Tr(P_t) during the mid-simulation phone-gaze episode
        phone_episode_traces = perc_mixed.states.cov_trace[24:]
        assert np.all(np.diff(phone_episode_traces) > 0.0)

    def test_foveated_beats_peripheral_and_bayesian_ehmi_log_odds(self):
        sim = Simulation(start_time=0.0, end_time=2.5, time_step=0.05)
        sim.conflict_point = np.zeros(2)
        prior = KalmanPrior(
            cp_dist_mean=30.0, cp_dist_stddev=6.0, speed_mean=9.0, speed_stddev=3.0
        )
        perc_fov = GazeGatedLoomingPerception(
            simulation=sim, prior=prior, noise_seed=99
        )
        perc_periph = GazeGatedLoomingPerception(
            simulation=sim, prior=prior, noise_seed=99
        )
        perc_phone = GazeGatedLoomingPerception(
            simulation=sim, prior=prior, noise_seed=99
        )

        ego = KinematicState(pos=np.array([0.0, -1.5]), long_speed=0.0, yaw_angle=math.pi / 2)
        for k in range(sim.settings.n_time_steps):
            t = sim.time_stamps[k]
            veh = KinematicState(
                pos=np.array([30.0 - 8.0 * t, 0.0]),
                long_speed=8.0,
                yaw_angle=math.pi,
            )
            perc_fov.update(
                k, ego, veh, head_pitch_deg=0.0, gaze_mode="foveal", ehmi_signal="yielding"
            )
            perc_periph.update(
                k, ego, veh, head_pitch_deg=15.0, gaze_mode="peripheral", ehmi_signal="yielding"
            )
            perc_phone.update(
                k, ego, veh, head_pitch_deg=45.0, ehmi_signal="yielding"
            )

        # Posterior covariance trace ordering: foveal < peripheral < phone-gated
        assert perc_fov.states.cov_trace[-1] < perc_periph.states.cov_trace[-1]
        assert perc_periph.states.cov_trace[-1] < perc_phone.states.cov_trace[-1]

        # Bayesian eHMI yielding belief ordering: foveal > peripheral > phone (= 0.5 prior)
        assert perc_fov.states.yield_prob[-1] > perc_periph.states.yield_prob[-1] > 0.5
        assert math.isclose(perc_phone.states.yield_prob[-1], 0.5, abs_tol=1e-12)
        assert math.isclose(perc_phone.states.yield_log_odds[-1], 0.0, abs_tol=1e-12)

        # Non-yielding eHMI cue reduces log-odds below 0
        lo_neg = perc_fov.update_ehmi_log_odds(
            0.0, ehmi_signal="non_yielding", head_pitch_deg=0.0, eccentricity_deg=0.0
        )
        assert lo_neg < 0.0

    def test_backward_compatibility_with_base_perception(self):
        sim = Simulation(start_time=0.0, end_time=1.0, time_step=0.1)
        sim.conflict_point = np.zeros(2)
        base_perc = Perception(sim)
        assert isinstance(perc := GazeGatedLoomingPerception(sim), Perception)
        assert base_perc.states.x_true.shape == perc.states.x_true.shape

    def test_closed_loop_coupled_simulation_and_sc_scenario_compat(self):
        import sc_scenario

        class CrossingSimulation(Simulation):
            def __init__(self):
                super().__init__(start_time=0.0, end_time=4.0, time_step=0.05)
                self.conflict_point = np.zeros(2)

            def after_time_step(self):
                k = self.state.i_time_step
                ped_ag, veh_ag = self.agents[0], self.agents[1]
                ped_ag.perception.update(
                    k,
                    ped_ag.get_kinematic_state(k),
                    veh_ag.get_kinematic_state(k),
                    head_pitch_deg=ped_ag.head_pitch_deg[k],
                    ehmi_signal="yielding",
                )

        sim = CrossingSimulation()
        prior = KalmanPrior(32.0, 5.0, 8.0, 2.0)
        perc = GazeGatedLoomingPerception(simulation=sim, prior=prior, noise_seed=21)
        ped = AgentWithIntermittentGaze(
            name="P",
            simulation=sim,
            goal=np.array([0.0, 4.0]),
            initial_kinematic_state=KinematicState(
                pos=np.array([0.0, -4.0]), long_speed=1.1, yaw_angle=None
            ),
            initial_gaze_state=ON_PHONE,
            perception=perc,
            rng_seed=21,
        )
        veh = AgentWithGoal(
            name="V",
            simulation=sim,
            goal=np.array([-20.0, 0.0]),
            initial_kinematic_state=KinematicState(
                pos=np.array([32.0, 0.0]), long_speed=8.0, yaw_angle=None
            ),
        )
        sim.run()
        assert ped.get_covariance_trace() > 0.0
        assert veh.trajectory.pos[0, -1] < 32.0

        # Ensure upstream sc_scenario.SCSimulation still runs with cooperative super()
        sc_sim = sc_scenario.SCSimulation(
            ctrl_types=(sc_scenario.CtrlType.SPEED, sc_scenario.CtrlType.ACCELERATION),
            widths=(0.5, 1.8),
            lengths=(0.5, 4.5),
            goal_positions=np.array([[0.0, 5.0], [-25.0, 0.0]]),
            initial_positions=np.array([[0.0, -5.0], [25.0, 0.0]]),
            initial_speeds=np.array([1.2, 8.0]),
            end_time=0.5,
            time_step=0.1,
            optional_assumptions=sc_scenario.get_assumptions_dict(default_value=False),
        )
        assert len(sc_sim.agents) == 2


class TestDevelopmentalPriorsAndAcousticPerception:
    """Tests for developmental/VRU cohorts and UN R138 acoustic EV perception."""

    def test_all_nine_cohorts_and_distance_over_speed_heuristic(self):
        from developmental_priors import (
            DEVELOPMENTAL_COHORT_PRIORS,
            compute_perceived_effective_ttc,
            evaluate_curb_crossing_rule,
            get_cohort_prior,
        )

        expected_cohorts = {
            "child_6_8yr",
            "child_7_9yr",
            "child_9_11yr",
            "child_12_14yr",
            "older_adult",
            "wheelchair_manual",
            "blind_low_vision",
            "adult_phone",
            "adult_baseline",
        }
        assert expected_cohorts == set(DEVELOPMENTAL_COHORT_PRIORS.keys())

        # Developmental monotonicity: eye height grows and Weber fraction drops from 6-8yr to adult
        assert (
            get_cohort_prior("child_6_8yr").eye_height_m
            < get_cohort_prior("child_9_11yr").eye_height_m
            < get_cohort_prior("adult_baseline").eye_height_m
        )
        assert (
            get_cohort_prior("child_6_8yr").weber_looming_fraction
            > get_cohort_prior("child_12_14yr").weber_looming_fraction
            > get_cohort_prior("adult_baseline").weber_looming_fraction
        )

        # High-speed vehicle (18 m/s > ref 11.18 m/s) at 45 m: true TTC = 2.5 s
        ttc_adult = compute_perceived_effective_ttc(45.0, 18.0, cohort="adult_baseline")
        ttc_child6 = compute_perceived_effective_ttc(45.0, 18.0, cohort="child_6_8yr")
        assert math.isclose(ttc_adult, 2.5, rel_tol=1e-9)
        assert ttc_child6 > 3.2  # Distance-over-speed bias overestimates high-speed TTC

        # Curb ramp gating for wheelchair_manual
        rule_blocked = evaluate_curb_crossing_rule(
            60.0, 8.0, cohort="wheelchair_manual", has_curb_ramp=False
        )
        assert rule_blocked["infrastructure_accessible"] is False
        assert rule_blocked["accepts_gap"] is False

    def test_acoustic_vehicle_perception_avas_vs_quiet_ev_deficit(self):
        from sc_scenario_perception import AcousticVehiclePerception

        sim = Simulation(start_time=0.0, end_time=2.5, time_step=0.05)
        sim.conflict_point = np.zeros(2)
        prior = KalmanPrior(22.0, 6.0, 4.0, 2.0)

        perc_avas = AcousticVehiclePerception(
            simulation=sim,
            prior=prior,
            vehicle_acoustic_mode="ev_avas",
            noise_seed=17,
        )
        perc_quiet = AcousticVehiclePerception(
            simulation=sim,
            prior=prior,
            vehicle_acoustic_mode="ev_quiet",
            noise_seed=17,
        )

        # At 14.4 km/h (4.0 m/s < 20 km/h), quiet EV has exact -12.5 dB deficit vs UN R138 AVAS
        spl_avas, _, snr_avas = perc_avas.compute_auditory_snr_db(15.0, 4.0)
        spl_quiet, _, snr_quiet = perc_quiet.compute_auditory_snr_db(15.0, 4.0)
        assert math.isclose(spl_avas - spl_quiet, 12.5, rel_tol=1e-9)
        assert math.isclose(snr_avas - snr_quiet, 12.5, rel_tol=1e-9)

        ego = KinematicState(pos=np.array([0.0, -1.5]), long_speed=0.0, yaw_angle=math.pi / 2)
        for k in range(sim.settings.n_time_steps):
            t = sim.time_stamps[k]
            veh = KinematicState(
                pos=np.array([22.0 - 4.0 * t, 0.0]),
                long_speed=4.0,
                yaw_angle=math.pi,
            )
            perc_avas.update(k, ego, veh)
            perc_quiet.update(k, ego, veh)

        # Blind/low-vision pedestrian achieves strictly lower uncertainty under UN R138 AVAS
        assert perc_avas.states.cov_trace[-1] < perc_quiet.states.cov_trace[-1]


class TestPerceptualMotorMARLAndHikerCalibration:
    """Tests for commotions_marl (Gym, PettingZoo, TorchRL PPO) and HIKER MLE."""

    def test_base_commotions_remains_zero_pytorch(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "commotions.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert "torch" not in imports
        assert "torchrl" not in imports
        assert "commotions_marl" not in imports

    def test_gym_and_pettingzoo_parallel_env_motor_delay(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("pettingzoo")
        from commotions_marl import (
            CrossingGlanceGymEnv,
            PerceptualMotorParallelEnv,
        )

        gym_env = CrossingGlanceGymEnv(
            cohort="adult_baseline", dt=0.06, tau_motor=0.18
        )
        obs, info = gym_env.reset(seed=5)
        assert obs.shape == (8,)
        assert info["delay_steps"] == 3
        # Step 1 command queued behind 0.18s motor lag -> executed_acc stays 0.0
        obs2, rew, _, _, step_info = gym_env.step(
            np.array([1.0, 1.0], dtype=np.float32)
        )
        assert obs2.shape == (8,)
        assert math.isclose(step_info["executed_acc"], 0.0, abs_tol=1e-9)
        assert isinstance(rew, float)

        # PettingZoo simultaneous ParallelEnv rollout
        pz_env = PerceptualMotorParallelEnv(ped_cohort="child_9_11yr")
        p_obs, _ = pz_env.reset(seed=9)
        assert set(p_obs.keys()) == {"pedestrian", "vehicle"}
        for _ in range(5):
            if not pz_env.agents:
                break
            acts = {a: pz_env.action_space(a).sample() for a in pz_env.agents}
            p_obs, p_rews, _, _, _ = pz_env.step(acts)
            assert "pedestrian" in p_rews and "vehicle" in p_rews

    def test_torchrl_ppo_trainer_step(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("torch")
        pytest.importorskip("tensordict")
        pytest.importorskip("torchrl")
        from commotions_marl import (
            CrossingGlanceGymEnv,
            TorchRLCrossingPPOTrainer,
        )

        gym_env = CrossingGlanceGymEnv(
            cohort="adult_baseline", dt=0.06, tau_motor=0.18
        )
        trainer = TorchRLCrossingPPOTrainer(env=gym_env, hidden_dim=32, lr=5e-4)
        metrics = trainer.train_step(rollout_steps=32, ppo_epochs=2, seed=3)
        assert math.isfinite(metrics["total_loss"])
        assert metrics["rollout_steps"] == 32.0

    def test_hiker_8547_trial_mle_calibration(self):
        from SCPaper.hiker_mle_calibration import fit_hiker_mle_parameters

        result = fit_hiker_mle_parameters()
        assert result.n_trials == 8547
        assert 2.8 < result.crit_ttc_s < 4.8
        assert result.ddm_drift_mu > 0.2
        assert result.delta_log_odds_ehmi > 0.3
        assert result.mcfadden_pseudo_r2 > 0.15
        assert (
            result.predicted_accept_rates["gap_3s_ehmi_1"]
            > result.predicted_accept_rates["gap_3s_ehmi_0"]
        )


