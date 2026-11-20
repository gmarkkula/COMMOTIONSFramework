"""Gymnasium and PettingZoo perceptual-motor road-crossing environments.

Reference:
    Wang, Y., Dogar, M. R., & Markkula, G. (2025). Modelling road user
    interactions as boundedly rational perceptual-motor multi-agent
    reinforcement learning (arXiv:2510.27383).
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

import commotions
from developmental_priors import get_cohort_prior
from sc_scenario_perception import GazeGatedLoomingPerception, KalmanPrior


class CrossingGlanceGymEnv(gym.Env):
    """Single-agent perceptual-motor crossing environment with active visual
    glance gating and motor execution delay ``tau_motor = 0.18 s``.

    Action ``u in [-1, 1]^2``:
        * ``u[0]``: normalized longitudinal acceleration command (scaled by ``max_ped_acc``)
        * ``u[1]``: visual attention command (``u[1] >= 0`` -> ``ON_ROAD`` foveal glance,
          ``u[1] < 0`` -> ``ON_PHONE`` / averted gaze saving attentional cost ``glance_cost``)

    Observation (8-dim ``float32`` belief vector):
        ``[ped_y_m, ped_speed_mps, est_veh_dist_m, est_veh_speed_mps,
           std_veh_dist_m, std_veh_speed_mps, gaze_on_road, yield_prob]``
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cohort: str = "adult_baseline",
        dt: float = 0.06,
        tau_motor: float = 0.18,
        max_steps: int = 120,
        glance_cost: float = 0.015,
        collision_radius_m: float = 1.35,
        goal_y_m: float = 4.0,
        start_y_m: float = -4.0,
    ) -> None:
        super().__init__()
        self.prior_profile = get_cohort_prior(cohort)
        self.dt = float(dt)
        self.tau_motor = float(tau_motor)
        self.max_steps = int(max_steps)
        self.glance_cost = float(glance_cost)
        self.collision_radius_m = float(collision_radius_m)
        self.goal_y_m = float(goal_y_m)
        self.start_y_m = float(start_y_m)
        self.max_ped_acc = 2.5

        self.delay_steps = max(1, int(round(self.tau_motor / self.dt)))
        self._motor_queue: deque[float] = deque(maxlen=self.delay_steps)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-20.0, 0.0, -50.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 4.0, 120.0, 30.0, 50.0, 20.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._step_idx = 0
        self._ped_y = self.start_y_m
        self._ped_v = 0.0
        self._exec_acc = 0.0
        self._veh_x = 32.0
        self._veh_v = 10.0
        self._veh_yielding = False
        self._has_ehmi = False
        self._sim: Optional[commotions.Simulation] = None
        self._perception: Optional[GazeGatedLoomingPerception] = None

    def _get_obs(self, k: int, gaze_on_road: float) -> np.ndarray:
        assert self._perception is not None
        est = self._perception.states.x_estimated[:, k]
        cov = self._perception.states.cov_matrix[:, :, k]
        std_d = math.sqrt(max(0.0, float(cov[0, 0])))
        std_v = math.sqrt(max(0.0, float(cov[1, 1])))
        y_prob = float(self._perception.states.yield_prob[k])
        return np.array(
            [
                self._ped_y,
                self._ped_v,
                float(np.clip(est[0], -50.0, 120.0)),
                float(np.clip(est[1], 0.0, 30.0)),
                float(min(std_d, 50.0)),
                float(min(std_v, 20.0)),
                float(gaze_on_road),
                y_prob,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_idx = 0
        self._ped_y = self.start_y_m
        self._ped_v = 0.0
        self._exec_acc = 0.0
        self._motor_queue.clear()
        for _ in range(self.delay_steps):
            self._motor_queue.append(0.0)

        opts = options or {}
        self._veh_v = float(opts.get("veh_speed_mps", self._rng.uniform(8.0, 14.0)))
        time_gap = float(opts.get("time_gap_s", self._rng.uniform(2.2, 5.0)))
        self._veh_x = self._veh_v * time_gap
        self._veh_yielding = bool(
            opts.get("veh_yielding", bool(self._rng.random() < 0.5))
        )
        self._has_ehmi = bool(opts.get("has_ehmi", self._veh_yielding))

        self._sim = commotions.Simulation(
            start_time=0.0,
            end_time=(self.max_steps + 2) * self.dt,
            time_step=self.dt,
        )
        self._sim.conflict_point = np.zeros(2)
        prior = KalmanPrior(
            cp_dist_mean=self._veh_x,
            cp_dist_stddev=4.5,
            speed_mean=self._veh_v,
            speed_stddev=2.0,
        )
        self._perception = GazeGatedLoomingPerception(
            simulation=self._sim,
            prior=prior,
            noise_seed=int(self._rng.integers(0, 1_000_000)),
        )

        ego_s = commotions.KinematicState(
            pos=np.array([0.0, self._ped_y]),
            long_speed=self._ped_v,
            yaw_angle=math.pi / 2.0,
        )
        veh_s = commotions.KinematicState(
            pos=np.array([self._veh_x, 0.0]),
            long_speed=self._veh_v,
            yaw_angle=math.pi,
        )
        ehmi = "yielding" if (self._veh_yielding and self._has_ehmi) else None
        self._perception.update(0, ego_s, veh_s, head_pitch_deg=5.0, ehmi_signal=ehmi)
        obs = self._get_obs(0, gaze_on_road=1.0)
        return obs, {"tau_motor_s": self.tau_motor, "delay_steps": self.delay_steps}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._perception is not None
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        cmd_acc = float(np.clip(act[0], -1.0, 1.0)) * self.max_ped_acc
        gaze_on_road = 1.0 if float(act[1]) >= 0.0 else 0.0

        # Delayed motor execution through FIFO buffer + first-order neuromuscular lag
        delayed_cmd = self._motor_queue[0]
        self._motor_queue.append(cmd_acc)
        alpha_lag = min(1.0, self.dt / max(self.tau_motor, 1e-4))
        self._exec_acc += alpha_lag * (delayed_cmd - self._exec_acc)

        prev_y = self._ped_y
        max_v = self.prior_profile.free_walking_speed_mps * 1.25
        self._ped_v = float(np.clip(self._ped_v + self._exec_acc * self.dt, 0.0, max_v))
        self._ped_y += self._ped_v * self.dt

        # Vehicle longitudinal kinematics (comfortable yielding deceleration if yielding)
        if self._veh_yielding and self._veh_x > 2.5:
            veh_acc = -(self._veh_v ** 2) / (2.0 * max(self._veh_x - 2.2, 0.5))
            veh_acc = float(np.clip(veh_acc, -4.5, 0.0))
        else:
            veh_acc = 0.0
        self._veh_v = max(0.0, self._veh_v + veh_acc * self.dt)
        self._veh_x -= self._veh_v * self.dt

        self._step_idx += 1
        ego_s = commotions.KinematicState(
            pos=np.array([0.0, self._ped_y]),
            long_speed=self._ped_v,
            yaw_angle=math.pi / 2.0,
        )
        veh_s = commotions.KinematicState(
            pos=np.array([self._veh_x, 0.0]),
            long_speed=self._veh_v,
            yaw_angle=math.pi,
        )
        head_pitch = 5.0 if gaze_on_road > 0.5 else 45.0
        ehmi = "yielding" if (self._veh_yielding and self._has_ehmi) else None
        self._perception.update(
            self._step_idx, ego_s, veh_s, head_pitch_deg=head_pitch, ehmi_signal=ehmi
        )

        sep_m = math.hypot(self._veh_x, self._ped_y)
        collided = sep_m <= self.collision_radius_m
        reached_goal = self._ped_y >= self.goal_y_m
        terminated = bool(collided or reached_goal)
        truncated = bool((not terminated) and (self._step_idx >= self.max_steps))

        progress = (self._ped_y - prev_y) / (self.goal_y_m - self.start_y_m)
        reward = (
            2.0 * progress
            - self.glance_cost * gaze_on_road
            - 0.005 * (cmd_acc ** 2)
            - 0.01 * self.dt
        )
        if reached_goal:
            reward += 5.0
        if collided:
            reward -= 15.0

        obs = self._get_obs(self._step_idx, gaze_on_road=gaze_on_road)
        info = {
            "collided": collided,
            "reached_goal": reached_goal,
            "separation_m": sep_m,
            "executed_acc": self._exec_acc,
        }
        return obs, float(reward), terminated, truncated, info


class PerceptualMotorParallelEnv(ParallelEnv):
    """Two-agent simultaneous-move PettingZoo ``ParallelEnv`` implementing
    boundedly rational pedestrian-vehicle conflict negotiation
    (Wang, Dogar & Markkula, 2025, arXiv:2510.27383).
    """

    metadata = {"name": "commotions_perceptual_motor_v0", "render_modes": []}

    def __init__(
        self,
        ped_cohort: str = "adult_baseline",
        dt: float = 0.06,
        tau_motor_ped: float = 0.18,
        tau_motor_veh: float = 0.15,
        max_steps: int = 120,
        collision_radius_m: float = 1.40,
    ) -> None:
        super().__init__()
        self.possible_agents = ["pedestrian", "vehicle"]
        self.agents = list(self.possible_agents)
        self.ped_prior = get_cohort_prior(ped_cohort)
        self.dt = float(dt)
        self.tau_motor_ped = float(tau_motor_ped)
        self.tau_motor_veh = float(tau_motor_veh)
        self.max_steps = int(max_steps)
        self.collision_radius_m = float(collision_radius_m)

        self._obs_spaces = {
            "pedestrian": spaces.Box(
                low=np.array([-20.0, 0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([20.0, 4.0, 120.0, 30.0, 50.0, 20.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "vehicle": spaces.Box(
                low=np.array([-60.0, 0.0, -20.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([120.0, 30.0, 20.0, 4.0, 20.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        }
        self._act_spaces = {
            # pedestrian: [norm_acc, glance_toggle]
            "pedestrian": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            # vehicle: [norm_acc, ehmi_yield_signal]
            "vehicle": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        }

        self._rng = np.random.default_rng()
        self._ped_queue: deque[float] = deque(
            maxlen=max(1, int(round(self.tau_motor_ped / self.dt)))
        )
        self._veh_queue: deque[float] = deque(
            maxlen=max(1, int(round(self.tau_motor_veh / self.dt)))
        )
        self._step_idx = 0
        self._ped_y = -4.0
        self._ped_v = 0.0
        self._veh_x = 30.0
        self._veh_v = 10.0
        self._sim: Optional[commotions.Simulation] = None
        self._ped_perc: Optional[GazeGatedLoomingPerception] = None

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_spaces[agent]

    def _make_obs(self, k: int, gaze_on_road: float, ehmi_on: float) -> Dict[str, np.ndarray]:
        assert self._ped_perc is not None
        est = self._ped_perc.states.x_estimated[:, k]
        cov = self._ped_perc.states.cov_matrix[:, :, k]
        ped_obs = np.array(
            [
                self._ped_y,
                self._ped_v,
                float(np.clip(est[0], -60.0, 120.0)),
                float(np.clip(est[1], 0.0, 30.0)),
                math.sqrt(max(0.0, float(cov[0, 0]))),
                math.sqrt(max(0.0, float(cov[1, 1]))),
                gaze_on_road,
                float(self._ped_perc.states.yield_prob[k]),
            ],
            dtype=np.float32,
        )
        noisy_ped_y = float(self._ped_y + self._rng.normal(0.0, 0.15))
        noisy_ped_v = float(max(0.0, self._ped_v + self._rng.normal(0.0, 0.10)))
        veh_obs = np.array(
            [
                self._veh_x,
                self._veh_v,
                float(np.clip(noisy_ped_y, -20.0, 20.0)),
                float(np.clip(noisy_ped_v, 0.0, 4.0)),
                float(min(math.hypot(self._veh_x, self._ped_y), 20.0)),
                ehmi_on,
            ],
            dtype=np.float32,
        )
        return {"pedestrian": ped_obs, "vehicle": veh_obs}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_idx = 0
        self._ped_y = -4.0
        self._ped_v = 0.0
        self._veh_v = float(self._rng.uniform(8.5, 13.5))
        self._veh_x = self._veh_v * float(self._rng.uniform(2.5, 4.5))

        self._ped_queue.clear()
        for _ in range(self._ped_queue.maxlen or 1):
            self._ped_queue.append(0.0)
        self._veh_queue.clear()
        for _ in range(self._veh_queue.maxlen or 1):
            self._veh_queue.append(0.0)

        self._sim = commotions.Simulation(
            start_time=0.0,
            end_time=(self.max_steps + 2) * self.dt,
            time_step=self.dt,
        )
        self._sim.conflict_point = np.zeros(2)
        prior = KalmanPrior(self._veh_x, 4.5, self._veh_v, 2.0)
        self._ped_perc = GazeGatedLoomingPerception(
            simulation=self._sim,
            prior=prior,
            noise_seed=int(self._rng.integers(0, 1_000_000)),
        )
        ego_s = commotions.KinematicState(
            pos=np.array([0.0, self._ped_y]), long_speed=0.0, yaw_angle=math.pi / 2.0
        )
        veh_s = commotions.KinematicState(
            pos=np.array([self._veh_x, 0.0]), long_speed=self._veh_v, yaw_angle=math.pi
        )
        self._ped_perc.update(0, ego_s, veh_s, head_pitch_deg=5.0, ehmi_signal=None)
        obs = self._make_obs(0, gaze_on_road=1.0, ehmi_on=0.0)
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}
        assert self._ped_perc is not None

        p_act = np.asarray(actions["pedestrian"], dtype=np.float32).reshape(-1)
        v_act = np.asarray(actions["vehicle"], dtype=np.float32).reshape(-1)

        p_cmd_acc = float(np.clip(p_act[0], -1.0, 1.0)) * 2.5
        gaze_on_road = 1.0 if float(p_act[1]) >= 0.0 else 0.0
        v_cmd_acc = float(np.clip(v_act[0], -1.0, 1.0)) * 4.5
        ehmi_on = 1.0 if float(v_act[1]) >= 0.0 else 0.0

        p_exec = self._ped_queue[0]
        self._ped_queue.append(p_cmd_acc)
        v_exec = self._veh_queue[0]
        self._veh_queue.append(v_cmd_acc)

        prev_y = self._ped_y
        prev_x = self._veh_x
        self._ped_v = float(
            np.clip(
                self._ped_v + p_exec * self.dt,
                0.0,
                self.ped_prior.free_walking_speed_mps * 1.25,
            )
        )
        self._ped_y += self._ped_v * self.dt

        self._veh_v = float(np.clip(self._veh_v + v_exec * self.dt, 0.0, 16.0))
        self._veh_x -= self._veh_v * self.dt

        self._step_idx += 1
        ego_s = commotions.KinematicState(
            pos=np.array([0.0, self._ped_y]),
            long_speed=self._ped_v,
            yaw_angle=math.pi / 2.0,
        )
        veh_s = commotions.KinematicState(
            pos=np.array([self._veh_x, 0.0]),
            long_speed=self._veh_v,
            yaw_angle=math.pi,
        )
        ehmi_sig = "yielding" if ehmi_on > 0.5 else None
        self._ped_perc.update(
            self._step_idx,
            ego_s,
            veh_s,
            head_pitch_deg=(5.0 if gaze_on_road > 0.5 else 45.0),
            ehmi_signal=ehmi_sig,
        )

        sep = math.hypot(self._veh_x, self._ped_y)
        collided = sep <= self.collision_radius_m
        both_cleared = (self._ped_y >= 4.0) and (self._veh_x <= -6.0)
        ped_cleared = self._ped_y >= 4.0

        terminated = bool(collided or both_cleared or ped_cleared)
        truncated = bool((not terminated) and (self._step_idx >= self.max_steps))

        ped_rew = (
            1.5 * ((self._ped_y - prev_y) / 8.0)
            - 0.015 * gaze_on_road
            - 0.005 * (p_cmd_acc ** 2)
        )
        veh_rew = (
            1.0 * ((prev_x - self._veh_x) / 40.0)
            - 0.004 * (v_cmd_acc ** 2)
            - 0.005 * ehmi_on
        )
        if ped_cleared:
            ped_rew += 4.0
        if self._veh_x <= -6.0:
            veh_rew += 2.0
        if collided:
            ped_rew -= 15.0
            veh_rew -= 15.0

        obs = self._make_obs(self._step_idx, gaze_on_road, ehmi_on)
        rewards = {"pedestrian": float(ped_rew), "vehicle": float(veh_rew)}
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {
            a: {"collided": collided, "separation_m": sep} for a in self.agents
        }
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos
