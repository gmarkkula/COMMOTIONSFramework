# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 07:32:13 2021

@author: tragma
"""
import math
from dataclasses import dataclass
import numpy as np
from numpy.random import default_rng
import commotions
import sc_scenario_helper


N_STATE_DIMS = 2 # position and speed

class PerceptionStates:
    pass
        

class KalmanPrior:
    def __init__(self, cp_dist_mean, cp_dist_stddev, speed_mean, speed_stddev): 
        self.x_estimated = np.array((cp_dist_mean, speed_mean))
        self.cov_matrix = np.array(((cp_dist_stddev ** 2, 0), 
                                    (0, speed_stddev ** 2)))
        

@dataclass
class Perception:
    """ A class to manage perception of another agent in an SC scenario. 
        
        Arguments to __init__:
            
            simulation : commotions.Simulation
                The SC scenario simulation.
            pos_obs_noise_stddev: float
                Standard deviation of Gaussian noise in sensing of the other
                agent's position. In radians if angular_perception is True,
                otherwise in meters. Set to zero (the default value) to assume
                non-noisy perception (disabling all other features of this 
                class).
            noise_seed: int
                Seed for the random noise generation. Default is None, for
                non-predictable noise.
            angular_perception: bool
                If True, assume that position estimation is based on angle below
                horizon, with constant angular noise, determines position noise,
                if False (default), assume that position estimation has constant
                spatial noise.
            ego_eye_height: float
                Eye height over the ground of the perceiving observer. Only
                needed if angular_perception is True. Default is None.
            kalman_filter: bool
                If True, the noisy distance to conflict point information
                will be Kalman filtered to yield filtered distance and speed
                estimates. Default is False. 
            x_initial: np.ndarray
                A numpy array of length 2, describing initial Kalman filter
                estimates for the other agent's distance to conflict point and 
                longitudinal speed. Only required if kalman_filter is True.
                Default is None.
            cov_matrix_initial: np.ndarray
                A numpy array of size 2 x 2, describing the initial Kalman filter
                estimate of the covariance matrix. Only required if kalman_filter 
                is True. Default is None.
            spd_proc_noise_stddev: float
                Standard deviation of Gaussian noise assumed for the longitudinal
                speed in the process model of the Kalman filter. Unit m/s. 
                Only relevant if kalman_filter is True. Default is zero.
            draw_from_estimate: bool
                If False (the default), the perceived state of the other agent
                will be the Kalman filter's current maximum probability 
                estimates of conflict point distance and speed. 
                If True, the perceived state will instead be a random draw 
                from the Kalman filter's current estimated joint distribution 
                for distance and speed. Only relevant if kalman_filter is True. 
        
        The intended use after __init__ is to call method .update() for each time 
        step in a simulation, after which the attribute .perc_oth_state 
        contains a currently perceived state for the other agent at this time
        step.
        
        After the entire simulation is done, the property .states contains 
        arrays with time histories of various perception quantities, all 
        numpy arrays of dimension 2 x N or 2 x 2 x N, where 2 is for position
        to conflict point and longitudinal speed, and N is the number
        of time steps in the simulation:
            
            x_true: true values
            x_noisy: with noise added (the speeds remain NaN if doing Kalman 
                                        filtering, since only observing noisy
                                        distances to conflict point)
            x_estimated: maximum probability Kalman filter estimates
            x_perceived: the perceived values; for many uses of this class
                          this will be a pointer to one of the arrays above
            cov_matrix: Kalman filter estimates of the covariance matrix 
        
    """
    
    simulation: commotions.Simulation
    pos_obs_noise_stddev: float = 0
    noise_seed: int = None
    angular_perception: bool = False
    ego_eye_height: float = None
    kalman_filter: bool = False
    prior: KalmanPrior = None
    spd_proc_noise_stddev: float = 0
    draw_from_estimate: bool = False
    
    
    def update_states(self, i_time_step, ego_state, oth_state):
        # make sure we are in synch with the user's simulation
        assert i_time_step == self.i_time_step + 1
        self.i_time_step = i_time_step
        # get true state of other agent
        self.states.x_true[:, i_time_step] = (oth_state.signed_CP_dist, 
                                              oth_state.long_speed)
        # if perception is noise-free, we are done now (since the other output 
        # arrays in self.states are already pointing to x_true)
        if not self.noisy_perception:
            return
        # get observation noise magnitude 
        if self.angular_perception:
            # perceiving distance from angle below horizon
            # (see 2021-11-02 handwritten notes)
            D = math.sqrt(ego_state.signed_CP_dist ** 2 
                          + oth_state.signed_CP_dist ** 2)
            curr_obs_noise_stddev = (
                abs(oth_state.signed_CP_dist)
                * (1 - (self.ego_eye_height
                        / (D * math.tan(math.atan(self.ego_eye_height / D) 
                                        + self.pos_obs_noise_stddev
                                        )
                           )
                        )
                   )
                )
            # very rarely, for small D, the expression above can yield a negative
            # value
            curr_obs_noise_stddev = max(0, curr_obs_noise_stddev)
        else:
            # constant spatial noise
            curr_obs_noise_stddev = self.pos_obs_noise_stddev
        # get a noisy observation of position
        self.states.x_noisy[0, i_time_step] = self.rng.normal(
            loc=self.states.x_true[0, i_time_step], scale=curr_obs_noise_stddev)
        # if no filtering is to be applied, we are almost done now
        if not self.kalman_filter:
            # get a simple noisy estimate of speed based on the two most recent 
            # position observations - this should only really used if perception
            # is noisy but without filtering
            self.states.x_noisy[1, i_time_step] = -(
                (self.states.x_noisy[0, i_time_step] 
                 - self.states.x_noisy[0, i_time_step-1])
                / self.simulation.settings.time_step)
            return
        # do Kalman filtering (see 2021-10-30 handwritten notes and 
        # https://en.wikipedia.org/wiki/Kalman_filter#Details)
        # - predicted state estimate
        x_pred = self.Fmatrix @ self.states.x_estimated[:, i_time_step-1]
        # - predicted estimate covariance
        pred_cov_matrix = (self.Fmatrix 
                           @ self.states.cov_matrix[:, :, i_time_step-1]
                           @ self.Fmatrix.T + self.Qmatrix)
        # - innovation
        ytilde = self.states.x_noisy[0, i_time_step] - self.Hmatrix @ x_pred
        # - innovation covariance
        innov_cov = (self.Hmatrix @ pred_cov_matrix @ self.Hmatrix.T 
                     + curr_obs_noise_stddev ** 2)
        # - optimal Kalman gain
        Kmatrix = pred_cov_matrix @ self.Hmatrix.T / innov_cov
        # - updated state estimate
        self.states.x_estimated[:, i_time_step] = x_pred + Kmatrix @ ytilde
        # - updated estimate covariance
        self.states.cov_matrix[:, :, i_time_step] = (
            np.eye(N_STATE_DIMS) - Kmatrix @ self.Hmatrix) @ pred_cov_matrix
        # we may also want do randomly draw the perceived noisy state from 
        # the Kalman estimate
        if self.draw_from_estimate:
            self.states.x_perceived[:, i_time_step] = self.rng.multivariate_normal(
                mean=self.states.x_estimated[:, i_time_step],
                cov=self.states.cov_matrix[:, :, i_time_step])


    def update(self, i_time_step, ego_state, oth_state):
        # call internal method for updating state arrays
        self.update_states(i_time_step, ego_state, oth_state)
        # set the perceived current state of other agent
        # - ideal perception of yaw angle
        self.perc_oth_state.yaw_angle = oth_state.yaw_angle
        # - get the perceived (possibly noisy, possibly filtered) position and
        # - speed. NB: Truncating the speed to non-negative values, since the 
        # - the value estimation in sc_scenario etc currently can't handle 
        # - reversing other agents, and if it could handle it, the values reported
        # - for a reversing agent should anyway be equal or very similar to the
        # - values reported for a stationary agent.
        self.perc_oth_state.signed_CP_dist = self.states.x_perceived[0, i_time_step]
        self.states.x_perceived[1, i_time_step] = max(
            0, self.states.x_perceived[1, i_time_step]) 
        self.perc_oth_state.long_speed = self.states.x_perceived[1, i_time_step]
        # - also set the position in the .pos attribute
        self.perc_oth_state.pos = sc_scenario_helper.get_pos_from_signed_dist_to_conflict_pt(
            self.simulation.conflict_point, self.perc_oth_state)
        
    
    def __post_init__(self):
        n_time_steps = self.simulation.settings.n_time_steps
        # expecting the first update to be to time step 0
        self.i_time_step = -1
        # if we are doing angular perception we need an ego eye height
        if self.angular_perception and self.ego_eye_height == None:
            raise Exception('Need an ego_eye_height to do angular perception.')
        # are we doing noisy perception?
        self.noisy_perception = self.pos_obs_noise_stddev > 0
        if self.noisy_perception:
            self.rng = default_rng(seed = self.noise_seed)
        # object for storing arrays of perception states - depending on user 
        # settings some of these may just be pointers
        self.states = PerceptionStates()
        self.states.x_true = np.full((2, n_time_steps), np.nan)
        if self.noisy_perception:
            self.states.x_noisy = np.full((2, n_time_steps), np.nan)
            if self.kalman_filter:
                self.states.x_estimated = np.full((2, n_time_steps), np.nan)
                self.states.cov_matrix = np.full((2, 2, n_time_steps), np.nan)
                # initial states
                self.states.x_estimated[:, -1] = self.prior.x_estimated
                self.states.cov_matrix[:, :, -1] = self.prior.cov_matrix
                if self.draw_from_estimate:
                    self.states.x_perceived = np.full((2, n_time_steps), np.nan)
                else:
                    # not drawing noisy samples from the Kalman estimate
                    self.states.x_perceived = self.states.x_estimated
            else:
                # not doing Kalman filtering
                self.states.x_perceived = self.states.x_noisy
        else:
            # not doing noisy perception
            self.states.x_noisy = self.states.x_true
            self.states.x_perceived = self.states.x_true
        # object for storing updated current perceived state of other agent
        self.perc_oth_state = commotions.KinematicState(
            pos=None, long_speed=None, yaw_angle=None)
        # Kalman filtering matrices, if needed (see 2021-10-30 hand written notes)
        if self.noisy_perception and self.kalman_filter:
            # state transition model (negative top right element since 
            # distance to conflict point decreases with positive speed)
            self.Fmatrix = np.array(((1, -self.simulation.settings.time_step),
                                     (0, 1)))
            # observation model (a row vector)
            self.Hmatrix = np.array(((1, 0),))
            # process noise covariance matrix
            self.Qmatrix = np.array(((0, 0),
                                     (0, self.spd_proc_noise_stddev ** 2)))


@dataclass
class GazeGatedLoomingPerception(Perception):
    """Extended Kalman Filter perception model combining foveated vs.
    peripheral optical looming (``theta = W / D``, ``theta_dot = W * V / D^2``),
    missing-observation gating under downward smartphone head pitch
    (``head_pitch_deg > pitch_cutoff_deg``), and Bayesian log-odds updating
    for external HMI (eHMI) vehicle yielding intent.
    """

    kalman_filter: bool = True
    spd_proc_noise_stddev: float = 0.25
    pos_proc_noise_stddev: float = 0.10
    oth_width: float = 1.8
    theta_obs_noise_foveal: float = 0.004
    theta_dot_obs_noise_foveal: float = 0.008
    eccentricity_scale_deg: float = 2.5
    peripheral_eccentricity_deg: float = 20.0
    pitch_cutoff_deg: float = 35.0
    min_looming_dist: float = 0.5
    prior_yield_prob: float = 0.5
    ehmi_tpr: float = 0.85
    ehmi_fpr: float = 0.15
    max_log_odds: float = 12.0

    def __post_init__(self):
        if self.prior is None:
            self.prior = KalmanPrior(
                cp_dist_mean=25.0,
                cp_dist_stddev=6.0,
                speed_mean=8.0,
                speed_stddev=2.5,
            )
        if not (0.0 < self.prior_yield_prob < 1.0):
            raise ValueError("prior_yield_prob must lie strictly in (0, 1).")
        if not (0.0 < self.ehmi_fpr < self.ehmi_tpr < 1.0):
            raise ValueError("Require 0 < ehmi_fpr < ehmi_tpr < 1.")

        n_time_steps = self.simulation.settings.n_time_steps
        self.i_time_step = -1
        self.noisy_perception = True
        self.rng = default_rng(seed=self.noise_seed)

        self.states = PerceptionStates()
        self.states.x_true = np.full((N_STATE_DIMS, n_time_steps), np.nan)
        self.states.x_noisy = np.full((N_STATE_DIMS, n_time_steps), np.nan)
        self.states.x_estimated = np.full((N_STATE_DIMS, n_time_steps), np.nan)
        self.states.cov_matrix = np.full(
            (N_STATE_DIMS, N_STATE_DIMS, n_time_steps), np.nan
        )
        self.states.x_estimated[:, -1] = self.prior.x_estimated
        self.states.cov_matrix[:, :, -1] = self.prior.cov_matrix
        if self.draw_from_estimate:
            self.states.x_perceived = np.full(
                (N_STATE_DIMS, n_time_steps), np.nan
            )
        else:
            self.states.x_perceived = self.states.x_estimated

        # Optical looming & gaze-gating diagnostic arrays
        self.states.looming_true = np.full((2, n_time_steps), np.nan)
        self.states.looming_noisy = np.full((2, n_time_steps), np.nan)
        self.states.observation_gated = np.zeros(n_time_steps, dtype=bool)
        self.states.cov_trace = np.full(n_time_steps, np.nan)
        self.states.yield_log_odds = np.full(n_time_steps, np.nan)
        self.states.yield_prob = np.full(n_time_steps, np.nan)

        self._initial_log_odds = math.log(
            self.prior_yield_prob / (1.0 - self.prior_yield_prob)
        )
        self.perc_oth_state = commotions.KinematicState(
            pos=None, long_speed=None, yaw_angle=None
        )

        dt = self.simulation.settings.time_step
        self.Fmatrix = np.array(((1.0, -dt), (0.0, 1.0)))
        self.Qmatrix = np.array(
            (
                (self.pos_proc_noise_stddev ** 2, 0.0),
                (0.0, self.spd_proc_noise_stddev ** 2),
            )
        )

    def _resolve_signed_cp_dist(self, state):
        if hasattr(state, 'signed_CP_dist') and state.signed_CP_dist is not None:
            return float(state.signed_CP_dist)
        conflict_pt = getattr(self.simulation, 'conflict_point', None)
        if conflict_pt is None:
            conflict_pt = np.zeros(2)
        yaw = 0.0 if state.yaw_angle is None else float(state.yaw_angle)
        tmp = commotions.KinematicState(
            pos=np.asarray(state.pos, dtype=float),
            long_speed=float(state.long_speed),
            yaw_angle=yaw,
        )
        return float(
            sc_scenario_helper.get_signed_dist_to_conflict_pt(conflict_pt, tmp)
        )

    def compute_looming_observation_and_jacobian(
        self, oth_cp_dist, oth_speed, ego_cp_dist=0.0, ego_speed=0.0
    ):
        """Compute optical angle ``theta = W / D`` and expansion rate
        ``theta_dot = W * V_closing / D^2`` together with the 2x2 EKF Jacobian
        with respect to ``[oth_cp_dist, oth_speed]``.
        """
        W = self.oth_width
        d_oth = float(oth_cp_dist)
        v_oth = float(oth_speed)
        d_ego = float(ego_cp_dist)
        v_ego = float(ego_speed)

        D_raw_sq = d_ego ** 2 + d_oth ** 2
        min_sq = self.min_looming_dist ** 2
        if D_raw_sq <= min_sq:
            D = self.min_looming_dist
            # Clamped near-field regime
            v_closing = v_oth if d_oth >= 0.0 else -v_oth
            theta = W / D
            theta_dot = W * v_closing / (D ** 2)
            sign_d = 1.0 if d_oth >= 0.0 else -1.0
            H = np.array(
                (
                    (-W * sign_d / (D ** 2), 0.0),
                    (-2.0 * W * v_closing * sign_d / (D ** 3), W * sign_d / (D ** 2)),
                )
            )
            return np.array((theta, theta_dot)), H

        D = math.sqrt(D_raw_sq)
        # Radial closing speed -dD/dt = (d_ego * v_ego + d_oth * v_oth) / D
        radial_num = d_ego * v_ego + d_oth * v_oth
        theta = W / D
        theta_dot = W * radial_num / (D ** 3)

        dtheta_dd = -W * d_oth / (D ** 3)
        dtheta_dv = 0.0
        dthetadot_dd = (W * v_oth / (D ** 3)) - (
            3.0 * W * d_oth * radial_num / (D ** 5)
        )
        dthetadot_dv = W * d_oth / (D ** 3)
        H = np.array(((dtheta_dd, dtheta_dv), (dthetadot_dd, dthetadot_dv)))
        return np.array((theta, theta_dot)), H

    def get_observation_noise_cov(self, eccentricity_deg):
        ecc = max(0.0, abs(float(eccentricity_deg)))
        scale = 1.0 + ecc / self.eccentricity_scale_deg
        sigma_theta = self.theta_obs_noise_foveal * scale
        sigma_theta_dot = self.theta_dot_obs_noise_foveal * scale
        return np.array(
            ((sigma_theta ** 2, 0.0), (0.0, sigma_theta_dot ** 2))
        )

    def update_ehmi_log_odds(
        self,
        prev_log_odds,
        ehmi_signal=None,
        head_pitch_deg=0.0,
        eccentricity_deg=0.0,
        dt=None,
    ):
        """Time-step-invariant Bayesian log-odds update for vehicle yielding
        intention given an eHMI cue, gated by head pitch and modulated by
        retinal eccentricity.
        """
        if ehmi_signal is None or ehmi_signal == 0:
            return float(prev_log_odds)
        if float(head_pitch_deg) > self.pitch_cutoff_deg:
            # Visual eHMI is outside vertical field of view during phone gaze
            return float(prev_log_odds)

        if dt is None:
            dt = self.simulation.settings.time_step

        ecc = max(0.0, abs(float(eccentricity_deg)))
        acuity_weight = 1.0 / (1.0 + ecc / (2.0 * self.eccentricity_scale_deg))
        eff_tpr = 0.5 + acuity_weight * (self.ehmi_tpr - 0.5)
        eff_fpr = 0.5 + acuity_weight * (self.ehmi_fpr - 0.5)

        if ehmi_signal in (1, True, 'yielding', 'yield'):
            ll_rate = math.log(eff_tpr / eff_fpr)
        elif ehmi_signal in (-1, False, 'non_yielding', 'assert'):
            ll_rate = math.log((1.0 - eff_tpr) / (1.0 - eff_fpr))
        else:
            raise ValueError(f"Unrecognised ehmi_signal: {ehmi_signal!r}")

        return float(
            np.clip(
                prev_log_odds + ll_rate * float(dt),
                -self.max_log_odds,
                self.max_log_odds,
            )
        )

    def update_states(
        self,
        i_time_step,
        ego_state,
        oth_state,
        head_pitch_deg=None,
        eccentricity_deg=None,
        gaze_mode=None,
        ehmi_signal=None,
    ):
        assert i_time_step == self.i_time_step + 1
        self.i_time_step = i_time_step

        ego_cp_dist = self._resolve_signed_cp_dist(ego_state)
        oth_cp_dist = self._resolve_signed_cp_dist(oth_state)
        ego_speed = float(ego_state.long_speed)
        oth_speed = float(oth_state.long_speed)

        self.states.x_true[:, i_time_step] = (oth_cp_dist, oth_speed)
        z_true, _ = self.compute_looming_observation_and_jacobian(
            oth_cp_dist, oth_speed, ego_cp_dist, ego_speed
        )
        self.states.looming_true[:, i_time_step] = z_true

        if head_pitch_deg is None:
            head_pitch_deg = getattr(ego_state, 'head_pitch_deg', 0.0)
        head_pitch_deg = float(head_pitch_deg)

        if eccentricity_deg is None:
            if gaze_mode == 'peripheral':
                eccentricity_deg = self.peripheral_eccentricity_deg
            elif gaze_mode == 'foveal':
                eccentricity_deg = 0.0
            else:
                eccentricity_deg = getattr(
                    ego_state, 'gaze_eccentricity_deg', abs(head_pitch_deg)
                )
        eccentricity_deg = float(eccentricity_deg)

        prev_x = self.states.x_estimated[:, i_time_step - 1]
        prev_P = self.states.cov_matrix[:, :, i_time_step - 1]

        # EKF prediction step
        x_pred = self.Fmatrix @ prev_x
        is_gated = head_pitch_deg > self.pitch_cutoff_deg
        self.states.observation_gated[i_time_step] = is_gated

        if is_gated:
            # Missing observation under downward smartphone gaze (> 35 deg):
            # Pure prediction. Any positive instantaneous retinal cross-term
            # loses sensory lock (Schur product with I preserves SPD & trace).
            P_base = prev_P.copy()
            if P_base[0, 1] > 0.0:
                P_base[0, 1] = 0.0
                P_base[1, 0] = 0.0
            P_pred = self.Fmatrix @ P_base @ self.Fmatrix.T + self.Qmatrix
            self.states.x_estimated[:, i_time_step] = x_pred
            self.states.cov_matrix[:, :, i_time_step] = P_pred
        else:
            P_pred = self.Fmatrix @ prev_P @ self.Fmatrix.T + self.Qmatrix
            R_mat = self.get_observation_noise_cov(eccentricity_deg)
            z_noisy = self.rng.multivariate_normal(mean=z_true, cov=R_mat)
            self.states.looming_noisy[:, i_time_step] = z_noisy
            # Approximate implied distance from noisy optical size for diagnostics
            noisy_D = self.oth_width / max(1e-4, abs(z_noisy[0]))
            self.states.x_noisy[0, i_time_step] = math.sqrt(
                max(0.0, noisy_D ** 2 - ego_cp_dist ** 2)
            )
            self.states.x_noisy[1, i_time_step] = (
                z_noisy[1] * (noisy_D ** 2) / self.oth_width
            )

            z_hat, H_mat = self.compute_looming_observation_and_jacobian(
                x_pred[0], x_pred[1], ego_cp_dist, ego_speed
            )
            ytilde = z_noisy - z_hat
            S_mat = H_mat @ P_pred @ H_mat.T + R_mat
            K_mat = P_pred @ H_mat.T @ np.linalg.inv(S_mat)

            self.states.x_estimated[:, i_time_step] = x_pred + K_mat @ ytilde
            I_KH = np.eye(N_STATE_DIMS) - K_mat @ H_mat
            # Joseph stabilized covariance update
            self.states.cov_matrix[:, :, i_time_step] = (
                I_KH @ P_pred @ I_KH.T + K_mat @ R_mat @ K_mat.T
            )

        self.states.cov_trace[i_time_step] = float(
            np.trace(self.states.cov_matrix[:, :, i_time_step])
        )

        if self.draw_from_estimate:
            self.states.x_perceived[:, i_time_step] = (
                self.rng.multivariate_normal(
                    mean=self.states.x_estimated[:, i_time_step],
                    cov=self.states.cov_matrix[:, :, i_time_step],
                )
            )

        prev_lo = (
            self._initial_log_odds
            if i_time_step == 0
            else self.states.yield_log_odds[i_time_step - 1]
        )
        curr_lo = self.update_ehmi_log_odds(
            prev_lo,
            ehmi_signal=ehmi_signal,
            head_pitch_deg=head_pitch_deg,
            eccentricity_deg=eccentricity_deg,
        )
        self.states.yield_log_odds[i_time_step] = curr_lo
        self.states.yield_prob[i_time_step] = 1.0 / (1.0 + math.exp(-curr_lo))

    def update(
        self,
        i_time_step,
        ego_state,
        oth_state,
        head_pitch_deg=None,
        eccentricity_deg=None,
        gaze_mode=None,
        ehmi_signal=None,
    ):
        self.update_states(
            i_time_step,
            ego_state,
            oth_state,
            head_pitch_deg=head_pitch_deg,
            eccentricity_deg=eccentricity_deg,
            gaze_mode=gaze_mode,
            ehmi_signal=ehmi_signal,
        )
        self.perc_oth_state.yaw_angle = (
            0.0 if oth_state.yaw_angle is None else oth_state.yaw_angle
        )
        self.perc_oth_state.signed_CP_dist = self.states.x_perceived[
            0, i_time_step
        ]
        self.states.x_perceived[1, i_time_step] = max(
            0.0, self.states.x_perceived[1, i_time_step]
        )
        self.perc_oth_state.long_speed = self.states.x_perceived[
            1, i_time_step
        ]
        conflict_pt = getattr(self.simulation, 'conflict_point', None)
        if conflict_pt is None:
            conflict_pt = np.zeros(2)
        self.perc_oth_state.pos = (
            sc_scenario_helper.get_pos_from_signed_dist_to_conflict_pt(
                conflict_pt, self.perc_oth_state
            )
        )


@dataclass
class AcousticVehiclePerception(Perception):
    """Auditory Kalman localization model for blind/low-vision pedestrians,
    contrasting UN Regulation No. 138 AVAS-equipped vehicles
    (``vehicle_acoustic_mode="ev_avas"`` or ``"ice"``) against quiet EVs below
    ``20 km/h`` suffering a ``-12.5 dB`` acoustic emission deficit
    (``vehicle_acoustic_mode="ev_quiet"``).
    """

    kalman_filter: bool = True
    vehicle_acoustic_mode: str = "ev_avas"
    ambient_spl_dba: float = 44.0
    quiet_ev_deficit_db: float = 12.5
    avas_cutoff_speed_kmh: float = 20.0
    reference_dist_m: float = 2.0
    atm_attenuation_db_per_m: float = 0.005
    masking_threshold_snr_db: float = -8.0
    base_range_weber_stddev: float = 0.14
    base_speed_obs_stddev: float = 0.75
    spd_proc_noise_stddev: float = 0.30
    pos_proc_noise_stddev: float = 0.15

    def __post_init__(self):
        valid_modes = {"ev_avas", "ev_quiet", "ice"}
        if self.vehicle_acoustic_mode not in valid_modes:
            raise ValueError(
                f"vehicle_acoustic_mode must be one of {sorted(valid_modes)}, "
                f"got {self.vehicle_acoustic_mode!r}."
            )
        if self.prior is None:
            self.prior = KalmanPrior(
                cp_dist_mean=25.0,
                cp_dist_stddev=7.5,
                speed_mean=6.0,
                speed_stddev=3.0,
            )

        n_time_steps = self.simulation.settings.n_time_steps
        self.i_time_step = -1
        self.noisy_perception = True
        self.rng = default_rng(seed=self.noise_seed)

        self.states = PerceptionStates()
        self.states.x_true = np.full((N_STATE_DIMS, n_time_steps), np.nan)
        self.states.x_noisy = np.full((N_STATE_DIMS, n_time_steps), np.nan)
        self.states.x_estimated = np.full((N_STATE_DIMS, n_time_steps), np.nan)
        self.states.cov_matrix = np.full(
            (N_STATE_DIMS, N_STATE_DIMS, n_time_steps), np.nan
        )
        self.states.x_estimated[:, -1] = self.prior.x_estimated
        self.states.cov_matrix[:, :, -1] = self.prior.cov_matrix
        self.states.x_perceived = (
            np.full((N_STATE_DIMS, n_time_steps), np.nan)
            if self.draw_from_estimate
            else self.states.x_estimated
        )

        self.states.source_spl_dba = np.full(n_time_steps, np.nan)
        self.states.received_spl_dba = np.full(n_time_steps, np.nan)
        self.states.snr_db = np.full(n_time_steps, np.nan)
        self.states.acoustically_masked = np.zeros(n_time_steps, dtype=bool)
        self.states.cov_trace = np.full(n_time_steps, np.nan)

        self.perc_oth_state = commotions.KinematicState(
            pos=None, long_speed=None, yaw_angle=None
        )
        dt = self.simulation.settings.time_step
        self.Fmatrix = np.array(((1.0, -dt), (0.0, 1.0)))
        self.Qmatrix = np.array(
            (
                (self.pos_proc_noise_stddev ** 2, 0.0),
                (0.0, self.spd_proc_noise_stddev ** 2),
            )
        )

    def _resolve_signed_cp_dist(self, state):
        if hasattr(state, 'signed_CP_dist') and state.signed_CP_dist is not None:
            return float(state.signed_CP_dist)
        conflict_pt = getattr(self.simulation, 'conflict_point', None)
        if conflict_pt is None:
            conflict_pt = np.zeros(2)
        yaw = 0.0 if state.yaw_angle is None else float(state.yaw_angle)
        tmp = commotions.KinematicState(
            pos=np.asarray(state.pos, dtype=float),
            long_speed=float(state.long_speed),
            yaw_angle=yaw,
        )
        return float(
            sc_scenario_helper.get_signed_dist_to_conflict_pt(conflict_pt, tmp)
        )

    def compute_source_spl_dba(self, speed_mps: float, mode: str | None = None) -> float:
        """Compute A-weighted source SPL at ``reference_dist_m`` (2.0 m).

        Below ``20 km/h`` (``5.56 m/s``), UN R138 ``ev_avas`` maintains >= 50-56 dBA,
        whereas ``ev_quiet`` exhibits a ``-12.5 dB`` deficit relative to ``ev_avas``.
        Above ``20 km/h``, tire-road rolling noise dominates across propulsion modes.
        """
        eff_mode = self.vehicle_acoustic_mode if mode is None else mode
        v_kmh = max(0.0, float(speed_mps) * 3.6)

        if v_kmh >= self.avas_cutoff_speed_kmh:
            tire_spl = 56.0 + 30.0 * math.log10(v_kmh / self.avas_cutoff_speed_kmh)
            if eff_mode == "ice":
                return max(tire_spl, 58.5 + 8.0 * (v_kmh / self.avas_cutoff_speed_kmh))
            return tire_spl

        # Low-speed regime (< 20 km/h): UN R138 AVAS critical zone
        avas_spl = 50.0 + 6.0 * np.clip((v_kmh - 5.0) / 15.0, 0.0, 1.0)
        if eff_mode == "ev_avas":
            return float(avas_spl)
        if eff_mode == "ev_quiet":
            return float(avas_spl - self.quiet_ev_deficit_db)
        # ICE propulsion baseline
        return float(55.0 + 3.5 * (v_kmh / self.avas_cutoff_speed_kmh))

    def compute_auditory_snr_db(
        self, distance_m: float, speed_mps: float, mode: str | None = None
    ) -> tuple[float, float, float]:
        """Return ``(source_spl_dba, received_spl_dba, snr_db)`` at ``distance_m``."""
        src_spl = self.compute_source_spl_dba(speed_mps, mode=mode)
        eff_d = max(float(distance_m), self.reference_dist_m)
        spreading_loss = 20.0 * math.log10(eff_d / self.reference_dist_m)
        atm_loss = self.atm_attenuation_db_per_m * max(
            0.0, eff_d - self.reference_dist_m
        )
        recv_spl = src_spl - spreading_loss - atm_loss
        snr_db = recv_spl - self.ambient_spl_dba
        return src_spl, recv_spl, snr_db

    def update_states(self, i_time_step, ego_state, oth_state):
        assert i_time_step == self.i_time_step + 1
        self.i_time_step = i_time_step

        ego_cp_dist = self._resolve_signed_cp_dist(ego_state)
        oth_cp_dist = self._resolve_signed_cp_dist(oth_state)
        oth_speed = float(oth_state.long_speed)

        self.states.x_true[:, i_time_step] = (oth_cp_dist, oth_speed)
        euclid_d = math.hypot(ego_cp_dist, oth_cp_dist)
        src_spl, recv_spl, snr_db = self.compute_auditory_snr_db(euclid_d, oth_speed)

        self.states.source_spl_dba[i_time_step] = src_spl
        self.states.received_spl_dba[i_time_step] = recv_spl
        self.states.snr_db[i_time_step] = snr_db

        prev_x = self.states.x_estimated[:, i_time_step - 1]
        prev_P = self.states.cov_matrix[:, :, i_time_step - 1]

        x_pred = self.Fmatrix @ prev_x
        P_pred = self.Fmatrix @ prev_P @ self.Fmatrix.T + self.Qmatrix

        is_masked = snr_db < self.masking_threshold_snr_db
        self.states.acoustically_masked[i_time_step] = is_masked

        if is_masked:
            self.states.x_estimated[:, i_time_step] = x_pred
            self.states.cov_matrix[:, :, i_time_step] = P_pred
        else:
            snr_penalty = 1.0 + math.exp(-snr_db / 6.0)
            sigma_d = max(0.35, self.base_range_weber_stddev * max(euclid_d, 2.0) * snr_penalty)
            sigma_v = max(0.20, self.base_speed_obs_stddev * snr_penalty)
            R_mat = np.array(((sigma_d ** 2, 0.0), (0.0, sigma_v ** 2)))
            z_noisy = self.rng.multivariate_normal(
                mean=self.states.x_true[:, i_time_step], cov=R_mat
            )
            self.states.x_noisy[:, i_time_step] = z_noisy

            S_mat = P_pred + R_mat
            K_mat = P_pred @ np.linalg.inv(S_mat)
            self.states.x_estimated[:, i_time_step] = x_pred + K_mat @ (
                z_noisy - x_pred
            )
            I_K = np.eye(N_STATE_DIMS) - K_mat
            self.states.cov_matrix[:, :, i_time_step] = (
                I_K @ P_pred @ I_K.T + K_mat @ R_mat @ K_mat.T
            )

        self.states.cov_trace[i_time_step] = float(
            np.trace(self.states.cov_matrix[:, :, i_time_step])
        )
        if self.draw_from_estimate:
            self.states.x_perceived[:, i_time_step] = self.rng.multivariate_normal(
                mean=self.states.x_estimated[:, i_time_step],
                cov=self.states.cov_matrix[:, :, i_time_step],
            )

    def update(self, i_time_step, ego_state, oth_state):
        self.update_states(i_time_step, ego_state, oth_state)
        self.perc_oth_state.yaw_angle = (
            0.0 if oth_state.yaw_angle is None else oth_state.yaw_angle
        )
        self.perc_oth_state.signed_CP_dist = self.states.x_perceived[0, i_time_step]
        self.states.x_perceived[1, i_time_step] = max(
            0.0, self.states.x_perceived[1, i_time_step]
        )
        self.perc_oth_state.long_speed = self.states.x_perceived[1, i_time_step]
        conflict_pt = getattr(self.simulation, 'conflict_point', None)
        if conflict_pt is None:
            conflict_pt = np.zeros(2)
        self.perc_oth_state.pos = (
            sc_scenario_helper.get_pos_from_signed_dist_to_conflict_pt(
                conflict_pt, self.perc_oth_state
            )
        )
            
            
        

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    AGENT_NAMES = ('P', 'V')
    AGENT_EYE_HEIGHT = 1.7
    AGENT_POS0 = (np.array((0, -5)), np.array((40, 0)))
    AGENT_FREE_SPEED = (1.5, 10)
    GOALS = np.array(((0, 5), (-50, 0)))
    N_SCENARIOS = 2
    END_TIMES = (4, 8)
    INIT_KIN_STATES = ((commotions.KinematicState(pos=AGENT_POS0[0], 
                                                  long_speed=0, 
                                                  yaw_angle=None),
                        commotions.KinematicState(pos=AGENT_POS0[1], 
                                                  long_speed=AGENT_FREE_SPEED[1], 
                                                  yaw_angle=None)),
                       (commotions.KinematicState(pos=AGENT_POS0[0], 
                                                  long_speed=AGENT_FREE_SPEED[0], 
                                                  yaw_angle=None),
                        commotions.KinematicState(pos=AGENT_POS0[1], 
                                                  long_speed=AGENT_FREE_SPEED[1], 
                                                  yaw_angle=None)))
    NO_ACTION_STATE = commotions.ActionState()
    veh_stop_dist = AGENT_POS0[1][0] - 2
    veh_stop_acc = -AGENT_FREE_SPEED[1] ** 2 / (2 * veh_stop_dist)
    STOPPING_ACTION_STATE = commotions.ActionState(long_acc=veh_stop_acc)
    INIT_ACT_STATES = ((NO_ACTION_STATE, NO_ACTION_STATE),
                       (NO_ACTION_STATE, STOPPING_ACTION_STATE))
    CONFLICT_POINT = np.zeros(2)
    
    DIST_OBS_NOISE = 5 # m
    ANG_OBS_NOISE = 0.01 # rad
    PRIOR_STDDEV_MULT = 1 # factor by which to multiply the _POS0 and _FREE_SPEED values above to get initial stddevs
    PROC_NOISE_MULT = 0.01 # factor by which to multiply the _FREE_SPEED values above to get process noise for speed 
    
    def get_signed_dist_to_conflict_pt(pos, yaw_angle):
        vect_to_conflict_point = CONFLICT_POINT - pos
        heading_vect = np.array((math.cos(yaw_angle), math.sin(yaw_angle)))
        return np.dot(heading_vect, vect_to_conflict_point)
    
    # loop through scenarios
    for i_scenario in range(N_SCENARIOS):
    
        sim = commotions.Simulation(start_time=0, end_time=END_TIMES[i_scenario], 
                                    time_step=0.02)
        sim.conflict_point = CONFLICT_POINT
        
        # create agents and add them to the simulation
        for i_agent in range(2):
            agent = commotions.AgentWithGoal(AGENT_NAMES[i_agent], sim, 
                                             GOALS[i_agent, :], 
                                             INIT_KIN_STATES[i_scenario][i_agent],
                                             initial_action_state=
                                             INIT_ACT_STATES[i_scenario][i_agent])
        
        # loop through agents and add various perception objects to test
        for i_agent, agent in enumerate(sim.agents):
            i_oth_agent = 1-i_agent
            agent.other_agent = sim.agents[i_oth_agent]
            oth_dist_prior = get_signed_dist_to_conflict_pt(
                AGENT_POS0[i_oth_agent], agent.other_agent.trajectory.yaw_angle[0])
            oth_dist_stddev_prior = (PRIOR_STDDEV_MULT * np.amax(np.abs(AGENT_POS0[i_oth_agent])))
            oth_speed_stddev_prior = (PRIOR_STDDEV_MULT * AGENT_FREE_SPEED[i_oth_agent])
            prior = KalmanPrior(oth_dist_prior, oth_dist_stddev_prior, 
                                AGENT_FREE_SPEED[i_oth_agent], oth_speed_stddev_prior)
            proc_noise_stddev = PROC_NOISE_MULT * AGENT_FREE_SPEED[i_oth_agent]
            noise_seed = i_scenario * 10 + i_agent
            agent.perc = {}
            agent.perc['base'] = Perception(sim)
            agent.perc['noisy'] = Perception(sim, 
                                             pos_obs_noise_stddev = DIST_OBS_NOISE,
                                             noise_seed = noise_seed)
            agent.perc['noisy_ang'] = Perception(sim, 
                                                  angular_perception = True,
                                                  ego_eye_height = AGENT_EYE_HEIGHT,
                                                  pos_obs_noise_stddev = ANG_OBS_NOISE,
                                                  noise_seed = noise_seed)
            agent.perc['kalman'] = Perception(sim, 
                                              pos_obs_noise_stddev = DIST_OBS_NOISE,
                                              kalman_filter = True,
                                              prior = prior,
                                              spd_proc_noise_stddev = proc_noise_stddev,
                                              noise_seed = noise_seed)
            agent.perc['kalman_draw'] = Perception(sim, 
                                                  pos_obs_noise_stddev = DIST_OBS_NOISE,
                                                  kalman_filter = True,
                                                  prior = prior,
                                                  spd_proc_noise_stddev = proc_noise_stddev,
                                                  draw_from_estimate = True,
                                                  noise_seed = noise_seed)
            agent.perc['kalman_ang'] = Perception(sim, 
                                                  angular_perception = True,
                                                  ego_eye_height = AGENT_EYE_HEIGHT,
                                                  pos_obs_noise_stddev = ANG_OBS_NOISE,
                                                  kalman_filter = True,
                                                  prior = prior,
                                                  spd_proc_noise_stddev = proc_noise_stddev,
                                                  noise_seed = noise_seed)
            agent.perc['kalman_ang_draw'] = Perception(sim, 
                                                      angular_perception = True,
                                                      ego_eye_height = AGENT_EYE_HEIGHT,
                                                      pos_obs_noise_stddev = ANG_OBS_NOISE,
                                                      kalman_filter = True,
                                                      prior = prior,
                                                      spd_proc_noise_stddev = proc_noise_stddev,
                                                      draw_from_estimate = True,
                                                      noise_seed = noise_seed)
            
        # run the actual simulation
        sim.run()
        
        # go through the simulation time steps, and perform perception updates
        for i_time_step in range(sim.settings.n_time_steps):
            for agent in sim.agents:
                agent.curr_state = agent.get_kinematic_state(i_time_step)
                agent.curr_state.signed_CP_dist = get_signed_dist_to_conflict_pt(
                    agent.curr_state.pos, agent.curr_state.yaw_angle)
            for agent in sim.agents:
                for perc in agent.perc.values():
                    perc.update(i_time_step, agent.curr_state, 
                                agent.other_agent.curr_state)
                    
        # plot results
        ylabel = ('Distance (m)', 'Speed (m/s)', 'TTA (s)')
        for i_agent, agent in enumerate(sim.agents):
            fig_name = (f'Scenario {i_scenario}, agent {agent.name}'
                        f' (observing {agent.other_agent.name})')
            fig = plt.figure(fig_name, figsize=(16, 7))
            fig.clf()
            axs = fig.subplots(N_STATE_DIMS+1, len(agent.perc))
            for i_perc, perc_key in enumerate(agent.perc.keys()):
                perc = agent.perc[perc_key]
                for i_dim in range(N_STATE_DIMS+1):
                    ax = axs[i_dim, i_perc]
                    if i_dim == 0:
                        ax.set_title(perc_key)
                    if i_dim <= 1:
                        y_true = perc.states.x_true[i_dim, :]
                        y_noisy = perc.states.x_noisy[i_dim, :]
                        if perc.kalman_filter:
                            y_estimated = perc.states.x_estimated[i_dim, :]
                        y_perceived = perc.states.x_perceived[i_dim, :]
                    else:
                        y_true = (perc.states.x_true[0, :]
                                  / perc.states.x_true[1, :])
                        y_noisy = (perc.states.x_noisy[0, :]
                                   /perc.states.x_noisy[1, :])
                        if perc.kalman_filter:
                            y_estimated = (perc.states.x_estimated[0, :]
                                           / perc.states.x_estimated[1, :])
                        y_perceived = (perc.states.x_perceived[0, :]
                                       / perc.states.x_perceived[1, :])
                    ax.plot(sim.time_stamps, y_true, lw=4, color='lightgreen')
                    if perc.noisy_perception:
                        ax.plot(sim.time_stamps, y_noisy, 
                                lw=3, color='red', alpha=0.2)
                    if perc.kalman_filter:
                        ax.plot(sim.time_stamps, y_estimated, 
                            lw=2, color='gray', alpha=0.5)
                        if i_dim <= 1:
                            for side in (-1, 1):
                                ax.plot(sim.time_stamps, y_estimated
                                        + side * np.sqrt(perc.states.cov_matrix[
                                            i_dim, i_dim, :]), '--',
                                        lw=2, color='gray', alpha=0.5)
                    ax.plot(sim.time_stamps, y_perceived, lw=1, color='k')
                    true_max = np.amax(y_true)
                    true_min = np.amin(y_true)
                    if math.isinf(true_max) or math.isinf(true_min):
                        ax.set_ylim(-100, 100)
                    else:
                        ax.set_ylim(true_min - 5, true_max + 5)
                    if i_perc == 0:
                        ax.set_ylabel(ylabel[i_dim])
                    if i_dim == N_STATE_DIMS:
                        ax.set_xlabel('Time (s)')
            plt.tight_layout()
