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