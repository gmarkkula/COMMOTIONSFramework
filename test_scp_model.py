import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.special
from scipy.stats import norm
import copy
import commotions

matplotlib.use('qt4agg')

N_AGENTS = 2 # this implementation supports only 2
AGENT_NAMES = ('A', 'B')
PLOT_COLORS = ('c', 'm')
INITIAL_POSITIONS = np.array([[0,-5.5], [5, 0]])
GOALS = np.array([[0, 5], [-5, 0]])

START_TIME = 0
END_TIME = 10
TIME_STEP = 0.1

BEHAVIORS = ('const.', 'proc.', 'yield')
N_BEHAVIORS = len(BEHAVIORS)
i_CONSTANT = 0
i_PROCEEDING = 1
i_YIELDING = 2
MIN_ADAPT_ACC = -10 # m/s^2

DEFAULT_PARAMS = commotions.Parameters()
DEFAULT_PARAMS.k_g = 1 
DEFAULT_PARAMS.k_c = 1
DEFAULT_PARAMS.k_a = 0
DEFAULT_PARAMS.k_v = 0.3
DEFAULT_PARAMS.alpha = 0.9
DEFAULT_PARAMS.beta = .5
DEFAULT_PARAMS.gamma = DEFAULT_PARAMS.alpha
DEFAULT_PARAMS.kappa = DEFAULT_PARAMS.alpha
DEFAULT_PARAMS.Lambda = 1
DEFAULT_PARAMS.Sigma = .05
DEFAULT_PARAMS.DeltaV_th = 0.1
DEFAULT_PARAMS.DeltaT = 0.3
DEFAULT_PARAMS.v_free = DEFAULT_PARAMS.k_g / (2 * DEFAULT_PARAMS.k_v) # speed at which value is maximum, if heading toward goal, and no obstacles
DEFAULT_PARAMS.deltavs = np.array([-1, -0.5, 0, 0.5, 1]) # available speed change actions, magnitudes in m/s
i_NO_ACTION = 2
N_ACTIONS = len(DEFAULT_PARAMS.deltavs)

SHARED_PARAMS = commotions.Parameters()
SHARED_PARAMS.T_P = 0.3 # prediction time
SHARED_PARAMS.d_C = 1 # collision distance
SHARED_PARAMS.n_prediction_time_steps = math.ceil(SHARED_PARAMS.T_P / TIME_STEP)

class States():
    pass

class SCPAgent(commotions.AgentWithGoal):

    def prepare_for_simulation(self):
        # make sure this agent isn't used for a simulation with more than two
        # agents
        assert(len(self.simulation.agents) == 2)
        # store a reference to the other agent
        for agent in self.simulation.agents:
            if agent is not self:
                self.other_agent = agent
        # allocate vectors for storing internal states
        self.n_actions = len(self.params.deltavs)
        n_time_steps = self.simulation.settings.n_time_steps
        self.states = States()
        # - states regarding my own actions
        self.states.mom_action_vals = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # V_a(t)
        self.states.est_action_vals = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # Vhat_a(t)
        self.states.est_action_surplus_vals = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # DeltaVhat_a(t)
        self.states.action_vals_given_behs = \
            math.nan * np.ones((self.n_actions, N_BEHAVIORS, n_time_steps)) # V_a|b(t)
        self.states.action_probs = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # P_a(t)
        # - states regarding the behavior of the other agent
        self.states.beh_activations = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # A_b(t)
        self.states.beh_activ_V = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # ^V A_b(t)
        self.states.beh_activ_O = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # ^O A_b(t)
        self.states.beh_probs = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # P_b(t)
        self.states.beh_vals_given_actions = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps)) # V_b|a(t)
        self.states.sensory_probs_given_behs = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # P_{x_o|b}(t)
        self.states.beh_long_accs = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # the acceleration that the other agent should be applying right now if doing behaviour b
        # - other states
        self.states.time_left_to_CA_entry = math.nan * np.ones(n_time_steps)
        self.states.time_left_to_CA_exit = math.nan * np.ones(n_time_steps)
        # set initial values for states that depend on the previous time step
        self.states.est_action_vals[:, -1] = 0
        self.states.beh_activ_V[:, -1] = 0
        self.states.beh_activ_O[:, -1] = 0
        # calculate where the two agents' paths intersect, if it has not already 
        # been done
        if not hasattr(self.simulation, 'conflict_point'):
            self.simulation.conflict_point = \
                commotions.get_intersection_of_lines(\
                    self.get_current_kinematic_state().pos, self.goal, \
                    self.other_agent.get_current_kinematic_state().pos, self.other_agent.goal)


    def get_signed_dist_to_conflict_pt(self, state):
        """Get the signed distance from the specified agent state to the conflict
        point. Positive sign means that the agent is on its way toward the conflict
        point. Does not take speed into account - i.e. does not check for backward
        traveling."""
        vect_to_conflict_point = \
            self.simulation.conflict_point - state.pos
        heading_vect = \
            np.array((math.cos(state.yaw_angle), math.sin(state.yaw_angle)))
        return np.dot(heading_vect, vect_to_conflict_point)


    def do_action_update(self):
        """Do the action update for the agent. 
        """

        i_time_step = self.simulation.state.i_time_step
        time_step = self.simulation.settings.time_step

        # calculate my own current projected time until entering and
        # exiting the conflict area
        curr_state = self.get_current_kinematic_state()
        proj_long_speeds = curr_state.long_speed \
            + np.cumsum(self.action_long_accs[i_time_step:] * time_step)
        signed_dist_to_confl_pt = self.get_signed_dist_to_conflict_pt(curr_state)
        proj_signed_dist_to_CP = signed_dist_to_confl_pt \
            - np.cumsum(proj_long_speeds * time_step)
        # - entry
        i_time_steps_entered = \
            np.nonzero(proj_signed_dist_to_CP < SHARED_PARAMS.d_C)[0]
        if len(i_time_steps_entered) == 0:
            # not currently projected to e enter he conflict area within the simulation duration
            self.states.time_left_to_CA_entry[i_time_step] = math.inf
        else:
            self.states.time_left_to_CA_entry[i_time_step] = \
                i_time_steps_entered[0] * time_step
        # - exit
        i_time_steps_exited = \
            np.nonzero(proj_signed_dist_to_CP < -SHARED_PARAMS.d_C)[0]
        if len(i_time_steps_exited) == 0:
            # not currently projected to exit the conflict area within the simulation duration
            self.states.time_left_to_CA_exit[i_time_step] = math.inf
        else:
            self.states.time_left_to_CA_exit[i_time_step] = \
                i_time_steps_exited[0] * time_step

        # calculate the accelerations needed for the different behaviors of the 
        # other agent, as of the current time step
        oth_state = self.other_agent.get_current_kinematic_state()
        # - constant behavior
        self.states.beh_long_accs[i_CONSTANT, i_time_step] = 0
        # - proceeding behavior (assuming straight acc to free speed; i.e., disregarding acceleration cost)
        self.states.beh_long_accs[i_PROCEEDING, i_time_step] = \
            (self.params.v_free - oth_state.long_speed) / self.params.DeltaT
        # - yielding behavior 
        oth_signed_dist_to_confl_pt = self.get_signed_dist_to_conflict_pt(oth_state)
        oth_signed_dist_to_CA_entry = \
            oth_signed_dist_to_confl_pt - SHARED_PARAMS.d_C
        use_stop_acc = False
        if oth_signed_dist_to_CA_entry > 0:
            stop_acc = - oth_state.long_speed ** 2 / (2 * oth_signed_dist_to_CA_entry)
            t_stop = - oth_state.long_speed / stop_acc
            if self.states.time_left_to_CA_entry[i_time_step] >= t_stop:
                use_stop_acc = True
            else:
                adapt_time_to_CA_entry = \
                    self.states.time_left_to_CA_exit[i_time_step]
        else:
            adapt_time_to_CA_entry = \
                self.states.time_left_to_CA_entry[i_time_step]
        if use_stop_acc:
            # acceleration to come to full stop before conflict area
            self.states.beh_long_accs[i_YIELDING, i_time_step] = stop_acc
        else:
            # acceleration to "adapt" - to reach CA entrance at the same time 
            # as I am exiting it, or before I am entering it
            # if other agent already past CA entrance
            if adapt_time_to_CA_entry == math.inf:
                adapt_acc = 0
            else:
                if adapt_time_to_CA_entry <= 0:
                    adapt_acc = MIN_ADAPT_ACC
                else:
                    adapt_acc = \
                        2 * (oth_signed_dist_to_CA_entry \
                        - oth_state.long_speed * adapt_time_to_CA_entry) \
                        / adapt_time_to_CA_entry ** 2
                # do not allow positive or overly large negative adaptation 
                # accelerations
                adapt_acc = max(MIN_ADAPT_ACC, min(0, adapt_acc))
            self.states.beh_long_accs[i_YIELDING, i_time_step] = adapt_acc
            

        # do first loops over all own actions and behaviors of the other
        # agent, and get the predicted states
        pred_own_states = []
        for i_action in range(self.n_actions):
            # get predicted own state with this action
            pred_own_states.append(self.get_predicted_own_state(i_action))
        pred_oth_states = []
        for i_beh in range(N_BEHAVIORS):
            # get predicted state of other agent with this behavior
            pred_oth_states.append(self.get_predicted_other_state(i_beh))

        # now loop over all combinations of own actions and other's behaviors, 
        # and get values from both agents' perspectives
        for i_action in range(self.n_actions):
            for i_beh in range(N_BEHAVIORS):
                # get value for me of this action/behavior combination
                self.states.action_vals_given_behs[i_action, i_beh, i_time_step] = \
                    self.get_value_for_me(pred_own_states[i_action], \
                        pred_oth_states[i_beh], i_action)
                # get value for the other agent of this action/behavior combination
                self.states.beh_vals_given_actions[i_beh, i_action, i_time_step] = \
                    self.get_value_for_other(pred_oth_states[i_beh], \
                        pred_own_states[i_action], i_beh)

        # get my estimated probabilities for my own actions - based on value 
        # estimates from the previous time step
        self.states.action_probs[:, i_time_step] = scipy.special.softmax(\
            self.params.Lambda * self.states.est_action_vals[:, i_time_step-1])

        # now loop over the other agent's behaviors, to update the corresponding
        # activations (my "belief" in these behaviors)
        for i_beh in range(N_BEHAVIORS):
            # update the game theoretic activations
            # - contribution from previous time step
            self.states.beh_activ_V[i_beh, i_time_step] = \
                self.params.gamma * \
                self.states.beh_activ_V[i_beh, i_time_step-1]
            # - contributions from estimated value of the behavior to the other
            #   agent, given my estimated probabilities of my actions
            for i_action in range(self.n_actions):
                self.states.beh_activ_V[i_beh, i_time_step] += \
                    (1 - self.params.gamma) * \
                    self.states.action_probs[i_action, i_time_step] \
                    * self.states.beh_vals_given_actions[i_beh, i_action, i_time_step]
            # update the "Kalman filter" activations
            # - get the expected state of the other agent in this time step,
            #   given the state in the last time step, and this behavior
            self.states.beh_activ_O[i_beh, i_time_step] = \
                self.params.kappa * self.states.beh_activ_O[i_beh, i_time_step-1]
            if i_time_step > 0:
                self.states.sensory_probs_given_behs[i_beh, i_time_step] = \
                    self.get_prob_of_current_state_given_beh(i_beh)
                self.states.beh_activ_O[i_beh, i_time_step] += \
                    (1 - self.params.kappa) * (1/self.params.Lambda) \
                    * math.log(self.states.sensory_probs_given_behs[i_beh, i_time_step])

        # get total activation for all behaviors of the other agent
        self.states.beh_activations[:, i_time_step] = \
            self.params.beta * self.states.beh_activ_V[:, i_time_step] \
            + self.states.beh_activ_O[:, i_time_step] 

        # get my estimated probabilities for the other agent's behavior
        self.states.beh_probs[:, i_time_step] = scipy.special.softmax(\
            self.params.Lambda * self.states.beh_activations[:, i_time_step])

        # loop through own action options and get momentary estimates
        # of the actions' values to me, as weighted average over the other 
        # agent's behaviors 
        self.states.mom_action_vals[:, i_time_step] = 0
        for i_action in range(self.n_actions):
            for i_beh in range(N_BEHAVIORS):
                self.states.mom_action_vals[i_action, i_time_step] += \
                    self.states.beh_probs[i_beh, i_time_step] \
                    * self.states.action_vals_given_behs[i_action, i_beh, i_time_step]

        # update the accumulative estimates of action value
        self.states.est_action_vals[:, i_time_step] = \
            self.params.alpha * self.states.est_action_vals[:, i_time_step-1] \
            + (1 - self.params.alpha) * self.states.mom_action_vals[:, i_time_step]

        # any action over threshold?
        self.states.est_action_surplus_vals[:, i_time_step] = \
            self.states.est_action_vals[:, i_time_step] \
            - self.states.est_action_vals[i_NO_ACTION, i_time_step]
        i_best_action = np.argmax(self.states.est_action_surplus_vals[:, i_time_step])
        if self.states.est_action_surplus_vals[i_best_action, i_time_step] \
            > self.params.DeltaV_th:
            acceleration_value = self.params.deltavs[i_best_action] \
                / self.params.DeltaT
            commotions.add_uniform_action_to_array(\
                self.action_long_accs, acceleration_value, \
                self.n_action_time_steps, self.simulation.state.i_time_step)
            self.states.est_action_vals[:, i_time_step] = 0

        # set long acc in actual trajectory
        self.trajectory.long_acc[i_time_step] = self.action_long_accs[i_time_step]
        

    def get_value_of_state_for_agent(self, own_state, own_goal, oth_state):

        heading_vector = np.array([math.cos(own_state.yaw_angle), \
            math.sin(own_state.yaw_angle)])
        
        # reward for progress toward goal and cost for speed
        vector_to_goal = own_goal - own_state.pos
        heading_to_goal_vector = vector_to_goal / np.linalg.norm(vector_to_goal)
        heading_toward_goal_component = \
            np.dot(heading_vector, heading_to_goal_vector)
        goal_distance_change_rate = \
            -heading_toward_goal_component * own_state.long_speed
        value = -self.params.k_g * goal_distance_change_rate \
            - self.params.k_v * own_state.long_speed ** 2 

        # cost for being on collision course with the other agent
        time_to_agent_collision = \
            commotions.get_time_to_agent_collision(\
                own_state, oth_state, SHARED_PARAMS.d_C)
        if time_to_agent_collision == 0:
            value = -math.inf
        elif time_to_agent_collision < math.inf:
            value += -self.params.k_c / time_to_agent_collision  
        
        return value


    def get_value_for_me(self, my_state, oth_state, i_action):            
        # cost for making a speed change, if any
        value = -self.params.k_a * self.params.deltavs[i_action] ** 2
        # add value of the state
        value += self.get_value_of_state_for_agent(\
            my_state, self.goal, oth_state)
        return value


    def get_value_for_other(self, oth_state, my_state, i_beh):
        # POSSIBLE TODO - add cost for the behavior itself
        # - disregarding acceleration discomfort for the other agent, for now at least
        #   (needs a bit of code restructuring, or storing the cost somewhere)
        value = 0
        # add value of the state
        value += self.get_value_of_state_for_agent(oth_state, \
            self.other_agent.goal, my_state) # assuming symmetric cost function
        return value


    def get_predicted_own_state(self, i_action):
        local_long_accs = np.copy(self.action_long_accs)
        acceleration_value = self.params.deltavs[i_action] / self.params.DeltaT
        commotions.add_uniform_action_to_array(\
            local_long_accs, acceleration_value, \
            self.n_action_time_steps, self.simulation.state.i_time_step)
        predicted_state = self.get_future_kinematic_state(\
            local_long_accs, yaw_rate = 0, \
            n_time_steps_to_advance = SHARED_PARAMS.n_prediction_time_steps)
        return predicted_state


    def get_predicted_other_state(self, i_beh):
        # get the longitudinal acceleration for this behavior, if implemented at
        # the current time step
        long_acc_for_this_beh = \
            self.states.beh_long_accs[i_beh, self.simulation.state.i_time_step]
        # let the other agent object calculate what its predicted state would
        # be with this acceleration 
        predicted_state = self.other_agent.get_future_kinematic_state(\
            long_acc_for_this_beh, yaw_rate = 0, \
            n_time_steps_to_advance = SHARED_PARAMS.n_prediction_time_steps)
        return predicted_state
        

    def get_prob_of_current_state_given_beh(self, i_beh):
        # retrieve the longitudinal acceleration for this behavior, as estimated on
        # the previous time step
        i_prev_time_step = self.simulation.state.i_time_step-1
        prev_long_acc_for_this_beh = \
            self.states.beh_long_accs[i_beh, i_prev_time_step]
        # let the other agent object calculate what its predicted state at the 
        # current time step would be with this acceleration     
        expected_curr_state = self.other_agent.get_future_kinematic_state(\
            prev_long_acc_for_this_beh, yaw_rate = 0, \
            n_time_steps_to_advance = 1, i_start_time_step = i_prev_time_step)
        # get the distance between expected and observed position
        pos_diff = np.linalg.norm(expected_curr_state.pos \
            - self.other_agent.get_current_kinematic_state().pos)
        # return the probability density for this observed difference
        prob_density = norm.pdf(pos_diff, scale = self.params.Sigma)
        return max(prob_density, np.finfo(float).eps) # don't return zero probability
        


    def __init__(self, simulation, i_agent):

        # set initial state and call inherited init method
        initial_state = commotions.KinematicState(\
            pos = INITIAL_POSITIONS[i_agent,:], yaw_angle = None)
        super().__init__(AGENT_NAMES[i_agent], simulation, GOALS[i_agent,:], \
            initial_state, plot_color = PLOT_COLORS[i_agent])

        # make and store a copy of the default parameters object
        self.params = copy.copy(DEFAULT_PARAMS)

        # POSSIBLE TODO: absorb this into a new class 
        #                commotions.AgentWithIntermittentActions or similar
        # store some derived constants
        self.n_action_time_steps = math.ceil(
            self.params.DeltaT / self.simulation.settings.time_step)
        self.n_actions_vector_length = \
            self.simulation.settings.n_time_steps + \
            SHARED_PARAMS.n_prediction_time_steps
        # prepare vectors for storing long acc and yaw rates, incl lookahead
        # with added actions
        self.action_long_accs = np.zeros(self.n_actions_vector_length)
        self.action_yaw_rates = np.zeros(self.n_actions_vector_length) # yaw rate should really always remain zero in this class - but leaving this vector here anyway


####################

# create the simulation and agents in it
scp_simulation = commotions.Simulation(START_TIME, END_TIME, TIME_STEP)
for i_agent in range(N_AGENTS):
    SCPAgent(scp_simulation, i_agent)

# run the simulation
scp_simulation.run()

# plot trajectories
plt.figure('Trajectories')
scp_simulation.plot_trajectories()
plt.legend()

# plot agent states
# - action values given behaviors
plt.figure('Action values given behaviours', figsize = (10.0, 10.0))
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_action, deltav in enumerate(DEFAULT_PARAMS.deltavs):
        plt.subplot(N_ACTIONS, N_AGENTS, i_action * N_AGENTS +  i_agent + 1)
        plt.ylim(-2, 2)
        for i_beh in range(N_BEHAVIORS):
            plt.plot(scp_simulation.time_stamps, agent.states.action_vals_given_behs[i_action, i_beh, :])
        if i_action == 0:
            plt.title('Agent %s' % agent.name)
            if i_agent == 1:
                plt.legend(BEHAVIORS)
        if i_agent == 0:
            plt.ylabel('$V(\\Delta v=%.1f | b)$' % deltav)

# - action probabilities
plt.figure('Action probabilities', figsize = (10.0, 10.0))
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_action, deltav in enumerate(DEFAULT_PARAMS.deltavs):
        plt.subplot(N_ACTIONS, N_AGENTS, i_action * N_AGENTS +  i_agent + 1)
        plt.plot(scp_simulation.time_stamps, agent.states.action_probs[i_action, :])
        plt.ylim(-.1, 1.1)
        if i_action == 0:
            plt.title('Agent %s' % agent.name)
        if i_agent == 0:
            plt.ylabel('$P(\\Delta v=%.1f)$' % deltav)

# - momentary and accumulative estimates of action values
plt.figure('Action value estimates', figsize = (10.0, 10.0))
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_action, deltav in enumerate(DEFAULT_PARAMS.deltavs):
        plt.subplot(N_ACTIONS, N_AGENTS, i_action * N_AGENTS +  i_agent + 1)
        plt.plot(scp_simulation.time_stamps, agent.states.mom_action_vals[i_action, :])
        plt.plot(scp_simulation.time_stamps, agent.states.est_action_vals[i_action, :])
        plt.ylim(-2, 2)
        if i_action == 0:
            plt.title('Agent %s' % agent.name)
            if i_agent == 1:
                plt.legend(('$\\tilde{V}_a$', '$\\hat{V}_a$'))
        if i_agent == 0:
            plt.ylabel('$V(\\Delta v=%.1f)$' % deltav)

# - surplus action values
plt.figure('Surplus action value estimates', figsize = (10.0, 10.0))
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_action, deltav in enumerate(DEFAULT_PARAMS.deltavs):
        plt.subplot(N_ACTIONS, N_AGENTS, i_action * N_AGENTS +  i_agent + 1)
        plt.plot(scp_simulation.time_stamps, agent.states.est_action_surplus_vals[i_action, :])
        plt.plot([scp_simulation.time_stamps[0], scp_simulation.time_stamps[-1]], \
            [agent.params.DeltaV_th, agent.params.DeltaV_th] , color = 'gray')
        plt.ylim(-.5, .3)
        if i_action == 0:
            plt.title('Agent %s' % agent.name)
        if i_agent == 0:
            plt.ylabel('$\\Delta V(\\Delta v=%.1f)$' % deltav)

# - behavior activations
plt.figure('Behaviour activations')
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_beh in range(N_BEHAVIORS):
        plt.subplot(N_BEHAVIORS, N_AGENTS, i_beh * N_AGENTS +  i_agent + 1)
        plt.plot(scp_simulation.time_stamps, agent.states.beh_activ_V[i_beh, :])
        plt.plot(scp_simulation.time_stamps, agent.states.beh_activ_O[i_beh, :])
        plt.plot(scp_simulation.time_stamps, agent.states.beh_activations[i_beh, :])
        plt.ylim(-.1, 3)
        if i_beh == 0:
            plt.title('Agent %s (observing %s)' % (agent.name, agent.other_agent.name))
            if i_agent == 1:
                plt.legend(('$^G A$', '$^K A$', '$A$'))
        if i_agent == 0:
            plt.ylabel('$A(%s)$' % BEHAVIORS[i_beh])

# - expected vs observed accelerations for behaviors
plt.figure('Expected vs observed accelerations for behaviors', figsize = (10.0, 10.0))
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_beh in range(N_BEHAVIORS):
        plt.subplot(N_BEHAVIORS, N_AGENTS, i_beh * N_AGENTS +  i_agent + 1)
        plt.plot(scp_simulation.time_stamps, agent.states.beh_long_accs[i_beh, :], \
            '--', color = 'gray', linewidth = 2)
        plt.plot(scp_simulation.time_stamps, agent.other_agent.trajectory.long_acc)
        plt.ylim(-4, 4)
        if i_beh == 0:
            plt.title('Agent %s (observing %s)' % (agent.name, agent.other_agent.name))
            if i_agent == 1:
                plt.legend(('expected', 'observed'))
        if i_agent == 0:
            plt.ylabel('%s a (m/s^2)' % BEHAVIORS[i_beh])

# - behavior probabilities
plt.figure('Behaviour probabilities')
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_beh in range(N_BEHAVIORS):
        # plt.subplot(N_BEHAVIORS, N_AGENTS, i_beh * N_AGENTS +  i_agent + 1)
        plt.subplot(1, N_AGENTS, i_agent + 1)
        plt.plot(scp_simulation.time_stamps, agent.states.beh_probs[i_beh, :])
        plt.ylim(-.1, 1.1)
        plt.title('Agent %s (observing %s)' % (agent.name, agent.other_agent.name))
    if i_agent == 0:
        plt.ylabel('P (-)')
    else:
        plt.legend(BEHAVIORS)
            


# - sensory probability densities
plt.figure('Sensory probability densities')
for i_agent, agent in enumerate(scp_simulation.agents):
    for i_beh in range(N_BEHAVIORS):
        plt.subplot(N_BEHAVIORS, N_AGENTS, i_beh * N_AGENTS +  i_agent + 1)
        plt.plot(scp_simulation.time_stamps, \
            np.log(agent.states.sensory_probs_given_behs[i_beh, :]))
        if i_beh == 0:
            plt.title('Agent %s' % agent.name)
        if i_agent == 0:
            plt.ylabel('$log p(O|%s)$' % BEHAVIORS[i_beh])    


# - kinematic/action states
plt.figure('Kinematic and action states')
N_PLOTROWS = 3
for i_agent, agent in enumerate(scp_simulation.agents):
    # position
    plt.subplot(N_PLOTROWS, N_AGENTS, 0 * N_AGENTS +  i_agent + 1)
    plt.plot(scp_simulation.time_stamps, agent.trajectory.pos.T)
    plt.title('Agent %s' % agent.name)
    if i_agent == 0:
        plt.ylabel('pos (m)')
    else:
        plt.legend(('x', 'y'))
    # speed
    plt.subplot(N_PLOTROWS, N_AGENTS, 1 * N_AGENTS +  i_agent + 1)
    plt.plot(scp_simulation.time_stamps, agent.trajectory.long_speed)
    plt.ylim(-1, 2)
    if i_agent == 0:
        plt.ylabel('v (m/s)')
    # speed
    plt.subplot(N_PLOTROWS, N_AGENTS, 2 * N_AGENTS +  i_agent + 1)
    plt.plot(scp_simulation.time_stamps, agent.trajectory.long_acc)
    plt.ylim(-4, 4)
    if i_agent == 0:
        plt.ylabel('a (m/s^2)')


# - time left to conflict area entry/exit
plt.figure('Time left to conflict area')
for i_agent, agent in enumerate(scp_simulation.agents):
    plt.subplot(1, N_AGENTS, i_agent + 1)
    plt.plot(scp_simulation.time_stamps, agent.states.time_left_to_CA_entry)
    plt.plot(scp_simulation.time_stamps, agent.states.time_left_to_CA_exit)
    plt.title('Agent %s' % agent.name)
    if i_agent == 0:
        plt.ylabel('Time left (s)')
    else:
        plt.legend(('To CA entry', 'To CA exit'))
    

    # self.states.beh_vals_given_actions = \
    #     math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps))
    # self.states.sensory_probs_given_behs = \
    #     math.nan * np.ones((N_BEHAVIORS, n_time_steps))
    # self.states.beh_long_accs = \
    #     math.nan * np.ones((N_BEHAVIORS, n_time_steps))
    # # - other states
    # self.states.time_left_to_CA_entry = math.nan * np.ones(n_time_steps)
    # self.states.time_left_to_CA_exit = math.nan * np.ones(n_time_steps)

plt.show()









        

