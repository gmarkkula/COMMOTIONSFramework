import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import copy
import commotions

N_AGENTS = 2 # this implementation supports only 2
AGENT_NAMES = ('A', 'B')
PLOT_COLORS = ('c', 'm')
INITIAL_POSITIONS = np.array([[0,-5], [5, 0]])
GOALS = np.array([[0, 5], [-5, 0]])

START_TIME = 0
END_TIME = 10
TIME_STEP = 0.1

BEHAVIORS = ('proceed', 'yield')
N_BEHAVIORS = len(BEHAVIORS)

DEFAULT_PARAMS = commotions.Parameters()
DEFAULT_PARAMS.k_g = 1
DEFAULT_PARAMS.k_c = 0.3
DEFAULT_PARAMS.k_a = 0.3
DEFAULT_PARAMS.k_v = 0.3
DEFAULT_PARAMS.alpha = 0.9
DEFAULT_PARAMS.beta = 1
DEFAULT_PARAMS.gamma = DEFAULT_PARAMS.alpha
DEFAULT_PARAMS.Lambda = 1
DEFAULT_PARAMS.DeltaT = 0.3
DEFAULT_PARAMS.deltavs = np.array([-1, -0.5, 0, 0.5, 1]) # m/s
i_NO_ACTION = 2

SHARED_PARAMS = commotions.Parameters()
SHARED_PARAMS.T_P = 0.3 # prediction time
SHARED_PARAMS.d_C = 1 # collision distance

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
            math.nan * np.ones((self.n_actions, n_time_steps)) # Vtilde_a(t)
        self.states.est_action_vals = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # Vhat_a(t)
        self.states.action_vals_given_behs = \
            math.nan * np.ones((self.n_actions, N_BEHAVIORS, n_time_steps)) # V_a|b(t)
        self.states.action_probs = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # P_a(t)
        # - states regarding the behavior of the other agent
        self.states.beh_activations = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps))
        self.states.beh_activ_G = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps))
        self.states.beh_activ_K = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps))
        self.states.beh_probs = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps))
        self.states.beh_vals_given_actions = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps))


    def prepare_for_action_update(self):
        """Make any preparations needed by this agent and/or other agents in 
        the simulation, before making the action update. 
        """
        pass

    def do_action_update(self):
        """Do the action update for the agent. 
        """

        i_time_step = self.simulation.state.i_time_step

        # do first loop over all combinatinos of own actions and the other 
        # agent's behaviors, and get the values of the predicted states, both 
        # for this agent and the other agent
        for i_action, deltav in enumerate(self.params.deltavs):
            # get predicted own state with this action
            pred_own_state = self.get_predicted_own_state(i_action)
            # add expected value of action as weighted average over behaviors 
            # of the other agent
            for i_beh in range(N_BEHAVIORS):
                # get predicted state of other agent with this behavior
                pred_oth_state = self.get_predicted_other_state(i_beh)
                # get value for me of this action/behavior combination
                self.states.action_vals_given_behs[i_action, i_beh, i_time_step] = \
                    self.get_value_for_me(pred_own_state, pred_oth_state, i_action)
                # get value for the other agent of this action/behavior combination
                self.states.beh_vals_given_actions[i_beh, i_action, i_time_step] = \
                    self.get_value_for_other(pred_oth_state, pred_own_state, i_beh)

        # get my estimated probabilities for my own actions - based on value 
        # estimates from the previouw time step
        self.states.action_probs[:, i_time_step] = scipy.special.softmax(\
            self.params.Lambda * self.states.est_action_vals[:, i_time_step-1])

        # now loop over the other agent's behaviors, to update the corresponding
        # activations (my "belief" in these behaviors)
        for i_beh in range(N_BEHAVIORS):
            # update the game theoretic activations
            # - contribution from previous time step
            self.states.beh_activ_G[i_beh, i_time_step] = \
                self.params.gamma * self.states.beh_activ_G[i_beh, i_time_step-1]
            # - contributions from estimated value of the behavior to the other
            #   agent, given my estimated probabilities of my actions
            for i_action in range(self.n_actions):
                self.states.beh_activ_G[i_beh, i_time_step] += \
                    (1 - self.params.gamma) \
                    * self.states.action_probs[i_action, i_time_step] \
                    * self.states.beh_vals_given_actions[i_beh, i_action, i_time_step]
            # update the "Kalman filter" activations
            # TODO TODO TODO 
            self.states.beh_activ_K[i_beh, i_time_step] = 0

        # get total activation for all behaviors of the other agent
        self.states.beh_activations[:, i_time_step] = \
            self.params.beta * self.states.beh_activ_G[:, i_time_step] \
            + self.states.beh_activ_K[:, i_time_step] 

        # get my estimated probabilities for the other agent's behavior
        self.states.beh_probs[:, i_time_step] = scipy.special.softmax(\
            self.params.Lambda * self.states.beh_activations[:, i_time_step])

        # loop through own action options and get momentary estimates
        # of the actions' values to me, as weighted average over the other 
        # agent's behaviors 
        for i_action, deltav in enumerate(self.params.deltavs):
            for i_beh in range(N_BEHAVIORS):
                self.states.mom_action_vals[i_action, i_time_step] += \
                    self.states.beh_probs[i_beh, i_time_step] \
                    * self.states.action_vals_given_behs[i_action, i_beh, i_time_step]

    def get_value_of_state_for_agent(self, own_state, oth_state):
        # TODO TODO TODO
        return 0

    def get_value_for_me(self, my_state, oth_state, i_action):            
        # cost for making a speed change, if any
        value = -self.params.k_a * self.params.deltavs[i_action] ** 2
        # add value of the state
        value += self.get_value_of_state_for_agent(my_state, oth_state)
        return value

    def get_value_for_other(self, oth_state, my_state, i_beh):
        # TODO TODO TODO add cost for the behavior itself
        value = 0
        # add value of the state
        value += self.get_value_of_state_for_agent(oth_state, my_state) # assuming symmetric cost function
        return value

    def get_predicted_own_state(self, i_action):
        # TODO TODO TODO
        return commotions.KinematicState()

    def get_predicted_other_state(self, i_beh):
        # TODO TODO TODO
        return commotions.KinematicState()


    def __init__(self, simulation, i_agent):
        # set initial state and call inherited init method
        initial_state = commotions.KinematicState(\
            pos = INITIAL_POSITIONS[i_agent,:], yaw_angle = None)
        super().__init__(AGENT_NAMES[i_agent], simulation, initial_state, \
            GOALS[i_agent,:], PLOT_COLORS[i_agent])
        # make and store a copy of the default parameters object
        self.params = copy.copy(DEFAULT_PARAMS)






# create the simulation and agents in it
scp_simulation = commotions.Simulation(START_TIME, END_TIME, TIME_STEP)
for i_agent in range(N_AGENTS):
    SCPAgent(scp_simulation, i_agent)

# run the simulation
scp_simulation.run()
scp_simulation.plot_trajectories()
plt.show()







        

