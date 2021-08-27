import warnings
import math
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import scipy.special
from scipy.stats import norm
import copy
from enum import Enum

import commotions
import sc_scenario_helper
from sc_scenario_helper import (CtrlType, AccessOrder, N_ACCESS_ORDERS, 
                                i_EGOFIRST, i_EGOSECOND,
                                SCAgentImage,
                                get_sc_agent_collision_margins, 
                                get_delay_discount)

#matplotlib.use('qt4agg')


class OptionalAssumption(Enum):
    oVA = 'oVA'
    oEA = 'oEA'
    oAN = 'oAN'
    oBEo = 'oBEo'
    oBEv = 'oBEv'
    oAI = 'oAI'

def get_assumptions_dict(default_value = False, **kwargs):
    """ Return a dictionary with all the members of OptionalAssumption as keys.
        The values in the dictionary are first all set to default_value, and
        then the items specified in keyword arguments are set accordingly, e.g., 
        
        get_assumptions_dict(default_value = False, oEA = True)
        
        returns a dictionary with all items set to False except the item with 
        key OptionalAssumption.oEA, which is set to True.
    """
    assumptions_dict = {}
    for assumption in OptionalAssumption:
        assumptions_dict[assumption] = default_value
    for kw in kwargs:
        assumptions_dict[OptionalAssumption(kw)] = kwargs[kw]
    return assumptions_dict

class DerivedAssumption(Enum):
    dBE = 'dBE'

N_AGENTS = 2 # this implementation supports only 2


BEHAVIORS = ('Const.', 'Pass 1st', 'Pass 2nd')
N_BEHAVIORS = len(BEHAVIORS)
i_CONSTANT = 0
i_PASS1ST = 1
i_PASS2ND = 2

# =============================================================================
# BEHAVIORS = ('const.', 'proc.', 'yield')
# N_BEHAVIORS = len(BEHAVIORS)
# i_CONSTANT = 0
# i_PROCEEDING = 1
# i_YIELDING = 2
# =============================================================================


# default parameters
DEFAULT_PARAMS = commotions.Parameters()
DEFAULT_PARAMS.T = 0.2 # action value accumulator / low-pass filter relaxation time (s)
DEFAULT_PARAMS.Tprime = DEFAULT_PARAMS.T  # behaviour value accumulator / low-pass filter relaxation time (s)
DEFAULT_PARAMS.beta_O = 1
DEFAULT_PARAMS.beta_V = 1
DEFAULT_PARAMS.T_O = 2 # "forgetting" time constant for behaviour observation (s)
DEFAULT_PARAMS.Lambda = 1
DEFAULT_PARAMS.sigma_O = .001 
DEFAULT_PARAMS.sigma_V = 0.1 # action value noise in evidence accumulation
DEFAULT_PARAMS.sigma_Vprime = DEFAULT_PARAMS.sigma_V # behaviour value noise in evidence accumulation
DEFAULT_PARAMS.DeltaV_th = 0.1 # action decision threshold when doing evidence accumulation
DEFAULT_PARAMS.DeltaT = 0.5 # action duration (s)
DEFAULT_PARAMS.T_P = DEFAULT_PARAMS.DeltaT # prediction time (s)
DEFAULT_PARAMS.T_delta = 30 # s; half-life of delay-discounted value
DEFAULT_PARAMS.V_ny = -15 * 0 # value function term for non-yielding 
DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.5, 0, 0.5, 1]) # available speed/acc change actions, magnitudes in m/s or m/s^2 dep on agent type
i_NO_ACTION = np.argwhere(DEFAULT_PARAMS.ctrl_deltas == 0)[0][0]
N_ACTIONS = len(DEFAULT_PARAMS.ctrl_deltas)
warnings.warn('N_ACTIONS set to no of actions in default params, so will not work if non-default params are set.')

# default gains for affordance-based value function
DEFAULT_PARAMS_K_VA = {}
DEFAULT_PARAMS_K_VA[CtrlType.SPEED] = commotions.Parameters()
FREE_SPEED_SPEED_CTRL = 1.5
FREE_SPEED_ACC_CTRL = 10
# set k_g and k_dv for normalised value rates across agent types 
# (see handwritten notes from 2021-01-16)
DEFAULT_PARAMS_K_VA[CtrlType.SPEED]._g = 2 / FREE_SPEED_SPEED_CTRL 
DEFAULT_PARAMS_K_VA[CtrlType.SPEED]._dv = 1 / FREE_SPEED_SPEED_CTRL ** 2
DEFAULT_PARAMS_K_VA[CtrlType.SPEED]._da = 0.5 # gives sensible-looking acceleration from standstill
DEFAULT_PARAMS_K_VA[CtrlType.ACCELERATION] = commotions.Parameters()
DEFAULT_PARAMS_K_VA[CtrlType.ACCELERATION]._g = 2 / FREE_SPEED_ACC_CTRL 
DEFAULT_PARAMS_K_VA[CtrlType.ACCELERATION]._dv = 1 / FREE_SPEED_ACC_CTRL ** 2
DEFAULT_PARAMS_K_VA[CtrlType.ACCELERATION]._da = 0.5 # gives sensible-looking acceleration from standstill

# default gains for original, non-affordance-based value function
DEFAULT_PARAMS_K_NVA = {}
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED] = commotions.Parameters()
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED]._g = 1 
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED]._dv = 0.3
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED]._c = 1   
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED]._e = 0.1   
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION] = commotions.Parameters()
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._g = 1 
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._dv = 0.05
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._da = 0.01
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._sc = 1    
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._sg = 0.25 
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._e = 0.1    
    
def get_default_params(oVA):
    params = copy.copy(DEFAULT_PARAMS)
    if oVA:
        params_k = copy.deepcopy(DEFAULT_PARAMS_K_VA)
    else:
        params_k = copy.deepcopy(DEFAULT_PARAMS_K_NVA)
    return (params, params_k)


SHARED_PARAMS = commotions.Parameters()
SHARED_PARAMS.d_C = 2 # collision distance

TTC_FOR_COLLISION = 0.1
MIN_BEH_PROB = 0.0 # behaviour probabilities below this value are considered zero

class States():
    pass

class SCAgent(commotions.AgentWithGoal):

    def prepare_for_simulation(self):
        # make sure this agent isn't used for a simulation with more than two
        # agents
        assert(len(self.simulation.agents) == 2)
        # store a reference to the other agent
        for agent in self.simulation.agents:
            if agent is not self:
                self.other_agent = agent
        # store an "image" of the other agent, with parameters assumed same as 
        # own parameters (but for the appropriate ctrl type)
        oth_params = copy.copy(self.params)
        oth_params.k = copy.copy(self.params.k_all[self.other_agent.ctrl_type])
        oth_v_free = sc_scenario_helper.get_agent_free_speed(oth_params.k)
        self.oth_image = SCAgentImage(ctrl_type = self.other_agent.ctrl_type,
                                      params = oth_params, v_free = oth_v_free)
        # allocate vectors for storing internal states
        self.n_actions = len(self.params.ctrl_deltas)
        n_time_steps = self.simulation.settings.n_time_steps
        self.states = States()
        # - states regarding my own actions
        self.states.mom_action_vals = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # Vtilde_A,a(t)
        self.states.est_action_vals = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # Vhat_A,a(t)
        self.states.est_action_surplus_vals = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # DeltaVhat_a(t)
        self.states.action_vals_given_behs_outcs = \
            math.nan * np.ones((self.n_actions, N_BEHAVIORS, n_time_steps, 
                                N_ACCESS_ORDERS)) # V_A[xtilde(t)|(a,b,Omega)]
        self.states.action_vals_given_behs = \
            math.nan * np.ones((self.n_actions, N_BEHAVIORS, 
                                n_time_steps)) # V_A[xtilde(t)|(a,b)]
        #self.states.action_probs = \
        #    math.nan * np.ones((self.n_actions, n_time_steps)) # P_a(t)
        # - states regarding the behavior of the other agent
        self.states.beh_activations_given_actions = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps)) # A_b|a(t)
        self.states.mom_beh_activ_V_given_actions = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps)) # Atilde_V,b|a(t)
        self.states.beh_activ_V_given_actions = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps)) # Ahat_V,b|a(t)
        self.states.beh_activ_O = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # Ahat_O,b(t)
        self.states.beh_probs_given_actions = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps)) # P_b|a(t)
        self.states.beh_vals_given_actions = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps)) # V_b|a(t)
        self.states.beh_vals_given_actions_outcs = \
            math.nan * np.ones((N_BEHAVIORS, self.n_actions, n_time_steps, N_ACCESS_ORDERS)) # V_b|a,Omega_B(t)
        self.states.sensory_probs_given_behs = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # P_{x_o|b}(t)
        self.states.beh_long_accs = \
            math.nan * np.ones((N_BEHAVIORS, n_time_steps)) # the acceleration that the other agent should be applying right now if doing behaviour b
        # - other states
        self.states.time_left_to_CA_entry = math.nan * np.ones(n_time_steps)
        self.states.time_left_to_CA_exit = math.nan * np.ones(n_time_steps)
        # set initial values for states that depend on the previous time step
        self.states.est_action_vals[:, -1] = 0
        self.states.beh_activ_V_given_actions[:, :, -1] = 0
        self.states.beh_activ_O[:, -1] = 0
        self.states.beh_activ_O[i_CONSTANT, -1] = 10
        warnings.warn('****** Setting initial value of i_CONSTANT behaviour activation to arbitrary high value.')

        # calculate where the two agents' paths intersect, if it has not already 
        # been done
        if not hasattr(self.simulation, 'conflict_point'):
            self.simulation.conflict_point = \
                commotions.get_intersection_of_lines(
                    self.get_current_kinematic_state().pos, self.goal, 
                    self.other_agent.get_current_kinematic_state().pos, 
                    self.other_agent.goal)


    def get_signed_dist_to_conflict_pt(self, state):
        """Get the signed distance from the specified agent state to the conflict
        point. Positive sign means that the agent has its front toward the conflict
        point. (Which will typically mean that it is on its way toward the 
        conflict point - however this function does not check for backward
        travelling.)
        """
        vect_to_conflict_point = \
            self.simulation.conflict_point - state.pos
        heading_vect = \
            np.array((math.cos(state.yaw_angle), math.sin(state.yaw_angle)))
        return np.dot(heading_vect, vect_to_conflict_point)

    
    def add_sc_state_info(self, state):
        state.signed_CP_dist = self.get_signed_dist_to_conflict_pt(state)
        state = sc_scenario_helper.add_entry_exit_times_to_state(
                state, SHARED_PARAMS.d_C)
        return state
    
    
    def plot_state_snapshot(self, state, plot_color, alpha=False, 
                            speed_label=False):
        if alpha:
            alpha_val = 0.5
        else:
            alpha_val = 1
        speed_vect = (state.long_speed * self.params.DeltaT
                      * np.array((math.cos(state.yaw_angle), 
                                  math.sin(state.yaw_angle))))
        vect_pos = state.pos + speed_vect
        self.snapshot_ax.plot((state.pos[0], vect_pos[0]),
                              (state.pos[1], vect_pos[1]), '-',
                              color=plot_color, alpha=alpha_val)
        self.snapshot_ax.plot(state.pos[0], state.pos[1], 'o', 
                              color=plot_color, alpha=alpha_val)
        if speed_label:
            text_pos = state.pos + 2 * speed_vect
            self.snapshot_ax.text(text_pos[0], text_pos[1], 
                                  '%.2f' % state.long_speed, 
                                  color=plot_color, alpha=alpha_val,
                                  ha='center', va='center', size=8)
    
    
    def noisy_lp_filter(self, T, sigma, prevXhat, currXtilde):
        noise = np.random.randn(len(prevXhat)) * sigma * math.sqrt(
            self.simulation.settings.time_step)
        f = self.simulation.settings.time_step / T
        currXhat = (1 - f) * prevXhat + f * currXtilde + noise
        return currXhat
    
    
    def prepare_for_action_update(self):
        """ Override the base class method with some per-timestep 
            precalculation - done here so that both agents can access these
            for both agents in do_action_update().
        """
        self.curr_state = self.get_current_kinematic_state()
        self.curr_state.long_acc = \
            self.trajectory.long_acc[self.simulation.state.i_time_step-1]
        self.curr_state = self.add_sc_state_info(self.curr_state)


    def do_action_update(self):
        """Do the action update for the agent. 
        """

        i_time_step = self.simulation.state.i_time_step
        time_step = self.simulation.settings.time_step
        time_stamp = self.simulation.time_stamps[i_time_step]
        
        # is this agent just supposed to keep constant speed?
        if self.const_acc != None:
            self.trajectory.long_acc[i_time_step] = self.const_acc
            return

        # if agent can't reverse and is now at zero speed, make sure that any
        # future negative accelerations from past actions are cleared
        if not self.can_reverse and self.curr_state.long_speed == 0:
            self.action_long_accs[i_time_step:] = \
                np.maximum(0, self.action_long_accs[i_time_step:] )

        # calculate my own current projected time until entering and
        # exiting the conflict area
        # TODO: These values aren't really used for anything now - so maybe 
        # remove these calculations?
        proj_long_speeds = self.curr_state.long_speed \
            + np.cumsum(self.action_long_accs[i_time_step:] * time_step)
        if not self.can_reverse:
            proj_long_speeds = np.maximum(0, proj_long_speeds)
        proj_signed_dist_to_CP = self.curr_state.signed_CP_dist \
            - np.cumsum(proj_long_speeds * time_step)
        # - entry
        i_time_steps_entered = \
            np.nonzero(proj_signed_dist_to_CP < SHARED_PARAMS.d_C)[0]
        if len(i_time_steps_entered) == 0:
            # not currently projected to enter he conflict area within the simulation duration
            self.states.time_left_to_CA_entry[i_time_step] = math.inf
        else:
            self.states.time_left_to_CA_entry[i_time_step] = \
                i_time_steps_entered[0] * time_step
        # - exit
        i_time_steps_exited = \
            np.nonzero(proj_signed_dist_to_CP < -SHARED_PARAMS.d_C)[0]
        if i_time_steps_exited.size == 0:
            # not currently projected to exit the conflict area within the simulation duration
            self.states.time_left_to_CA_exit[i_time_step] = math.inf
        else:
            self.states.time_left_to_CA_exit[i_time_step] = \
                i_time_steps_exited[0] * time_step

        # calculate the accelerations needed for the different behaviors of the 
        # other agent, as of the current time 
        # - constant behavior
        self.states.beh_long_accs[i_CONSTANT, i_time_step] = 0  
        # - use helper function to get other agent's expected accelerations to
        #   pass in front of or behind me, given my current position and speed
        (self.states.beh_long_accs[i_PASS1ST, i_time_step], 
         self.states.beh_long_accs[i_PASS2ND, i_time_step]) = \
             sc_scenario_helper.get_access_order_accs(
                     self.oth_image, self.other_agent.curr_state, 
                     self.curr_state, SHARED_PARAMS.d_C)
             
        # determine which behaviours are valid at this time step
        # - the helper function above returns nan if behaviour is invalid for 
        # - this time step
        beh_is_valid = np.invert(np.isnan(
                self.states.beh_long_accs[:, i_time_step]))
        # - is the constant behaviour valid for this time step?
        if (self.assumptions[DerivedAssumption.dBE] and
            (beh_is_valid[i_PASS1ST] or beh_is_valid[i_PASS2ND])):
            # no - we are estimating behaviours and at least one of the 
            # behaviours is valid
            beh_is_valid[i_CONSTANT] = False
            self.states.beh_long_accs[i_CONSTANT, i_time_step] = math.nan
         
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
            
        # are we doing a value function snapshot at this time step? 
        if self.doing_snapshots and (time_stamp in self.snapshot_times):
            self.do_snapshot_now = True
            # set up the figure
            fig_name = 'Snapshot for %s at t = %.2f s' % (self.name, time_stamp)
            fig = plt.figure(num=fig_name, figsize=(15, 10))
            fig.clf()
            axs = fig.subplots(nrows=N_BEHAVIORS, ncols=N_ACTIONS,
                                    sharex=True, sharey=True)
        else:
            self.do_snapshot_now = False
            
        # now loop over all combinations of own actions and other's behaviors, 
        # and get values from both agents' perspectives
        for i_action in range(self.n_actions):
            for i_beh in range(N_BEHAVIORS):
                # is this behaviour valid for this time step? 
                # (if not leave values as NaNs)
                if beh_is_valid[i_beh]:
                    # what type of value functions (affordance-based or not?)
                    if self.assumptions[OptionalAssumption.oVA]:
                        # affordance-based value functions
                        # doing snapshot?
                        if self.do_snapshot_now:
                            self.snapshot_ax = axs[i_beh, i_action]
                            self.plot_state_snapshot(self.curr_state, 
                                                     self.plot_color, 
                                                     alpha=True)
                            self.plot_state_snapshot(self.other_agent.curr_state, 
                                                     self.other_agent.plot_color, 
                                                     alpha=True)
                            self.plot_state_snapshot(pred_own_states[i_action],
                                                     self.plot_color, 
                                                     speed_label = True)
                            self.plot_state_snapshot(pred_oth_states[i_beh],
                                                     self.other_agent.plot_color, 
                                                     speed_label = True)
                        # get value for me of this action/behavior combination,
                        # storing both the per-access-order values and the max value
                        # of those                        
                        self.states.action_vals_given_behs_outcs[
                                i_action, i_beh, i_time_step, :] = (
                                self.get_access_order_values_for_me_v02(
                                        pred_own_states[i_action],
                                        pred_oth_states[i_beh]))
                        self.states.action_vals_given_behs[
                                i_action, i_beh, i_time_step] = np.nanmax(
                                self.states.action_vals_given_behs_outcs[
                                        i_action, i_beh, i_time_step, :])
                        # get value for for the other agent of this 
                        # action/behavior combination
                        # - first across both possible outcomes
                        if self.assumptions[OptionalAssumption.oAI]:
                            # considering impact of own actions on value for other agent
                            self.states.beh_vals_given_actions_outcs[
                                    i_beh, i_action, i_time_step, :] = (
                                    self.get_access_order_values_for_other_v02(
                                            pred_oth_states[i_beh],
                                            pred_own_states[i_action]) )
                        else:
                            # not considering impact of own actions on value for other agent
                            if i_action == 0:
                                self.states.beh_vals_given_actions_outcs[
                                    i_beh, i_action, i_time_step, :] = (
                                    self.get_access_order_values_for_other_v02(
                                            pred_oth_states[i_beh],
                                            pred_own_states[i_NO_ACTION]) )
                            else:
                                # no need to recalculate the same value as above
                                self.states.beh_vals_given_actions_outcs[
                                    i_beh, i_action, i_time_step, :] = (
                                        self.states.beh_vals_given_actions_outcs[
                                            i_beh, 0, i_time_step, :])
                        # - then use the value for the outcome corresponding
                        #   to the behaviour in question
                        if i_beh == i_PASS1ST:
                            self.states.beh_vals_given_actions[
                                    i_beh, i_action, i_time_step] = (
                                    self.states.beh_vals_given_actions_outcs[
                                            i_beh, i_action, i_time_step, 
                                            i_EGOFIRST])
                        elif i_beh == i_PASS2ND:
                            self.states.beh_vals_given_actions[
                                    i_beh, i_action, i_time_step] = (
                                    self.states.beh_vals_given_actions_outcs[
                                            i_beh, i_action, i_time_step, 
                                            i_EGOSECOND])
                        elif i_beh == i_CONSTANT:
                            # not quite sure what is appropriate to do here -
                            # constant behaviour being considered valid means
                            # that we are either not estimating behaviours (in
                            # which case these behaviour values shouldn't
                            # be used anywhere), or no other behaviour is 
                            # considered valid, which means that the
                            # interaction is over anyway (I am pretty sure of
                            # that at least?) - so the value set here may
                            # not matter much anyway
                            self.states.beh_vals_given_actions[
                                    i_beh, i_action, i_time_step] = np.nanmax(
                                    self.states.beh_vals_given_actions_outcs[
                                            i_beh, i_action, i_time_step, :])
                        else:
                            raise Exception('Unexpected behaviour.')
                    else:
                        # original, non-affordance-based value functions
                        # get value for me of this action/behavior combination
                        self.states.action_vals_given_behs[
                            i_action, i_beh, i_time_step] = \
                            self.get_value_for_me(pred_own_states[i_action], 
                                                  pred_oth_states[i_beh], 
                                                  i_action)
                        # get value for the other agent of this action/behavior 
                        # combination
                        if self.assumptions[OptionalAssumption.oAI]:
                            # considering impact of own actions on value for other agent
                            self.states.beh_vals_given_actions[
                                i_beh, i_action, i_time_step] = \
                                self.get_value_for_other(pred_oth_states[i_beh], 
                                                         pred_own_states[i_action], 
                                                         i_beh)
                        else:
                            # not considering impact of own actions on value for other agent
                            if i_action == 0:
                                # if this is the first action being considered,
                                # calculate value, assuming no ego action
                                self.states.beh_vals_given_actions[
                                    i_beh, i_action, i_time_step] = \
                                    self.get_value_for_other(pred_oth_states[i_beh], 
                                                             pred_own_states[i_NO_ACTION], 
                                                             i_beh)
                            else:
                                # no need to recalculate the same value as above
                                self.states.beh_vals_given_actions[
                                    i_beh, i_action, i_time_step] = \
                                    self.states.beh_vals_given_actions[
                                        i_beh, 0, i_time_step]
                                
                            

        ## get my estimated probabilities for my own actions - based on value 
        ## estimates from the previous time step
        #self.states.action_probs[:, i_time_step] = scipy.special.softmax(\
        #    self.params.Lambda * self.states.est_action_vals[:, i_time_step-1])

        # now loop over the other agent's behaviors, to update the corresponding
        # activations (my "belief" in these behaviors)
        for i_beh in range(N_BEHAVIORS):
            if not beh_is_valid[i_beh]:
                self.states.mom_beh_activ_V_given_actions[
                    i_beh, :, i_time_step] = 0
                self.states.beh_activ_V_given_actions[
                    i_beh, :, i_time_step] = 0
                self.states.beh_activ_O[i_beh, i_time_step] = 0
                self.states.beh_activations_given_actions[
                    i_beh, :, i_time_step] = 0
            else:
                # update value-based activations
                if self.assumptions[OptionalAssumption.oBEv]:
                    # get momentary estimates of the value-based activations
                    # (currently simply equal to the predicted value of
                    # the behaviour given the actions - but in more advanced 
                    # formulations could also take into account probabilities 
                    # of ego agent behaviours, etc)
                    self.states.mom_beh_activ_V_given_actions[
                        i_beh, :, i_time_step] = \
                        self.states.beh_vals_given_actions[i_beh, :, i_time_step] 
                    # update accumulative estimates of the value-based activations
                    self.states.beh_activ_V_given_actions[
                        i_beh, :, i_time_step] = self.noisy_lp_filter(
                            self.params.Tprime, self.params.sigma_Vprime,
                            self.states.beh_activ_V_given_actions[
                                i_beh, :, i_time_step-1],
                            self.states.mom_beh_activ_V_given_actions[
                                i_beh, :, i_time_step])
                else:
                    self.states.mom_beh_activ_V_given_actions[
                        i_beh, :, i_time_step] = 0
                    self.states.beh_activ_V_given_actions[
                        i_beh, :, i_time_step] = 0
                # update the "Kalman filter" activations
                if self.assumptions[OptionalAssumption.oBEo]:
                    self.states.beh_activ_O[i_beh, i_time_step] = (
                        (1 - self.simulation.settings.time_step / self.params.T_O) 
                        * self.states.beh_activ_O[i_beh, i_time_step-1])
                    if i_time_step > 0:
                        self.states.sensory_probs_given_behs[
                            i_beh, i_time_step] = \
                            self.get_prob_of_current_state_given_beh(i_beh)
                        self.states.beh_activ_O[i_beh, i_time_step] += \
                            (1/self.params.Lambda) \
                            * math.log(self.states.sensory_probs_given_behs[
                                i_beh, i_time_step])
                else:
                    self.states.beh_activ_O[i_beh, i_time_step] = 0
                # get total activation for this behaviour
                self.states.beh_activations_given_actions[
                    i_beh, :, i_time_step] = \
                    self.params.beta_V * \
                        self.states.beh_activ_V_given_actions[
                            i_beh, :, i_time_step] \
                    + self.params.beta_O * \
                        self.states.beh_activ_O[i_beh, i_time_step] 

        # get my estimated probabilities for the other agent's behavior
        if self.assumptions[DerivedAssumption.dBE]:
            for i_action in range(self.n_actions):
                # get probabilities as softmax over activations for valid
                # behaviors
                self.states.beh_probs_given_actions[
                        beh_is_valid, i_action, i_time_step] = \
                    scipy.special.softmax(
                            self.params.Lambda 
                            * self.states.beh_activations_given_actions[
                                    beh_is_valid, i_action, i_time_step])
                # set probabilities of invalid behaviors to zero
                self.states.beh_probs_given_actions[
                        np.invert(beh_is_valid), i_action, i_time_step] = 0
        else:
            # not estimating behaviors, so set probability for constant 
            # acceleration behavior to one
            self.states.beh_probs_given_actions[:, :, i_time_step] = 0
            self.states.beh_probs_given_actions[i_CONSTANT, :, i_time_step] = 1

        # loop through own action options and get momentary estimates
        # of the actions' values to me, as weighted average over the other 
        # agent's behaviors 
        self.states.mom_action_vals[:, i_time_step] = 0
        for i_action in range(self.n_actions):
            for i_beh in range(N_BEHAVIORS):
                if beh_is_valid[i_beh] and self.states.beh_probs_given_actions[
                        i_beh, i_action, i_time_step] > MIN_BEH_PROB: 
                    self.states.mom_action_vals[i_action, i_time_step] += \
                        self.states.beh_probs_given_actions[
                                i_beh, i_action, i_time_step] \
                        * self.states.action_vals_given_behs[
                                i_action, i_beh, i_time_step]

        # update the accumulative estimates of action value
        self.states.est_action_vals[:, i_time_step] = self.noisy_lp_filter(
            self.params.T, self.params.sigma_V,
            self.states.est_action_vals[:, i_time_step-1],
            self.states.mom_action_vals[:, i_time_step])

        # any action over threshold?
        self.states.est_action_surplus_vals[:, i_time_step] = \
            self.states.est_action_vals[:, i_time_step] \
            - self.states.est_action_vals[i_NO_ACTION, i_time_step]
        i_best_action = np.argmax(self.states.est_action_surplus_vals[:, i_time_step])
        if self.states.est_action_surplus_vals[i_best_action, i_time_step] \
            > self.params.DeltaV_th:
            # add action to the array of future acceleration values
            self.add_action_to_acc_array(self.action_long_accs, i_best_action, \
                self.simulation.state.i_time_step)
            if self.assumptions[OptionalAssumption.oEA]:
                # reset the value accumulators
                self.states.est_action_vals[:, i_time_step] = 0
                self.states.beh_activ_V_given_actions[:, :, i_time_step] = 0

        # set long acc in actual trajectory
        self.trajectory.long_acc[i_time_step] = self.action_long_accs[i_time_step]
        

    def get_access_order_values_for_agent_v02(self, ego_image, ego_curr_state,
                                         ego_pred_state, oth_pred_state,
                                         snapshot_color, snapshot_loc):
        
        # skipping goal stopping for now - I think the sensible way of adding
        # it back in later is to include an independent term for it, valuating
        # the needed manoeuvre for stopping just like this function valuates
        # the needed manoevure for achieving each access order
       
        # get the effective average acceleration or jerk (depending on agent 
        # type) in the action/prediction interval 
        if ego_image.ctrl_type is CtrlType.SPEED:
            action_acc0 = (ego_pred_state.long_speed 
                          - ego_curr_state.long_speed) / ego_image.params.DeltaT
            action_jerk = 0
        else:
            action_acc0 = ego_curr_state.long_acc
            action_jerk = (ego_pred_state.long_acc
                           - ego_curr_state.long_acc) / ego_image.params.DeltaT
        action_value = sc_scenario_helper.get_value_of_const_jerk_interval(
                v0 = ego_curr_state.long_speed, a0 = action_acc0, j = action_jerk, 
                T = ego_image.params.DeltaT, k = ego_image.params.k)
        
        # get constant value for remainder of trip after action and possible
        # interaction
        post_value = ( (ego_image.params.T_delta / math.log(2)) * 
                      sc_scenario_helper.get_const_value_rate(
                              v=ego_image.v_free, a=0, k=ego_image.params.k) ) 
        
        # call helper function to get needed manoeuvring and delay times for
        # each access order, starting from this state
        implications = sc_scenario_helper.get_access_order_implications(
                ego_image, ego_pred_state, oth_pred_state, SHARED_PARAMS.d_C)
        
        # get the estimated time needed for the agent to regain free speed, 
        # if not already at it
        if ego_image.ctrl_type is CtrlType.SPEED:
            agent_time_to_v_free = ego_image.params.DeltaT
        else:
            agent_time_to_v_free = sc_scenario_helper.ACC_CTRL_REGAIN_SPD_TIME
        
        # loop over the access orders and get value for each
        if self.do_snapshot_now:
            snapshot_str = ''
        access_order_values = np.full(N_ACCESS_ORDERS, math.nan)
        for access_order in AccessOrder:
            if np.isnan(implications[access_order].acc):
                # access order not valid from this state
                value = -math.inf
            else:
                # valuation of the action/prediction time interval
                value = action_value
                # get the duration of the interval where the access order is 
                # achieved (the get_value_... helper function can't handle 
                # infinite times so cap at an hour...)
                T_omega = min(3600, implications[access_order].T_acc)
                # valuation of the manoeuvre needed to achieve the access order
                ach_access_value = (
                        get_delay_discount(ego_image.params.DeltaT, 
                                           ego_image.params.T_delta) 
                        * sc_scenario_helper.get_value_of_const_jerk_interval(
                        ego_pred_state.long_speed, implications[access_order].acc, 
                        j = 0, T = T_omega, 
                        k = ego_image.params.k) )
                value += ach_access_value
                # inherent value of this access order
                if (ego_image.ctrl_type is CtrlType.ACCELERATION 
                    and access_order is AccessOrder.EGOFIRST):
                    inh_access_value = ego_image.params.V_ny
                else:
                    inh_access_value = 0
                value += inh_access_value
                # valuation of the step of regaining the free speed after the
                # access order has been achieved (may be a zero duration
                # step if the access order was achieved by an acceleration to
                # the free speed)
                # - get the delay before the regaining can happen, including
                #   any waiting time
                delay_bef_reg_free = (ego_image.params.DeltaT
                                    + T_omega 
                                    + implications[access_order].T_dw)
                # - get the speed after having achieved the access order
                vprime = (ego_pred_state.long_speed 
                          + T_omega * implications[access_order].acc)
                # - get the acceleration that will be applied to regain free speed
                accprime = -(vprime - ego_image.v_free) / agent_time_to_v_free
                # - calculate the value contribution
                regain_value = (
                    get_delay_discount(delay_bef_reg_free,
                                       ego_image.params.T_delta) 
                    * sc_scenario_helper.get_value_of_const_jerk_interval(
                        vprime, accprime, j = 0, T = agent_time_to_v_free, 
                        k = ego_image.params.k) )
                value += regain_value
                # valuation of the remainder of the journey, factoring in the
                # delays associated with this access order
                total_delay = delay_bef_reg_free + agent_time_to_v_free
                final_value = get_delay_discount(
                        total_delay, ego_image.params.T_delta) * post_value
                value += final_value
                # snapshot info
                if self.do_snapshot_now:
                    snapshot_str += ('(%.1f m/s^2, %.1f s, %.1f s)' %
                                     (implications[access_order].acc,
                                      implications[access_order].T_acc,
                                      implications[access_order].T_dw) + '\n'
                                     + 'V = %.2f' % value + '\n'
                                     +'(%.1f, %.1f, %.1f, %.1f, %.1f)' % 
                                     (action_value, ach_access_value,
                                      inh_access_value, regain_value,
                                      final_value) + '\n')
            # store the value of this access order in the output numpy array
            value = max(-100, min(100, value)) # awaiting some proper value fcn squashing
            access_order_values[access_order.value] = value
            
        # snapshot info
        if self.do_snapshot_now:
            if snapshot_loc == 'topleft':
                text_x = 0.05
                text_y = 0.95
                halign = 'left'
                valign = 'top'
            elif snapshot_loc == 'bottomright':
                text_x = 0.95
                text_y = 0.05
                halign = 'right'
                valign = 'bottom'
            else:
                raise Exception('Unexpected location for snapshot info.')
            self.snapshot_ax.text(text_x, text_y, snapshot_str, 
                                  transform=self.snapshot_ax.transAxes,
                                  ha=halign, va=valign, fontsize=7,
                                  color=snapshot_color)
            
        return access_order_values
        
    
    def get_access_order_values_for_me_v02(self, my_pred_state, oth_pred_state):
        access_order_values = self.get_access_order_values_for_agent_v02(
                ego_image = self.self_image, 
                ego_curr_state = self.curr_state, 
                ego_pred_state = my_pred_state, 
                oth_pred_state = oth_pred_state,
                snapshot_color = self.plot_color,
                snapshot_loc = 'topleft')
        return access_order_values
        
    
    def get_access_order_values_for_other_v02(self, oth_pred_state, my_pred_state):
        access_order_values = self.get_access_order_values_for_agent_v02(
                ego_image = self.oth_image, 
                ego_curr_state = self.other_agent.curr_state, 
                ego_pred_state = oth_pred_state,
                oth_pred_state = my_pred_state,
                snapshot_color = self.other_agent.plot_color,
                snapshot_loc = 'bottomright')
        return access_order_values


    def get_value_of_state_for_agent(self, own_state, own_goal, oth_state, \
        ctrl_type, k):

        heading_vector = np.array([math.cos(own_state.yaw_angle), \
            math.sin(own_state.yaw_angle)])
        
        # reward for progress toward goal and speed discomfort cost
        vector_to_goal = own_goal - own_state.pos
        heading_to_goal_vector = vector_to_goal / np.linalg.norm(vector_to_goal)
        heading_toward_goal_component = \
            np.dot(heading_vector, heading_to_goal_vector)
        goal_distance_change_rate = \
            -heading_toward_goal_component * own_state.long_speed
        value = -k._g * goal_distance_change_rate \
            - k._dv * own_state.long_speed ** 2 
        
        if ctrl_type is CtrlType.ACCELERATION:
            # acceleration discomfort cost
            value += -k._da * own_state.long_acc ** 2
            # cost for acceleration required to stop at goal
            goal_distance = np.linalg.norm(vector_to_goal)
            req_acc_to_goal = -(own_state.long_speed ** 2 / (2 * goal_distance))
            value += -k._sg * req_acc_to_goal ** 2


        # cost for being on collision course with the other agent
        time_to_agent_collision = \
            sc_scenario_helper.get_time_to_sc_agent_collision(own_state, 
                                                              oth_state)
        
        if time_to_agent_collision == 0:
            time_to_agent_collision = TTC_FOR_COLLISION

        if time_to_agent_collision < math.inf:
            if ctrl_type is CtrlType.SPEED:
                value += -k._c / time_to_agent_collision  
            elif self.ctrl_type is CtrlType.ACCELERATION:
                value += -k._sc * (own_state.long_speed \
                    / (2 * time_to_agent_collision)) ** 2  
        
        return value


    def get_value_for_me(self, my_state, oth_state, i_action):            
        # cost for making a speed change, if any
        value = -self.params.k._e * self.params.ctrl_deltas[i_action] ** 2
        # add value of the state
        value += self.get_value_of_state_for_agent(\
            my_state, self.goal, oth_state, self.ctrl_type, self.params.k)
        return value


    def get_value_for_other(self, oth_state, my_state, i_beh):
        # POSSIBLE TODO - add cost for the behavior itself
        # - disregarding acceleration discomfort for the other agent, for now at least
        #   (needs a bit of code restructuring, or storing the cost somewhere)
        value = 0
        # add value of the state
        value += self.get_value_of_state_for_agent(
                oth_state, self.other_agent.goal, my_state, 
                self.other_agent.ctrl_type, self.oth_image.params.k) 
        return value


    def add_action_to_acc_array(self, acc_array, i_action, i_time_step):
        if self.ctrl_type is CtrlType.SPEED:
            acc_value = self.params.ctrl_deltas[i_action] / self.params.DeltaT
            commotions.add_uniform_action_to_array(acc_array, acc_value, \
                self.n_action_time_steps, i_time_step)
        elif self.ctrl_type is CtrlType.ACCELERATION:
            acc_delta = self.params.ctrl_deltas[i_action]
            commotions.add_linear_ramp_action_to_array(acc_array, acc_delta, \
                self.n_action_time_steps, i_time_step)
        else:
            raise RuntimeError('Unexpected control type %s.' % self.ctrl_type)

    def get_predicted_own_state(self, i_action):
        local_long_accs = np.copy(self.action_long_accs)
        self.add_action_to_acc_array(
                local_long_accs, i_action, self.simulation.state.i_time_step)
        predicted_state = self.get_future_kinematic_state(
                local_long_accs, yaw_rate = 0, 
                n_time_steps_to_advance = self.n_prediction_time_steps)
        predicted_state.long_acc = local_long_accs[
                self.simulation.state.i_time_step + self.n_prediction_time_steps]
        # add SC scenario specific state info
        predicted_state = self.add_sc_state_info(predicted_state)
        return predicted_state


    def get_predicted_other_state(self, i_beh):
        # get the longitudinal acceleration for this behavior, if implemented at
        # the current time step
        long_acc_for_this_beh = \
            self.states.beh_long_accs[i_beh, self.simulation.state.i_time_step]
        if math.isnan(long_acc_for_this_beh):
            predicted_state = None
        else:
            # let the other agent object calculate what its predicted state would
            # be with this acceleration 
            predicted_state = self.other_agent.get_future_kinematic_state(
                    long_acc_for_this_beh, yaw_rate = 0, 
                    n_time_steps_to_advance = self.n_prediction_time_steps)
            predicted_state.long_acc = long_acc_for_this_beh
            # add SC scenario specific state info
            predicted_state = self.add_sc_state_info(predicted_state)
        return predicted_state
        

    def get_prob_of_current_state_given_beh(self, i_beh):
        i_prev_time_step = self.simulation.state.i_time_step-1
        if math.isnan(self.states.beh_long_accs[i_beh, i_prev_time_step]):
            prob_density = 0
        else:
            # retrieve the longitudinal acceleration for this behavior, as estimated on
            # the previous time step
            prev_long_acc_for_this_beh = \
                self.states.beh_long_accs[i_beh, i_prev_time_step]
            # let the other agent object calculate what its predicted state at the 
            # current time step would be with this acceleration     
            expected_curr_state = self.other_agent.get_future_kinematic_state(
                    prev_long_acc_for_this_beh, yaw_rate = 0, 
                    n_time_steps_to_advance = 1, 
                    i_start_time_step = i_prev_time_step)
            # get the distance between expected and observed position
            pos_diff = np.linalg.norm(
                    expected_curr_state.pos 
                    - self.other_agent.get_current_kinematic_state().pos)
            # return the probability density for this observed difference
            prob_density = norm.pdf(pos_diff, scale = self.params.sigma_O)
        return max(prob_density, np.finfo(float).eps) # don't return zero probability
        


    def __init__(self, name, ctrl_type, simulation, goal_pos, initial_state, 
                 optional_assumptions = get_assumptions_dict(False), 
                 params = None, params_k = None, const_acc = None, 
                 plot_color = 'k', snapshot_times = None):

        # set control type
        self.ctrl_type = ctrl_type
        
        # set initial state and call inherited init method
        can_reverse = (self.ctrl_type is CtrlType.SPEED) # no reversing for acceleration-controlling agents
        super().__init__(name, simulation, goal_pos, \
            initial_state, can_reverse = can_reverse, plot_color = plot_color)
            
        # is this agent to just keep a constant acceleration?
        self.const_acc = const_acc
        
        # doing any value function snapshots?
        self.snapshot_times = snapshot_times
        self.doing_snapshots = snapshot_times != None
        
        # store the optional assumptions
        self.assumptions = optional_assumptions

        # get default parameters or use user-provided parameters
        # - non-value function parameters
        if params is None:
            self.params = copy.copy(DEFAULT_PARAMS)
        else:
            self.params = copy.copy(params)
        # - value function gains
        if params_k is None:
            if self.assumptions[OptionalAssumption.oVA]:
                self.params.k_all = copy.deepcopy(DEFAULT_PARAMS_K_VA)
            else:
                self.params.k_all = copy.deepcopy(DEFAULT_PARAMS_K_NVA)
        else:
            self.params.k_all = copy.deepcopy(params_k)
        # get value function gains for own ctrl type, for quick access
        self.params.k = copy.copy(self.params.k_all[self.ctrl_type])

        # parse the optional assumptions
        if not self.assumptions[OptionalAssumption.oEA]:
            # no evidence accumulation, implemented by value accumulation 
            # reaching input value in one time step...
            self.params.T = self.simulation.settings.time_step 
            self.params.Tprime = self.simulation.settings.time_step 
            # ... and decision threshold at zero
            self.params.DeltaV_th = 0
        if not self.assumptions[OptionalAssumption.oAN]:
            self.params.sigma_V = 0
            self.params.sigma_Vprime = 0
        if not self.assumptions[OptionalAssumption.oBEo]:
            self.params.beta_O = 0
        if not self.assumptions[OptionalAssumption.oBEv]:
            self.params.beta_V = 0
        self.assumptions[DerivedAssumption.dBE] = \
            self.assumptions[OptionalAssumption.oBEo] \
            or self.assumptions[OptionalAssumption.oBEv]

        # get and store own free speed
        self.v_free = sc_scenario_helper.get_agent_free_speed(self.params.k)
        
        # store a (correct) representation of oneself
        self.self_image = SCAgentImage(ctrl_type = self.ctrl_type, 
                                       params = self.params, 
                                       v_free = self.v_free)

        # POSSIBLE TODO: absorb this into a new class 
        #                commotions.AgentWithIntermittentActions or similar
        # store derived constants relating to the actions
        self.n_action_time_steps = math.ceil(
            self.params.DeltaT / self.simulation.settings.time_step)
        self.n_prediction_time_steps = math.ceil(self.params.T_P \
            / self.simulation.settings.time_step)
        self.n_actions_vector_length = \
            self.simulation.settings.n_time_steps + self.n_prediction_time_steps
        # prepare vectors for storing long acc and yaw rates, incl lookahead
        # with added actions
        self.action_long_accs = np.zeros(self.n_actions_vector_length)
        self.action_yaw_rates = np.zeros(self.n_actions_vector_length) # yaw rate should really always remain zero in this class - but leaving this vector here anyway


class SCSimulation(commotions.Simulation):

    def __init__(self, ctrl_types, goal_positions, initial_positions, 
                 initial_speeds = np.array((0, 0)), 
                 start_time = 0, end_time = 10, time_step = 0.1, 
                 optional_assumptions = get_assumptions_dict(False), 
                 params = None, params_k = None, const_accs = (None, None),
                 agent_names = ('A', 'B'), plot_colors = ('c', 'm'), 
                 snapshot_times = (None, None)):

        super().__init__(start_time, end_time, time_step)
       
        for i_agent in range(N_AGENTS):
            initial_state = commotions.KinematicState(
                    pos = initial_positions[i_agent,:], 
                    long_speed = initial_speeds[i_agent], yaw_angle = None)
            SCAgent(name = agent_names[i_agent], 
                    ctrl_type = ctrl_types[i_agent], 
                    simulation = self, 
                    goal_pos = goal_positions[i_agent, :], 
                    initial_state = initial_state, 
                    optional_assumptions = optional_assumptions, 
                    params = params, params_k = params_k, 
                    const_acc = const_accs[i_agent],
                    plot_color = plot_colors[i_agent],
                    snapshot_times = snapshot_times[i_agent])

    def do_plots(self, trajs = False, action_vals = False, action_probs = False, 
                 action_val_ests = False, surplus_action_vals = False, 
                 beh_activs = False, beh_accs = False, beh_probs = False, 
                 sensory_prob_dens = False, kinem_states = False, 
                 times_to_ca = False):


        if self.agents[0].assumptions[DerivedAssumption.dBE]:
            plot_behaviors = BEHAVIORS
        else:
            plot_behaviors = (BEHAVIORS[i_CONSTANT],)
        n_plot_behaviors = len(plot_behaviors)

        if trajs:
            # plot trajectories
            plt.figure('Trajectories')
            plt.clf()
            self.plot_trajectories()
            plt.legend()

        # agent state plots

        if action_vals:
            # - action values given behaviors
            plt.figure('Action values given behaviours', figsize = (10.0, 10.0))
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(DEFAULT_PARAMS.ctrl_deltas):
                    plt.subplot(N_ACTIONS, N_AGENTS, \
                                i_action * N_AGENTS +  i_agent + 1)
                    plt.ylim(-0, 50)
                    for i_beh in range(n_plot_behaviors):
                        plt.plot(self.time_stamps, \
                                 agent.states.action_vals_given_behs[i_action, 
                                                                     i_beh, :])
                    if i_action == 0:
                        plt.title('Agent %s' % agent.name)
                        if i_agent == 1:
                            plt.legend(plot_behaviors)
                    if i_agent == 0:
                        plt.ylabel('$V(%.1f | b)$' % deltav)

        if action_probs:
            # - action probabilities
            plt.figure('Action probabilities', figsize = (10.0, 10.0))
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(DEFAULT_PARAMS.ctrl_deltas):
                    plt.subplot(N_ACTIONS, N_AGENTS, \
                                i_action * N_AGENTS + i_agent + 1)
                    plt.plot(self.time_stamps, 
                             agent.states.action_probs[i_action, :])
                    plt.ylim(-.1, 1.1)
                    if i_action == 0:
                        plt.title('Agent %s' % agent.name)
                    if i_agent == 0:
                        plt.ylabel('$P(\\Delta v=%.1f)$' % deltav)

        if action_val_ests:
            # - momentary and accumulative estimates of action values
            plt.figure('Action value estimates', figsize = (10.0, 10.0))
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(DEFAULT_PARAMS.ctrl_deltas):
                    plt.subplot(N_ACTIONS, N_AGENTS, 
                                i_action * N_AGENTS + i_agent + 1)
                    plt.plot(self.time_stamps, 
                             agent.states.mom_action_vals[i_action, :])
                    plt.plot(self.time_stamps, 
                             agent.states.est_action_vals[i_action, :])
                    #plt.ylim(-2, 2)
                    if i_action == 0:
                        plt.title('Agent %s' % agent.name)
                        if i_agent == 1:
                            plt.legend(('$\\tilde{V}_a$', '$\\hat{V}_a$'))
                    if i_agent == 0:
                        plt.ylabel('$V(\\Delta v=%.1f)$' % deltav)

        if surplus_action_vals:
            # - surplus action values
            plt.figure('Surplus action value estimates', figsize = (6, 5))
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(DEFAULT_PARAMS.ctrl_deltas):
                    plt.subplot(N_ACTIONS, N_AGENTS, 
                                i_action * N_AGENTS + i_agent + 1)
                    plt.plot(self.time_stamps, 
                             agent.states.est_action_surplus_vals[i_action, :])
                    plt.plot([self.time_stamps[0], self.time_stamps[-1]], 
                        [agent.params.DeltaV_th, agent.params.DeltaV_th], 
                        '--', color = 'gray')
                    #plt.ylim(-.5, .3)
                    if i_action == 0:
                        plt.title('Agent %s' % agent.name)
                    if i_agent == 0:
                        plt.ylabel('$\\Delta V(%.1f)$' % deltav)

        if beh_activs:
            # - behavior activations
            for i_agent, agent in enumerate(self.agents):
                plt.figure('Behaviour activations - Agent %s (observing %s)' %
                           (agent.name, agent.other_agent.name), 
                           figsize = [7, 7])
                plt.clf()
                for i_beh in range(n_plot_behaviors):
                    # action observation contribution
                    plt.subplot(N_ACTIONS+1, n_plot_behaviors, i_beh + 1)
                    plt.plot(self.time_stamps, agent.states.beh_activ_O[i_beh, :])
                    plt.title(BEHAVIORS[i_beh])
                    if i_beh == n_plot_behaviors-1:
                        plt.legend(('$A_{O,b}$',))
                    # value contribution and total activation - both per action
                    for i_action, deltav in enumerate(DEFAULT_PARAMS.ctrl_deltas):
                        plt.subplot(N_ACTIONS+1, n_plot_behaviors, 
                                    (i_action+1) * n_plot_behaviors + i_beh + 1)
                        plt.plot(self.time_stamps, 
                                 agent.states.beh_activ_V_given_actions[
                                         i_beh, i_action,  :])
                        plt.plot(self.time_stamps, 
                                 agent.states.beh_activations_given_actions[
                                         i_beh, i_action, :])
                        #plt.ylim(-2, 5)
                        if i_beh == n_plot_behaviors-1 and i_action == 0:
                            plt.legend(('$A_{V,b|a}$', '$A_{b|a}$'))
                        if i_beh == 0:
                            plt.ylabel('$\\Delta v=%.1f$' % deltav)

        if beh_accs:
            # - expected vs observed accelerations for behaviors
            plt.figure('Expected vs observed accelerations for behaviors', 
                       figsize = (10.0, 10.0))
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                for i_beh in range(n_plot_behaviors):
                    plt.subplot(n_plot_behaviors, N_AGENTS, 
                                i_beh * N_AGENTS + i_agent + 1)
                    plt.plot(self.time_stamps, agent.states.beh_long_accs[i_beh, :], 
                        '--', color = 'gray', linewidth = 2)
                    plt.plot(self.time_stamps, agent.other_agent.trajectory.long_acc)
                    plt.ylim(-4, 4)
                    if i_beh == 0:
                        plt.title('Agent %s (observing %s)' % 
                                  (agent.name, agent.other_agent.name))
                        if i_agent == 1:
                            plt.legend(('expected', 'observed'))
                    if i_agent == 0:
                        plt.ylabel('%s a (m/s^2)' % BEHAVIORS[i_beh])

        if beh_probs:
            # - behavior probabilities
            plt.figure('Behaviour probabilities', figsize = [8, 7])
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(DEFAULT_PARAMS.ctrl_deltas):
                    plt.subplot(N_ACTIONS, N_AGENTS, i_action * N_AGENTS + i_agent + 1)
                    for i_beh in range(n_plot_behaviors):
                        # plt.subplot(N_BEHAVIORS, N_AGENTS, i_beh * N_AGENTS +  i_agent + 1)
                        plt.plot(self.time_stamps, 
                                 agent.states.beh_probs_given_actions[
                                         i_beh, i_action, :])
                        plt.ylim(-.1, 1.1)
                        if i_action == 0:
                            plt.title('Agent %s (observing %s)' % 
                                      (agent.name, agent.other_agent.name))
                    if i_agent == 0:
                        plt.ylabel('$P_{b|\\Delta v=%.1f}$ (-)' % deltav)
                    elif i_action == 0:
                        plt.legend(plot_behaviors)
                    
        if sensory_prob_dens:
            # - sensory probability densities
            plt.figure('Sensory probability densities')
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                for i_beh in range(n_plot_behaviors):
                    plt.subplot(n_plot_behaviors, N_AGENTS, 
                                i_beh * N_AGENTS + i_agent + 1)
                    plt.plot(self.time_stamps, \
                        np.log(agent.states.sensory_probs_given_behs[i_beh, :]))
                    if i_beh == 0:
                        plt.title('Agent %s (observing %s)' %
                                  (agent.name, agent.other_agent.name))
                    if i_agent == 0:
                        plt.ylabel('$log p(O|%s)$' % BEHAVIORS[i_beh])    

        if kinem_states:
            # - kinematic/action states
            fig = plt.figure('Kinematic and action states')
            plt.clf()
            """             N_PLOTROWS = 3
            for i_agent, agent in enumerate(self.agents):
                # position
                plt.subplot(N_PLOTROWS, N_AGENTS, 0 * N_AGENTS +  i_agent + 1)
                plt.plot(self.time_stamps, agent.trajectory.pos.T)
                plt.title('Agent %s' % agent.name)
                if i_agent == 0:
                    plt.ylabel('pos (m)')
                else:
                    plt.legend(('x', 'y'))
                # speed
                plt.subplot(N_PLOTROWS, N_AGENTS, 1 * N_AGENTS +  i_agent + 1)
                plt.plot(self.time_stamps, agent.trajectory.long_speed)
                #plt.ylim(-1, 2)
                if i_agent == 0:
                    plt.ylabel('v (m/s)')
                # acceleration
                plt.subplot(N_PLOTROWS, N_AGENTS, 2 * N_AGENTS +  i_agent + 1)
                plt.plot(self.time_stamps, agent.trajectory.long_acc)
                plt.ylim(-6, 6)
                if i_agent == 0:
                    plt.ylabel('a (m/s^2)') """
            N_PLOTROWS = 4
            distance_CPs = []
            axs = fig.subplots(N_PLOTROWS, 1)
            for i_agent, agent in enumerate(self.agents):
                
                # acceleration
                axs[0].plot(self.time_stamps, agent.trajectory.long_acc, 
                         '-' + agent.plot_color)
                axs[0].set_ylabel('a (m/s^2)') 
                
                # speed
                axs[1].plot(self.time_stamps, agent.trajectory.long_speed, 
                         '-' + agent.plot_color)
                axs[1].set_ylabel('v (m/s)') 
                
                # distance to conflict point
                # - get signed distances to CP
                vectors_to_CP = self.conflict_point - agent.trajectory.pos.T
                yaw_angle = agent.trajectory.yaw_angle[0] # constant throughout in SC scenario
                yaw_vector = np.array((math.cos(yaw_angle), math.sin(yaw_angle)))
                distance_CPs.append(np.dot(vectors_to_CP, yaw_vector))
                # - get CS entry/exit times
                in_CS_idxs = np.nonzero(np.abs(distance_CPs[i_agent]) 
                                        <= SHARED_PARAMS.d_C)[0]
                if len(in_CS_idxs) > 0:
                    t_en = self.time_stamps[in_CS_idxs[0]]
                    t_ex = self.time_stamps[in_CS_idxs[-1]]
                else:
                    t_en = math.nan
                    t_ex = math.nan
                # - illustrate when agent is in CS
                axs[2].fill(np.array((t_en, t_ex, t_ex, t_en)), 
                         np.array((-1, -1, 1, 1)) * SHARED_PARAMS.d_C, 
                         color = agent.plot_color, alpha = 0.3,
                         edgecolor = None)
                # - horizontal lines
                if i_agent == 0:
                    axs[2].axhline(SHARED_PARAMS.d_C, 
                                color='r', linestyle='--', lw=0.5)
                    axs[2].axhline(-SHARED_PARAMS.d_C, 
                                color='r', linestyle='--', lw=0.5)
                    axs[2].axhline(0, color='k', linestyle=':')
                # - plot the distance itself
                axs[2].plot(self.time_stamps, distance_CPs[i_agent], 
                         '-' + agent.plot_color)
                axs[2].set_ylim(-5, 5)
                axs[2].set_ylabel('$d_{CP}$ (m)') 
                
            # distance margin to agent collision
            axs[3].axhline(0, color='k', linestyle=':')
            coll_margins, coll_idxs = \
                get_sc_agent_collision_margins(distance_CPs[0], 
                                               distance_CPs[1],
                                               SHARED_PARAMS.d_C)
            axs[3].plot(self.time_stamps, coll_margins, 'k-')
            axs[3].plot(self.time_stamps[coll_idxs], 
                     coll_margins[coll_idxs], 'r-')
            axs[3].set_ylim(-1, 10)
            axs[3].set_ylabel('$d_{coll}$ (m)')
            axs[3].set_xlabel('t (s)')      
            

        if times_to_ca:
            # - time left to conflict area entry/exit
            plt.figure('Time left to conflict area')
            plt.clf()
            for i_agent, agent in enumerate(self.agents):
                plt.subplot(1, N_AGENTS, i_agent + 1)
                plt.plot(self.time_stamps, agent.states.time_left_to_CA_entry)
                plt.plot(self.time_stamps, agent.states.time_left_to_CA_exit)
                plt.title('Agent %s' % agent.name)
                if i_agent == 0:
                    plt.ylabel('Time left (s)')
                else:
                    plt.legend(('To CA entry', 'To CA exit'))

        plt.show()



### just some test code

if __name__ == "__main__":

    # scenario basics
    NAMES = ('P', 'V')
    CTRL_TYPES = (CtrlType.SPEED, CtrlType.ACCELERATION) 
    
    # scenario
    GOALS = np.array([[0, 5], [-50, 0]])
    SCE_BASELINE = 0 # "baseline" kinematics
    SCE_KEIODECEL = 1 # a deceleration scenario from the Keio study
    SCENARIO = SCE_BASELINE
    if SCENARIO == SCE_BASELINE:
        INITIAL_POSITIONS = np.array([[0,-5], [40, 0]])
        SPEEDS = np.array((0, 10))
        CONST_ACCS = (None, None)
        
    elif SCENARIO == SCE_KEIODECEL:
        INITIAL_POSITIONS = np.array([[0,-2.5], [13.9*2.29, 0]])
        SPEEDS = np.array((0, 13.9))
        stop_dist = INITIAL_POSITIONS[1][0] - SHARED_PARAMS.d_C
        # fix car behaviour to yielding, and simplify to only a single speed
        # increase option for the pedestrian
        CONST_ACCS = (None, -SPEEDS[1] ** 2 / (2 * stop_dist))
        DEFAULT_PARAMS.ctrl_deltas = np.array([0, 1.3])
        i_NO_ACTION = np.argwhere(DEFAULT_PARAMS.ctrl_deltas == 0)[0][0]
        N_ACTIONS = len(DEFAULT_PARAMS.ctrl_deltas)
    
    # set parameters and optional assumptions
    AFF_VAL_FCN = True
    (params, params_k) = get_default_params(oVA = AFF_VAL_FCN)
    #params.T_delta = 30
    #params.V_ny = -60
    #params.T_P = 1
    optional_assumptions = get_assumptions_dict(default_value = False,
                                                oVA = AFF_VAL_FCN,
                                                oBEo = False, 
                                                oBEv = False, 
                                                oAI = False,
                                                oEA = False, 
                                                oAN = False)  
    
    # run simulation
    SNAPSHOT_TIMES = (None, None)
    sc_simulation = SCSimulation(
            CTRL_TYPES, GOALS, INITIAL_POSITIONS, initial_speeds = SPEEDS, 
            end_time = 8, optional_assumptions = optional_assumptions,
            const_accs = CONST_ACCS, agent_names = NAMES, 
            params = params, snapshot_times = SNAPSHOT_TIMES, time_step=0.1)
    sc_simulation.run()
    
    # plot and give some results feedback
    sc_simulation.do_plots(
            trajs = True, action_val_ests = True, surplus_action_vals = True, 
            kinem_states = True, beh_accs = True, beh_probs = True, action_vals = True, 
            sensory_prob_dens = False, beh_activs = True)
    for agent in sc_simulation.agents:
        ca_entered = np.nonzero(np.linalg.norm(agent.trajectory.pos, axis = 0) 
                                <= SHARED_PARAMS.d_C)[0]
        if len(ca_entered) == 0:
            print('Agent %s did not enter the conflict area.' % agent.name)
        else:
            print('Agent %s entered conflict area at t = %.2f s' 
                  % (agent.name, sc_simulation.time_stamps[ca_entered[0]]))
    









        

