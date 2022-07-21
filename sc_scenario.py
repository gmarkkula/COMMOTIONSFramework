import warnings
import math
import numpy as np
from numpy.random import default_rng
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
import sc_scenario_perception
 

NEW_AFF_VAL_CALCS = True


class OptionalAssumption(Enum):
    oVA = 'oVA'
    oVAa = 'oVAa'
    oVAl = 'oVAl'
    oSNc = 'oSNc'
    oSNv = 'oSNv'
    oPF = 'oPF'
    oEA = 'oEA'
    oDA = 'oDA'
    oAN = 'oAN'
    oBEo = 'oBEo'
    oBEv = 'oBEv'
    oBEc = 'oBEc'
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

def get_assumptions_dict_from_string(string):
    """
    Return a dictionary with all the members of OptionalAssumption as keys, 
    setting the values to True if the string corresponding to the assumption
    is found anywhere in the input string.

    """
    assumptions_dict = {}
    for assumption in OptionalAssumption:
        assumptions_dict[assumption] = (assumption.value in string)
    return assumptions_dict

class DerivedAssumption(Enum):
    dBE = 'dBE'
    

class SimStopCriterion(Enum):
    ACTIVE_AGENT_HALFWAY_TO_CS = 'at least one actively behaving agent halfway to conflict space'
    ACTIVE_AGENT_IN_CS = 'at least one actively behaving agent in conflict space'
    AGENT_IN_CS = 'at least one agent in conflict space'
    BOTH_AGENTS_HAVE_MOVED = 'both agents have moved since simulation start'
    BOTH_AGENTS_STOPPED = 'both agents stopped'
    BOTH_AGENTS_EXITED_CS = 'both agents exited conflict space'
    

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
DEFAULT_PARAMS.tau_d = 1 # m; std dev of sensory observation noise (for oSNc)
DEFAULT_PARAMS.tau_theta = 0.05 # rad; std dev of sensory observation noise (for oSNc)
DEFAULT_PARAMS.c_tau = 0.01 # scale factor for tau if oPF is not enabled
DEFAULT_PARAMS.H_e = 1.5 # m; height over ground of the eyes of an observed doing oSNv perception
DEFAULT_PARAMS.sigma_xdot = 0.1 # m/s; std dev of speed process noise in Kalman filtering (oPF)
DEFAULT_PARAMS.T = 0.2 # action value accumulator / low-pass filter relaxation time (s)
#DEFAULT_PARAMS.Tprime = DEFAULT_PARAMS.T  # behaviour value accumulator / low-pass filter relaxation time (s)
DEFAULT_PARAMS.xi_th = 0.0003 # decision evidence threshold
DEFAULT_PARAMS.beta_O = 1 # scaling of action observation evidence in behaviour belief activation (no good theoretical reason for it not to be =1)
DEFAULT_PARAMS.beta_V = 160 # scaling of value evidence in behaviour belief activation
DEFAULT_PARAMS.T_O1 = 0.1 # "sampling" time constant for behaviour observation (s)
DEFAULT_PARAMS.T_Of = 0.5 # "forgetting" time constant for behaviour observation (s)
DEFAULT_PARAMS.sigma_O = 0.01 # std dev of behaviour observation noise (m)
DEFAULT_PARAMS.sigma_V = 0.1 # action value noise in evidence accumulation
#DEFAULT_PARAMS.sigma_Vprime = DEFAULT_PARAMS.sigma_V # behaviour value noise in evidence accumulation
DEFAULT_PARAMS.DeltaV_th_rel = 0.001 # action decision threshold when doing evidence accumulation, in multiples of squashed V_free
DEFAULT_PARAMS.DeltaT = 0.5 # action duration (s)
DEFAULT_PARAMS.T_P = DEFAULT_PARAMS.DeltaT # prediction time (s)
DEFAULT_PARAMS.T_s = 0 # safety margin (post encroachment) time (s)
DEFAULT_PARAMS.D_s = 0 # safety margin distance (m)
DEFAULT_PARAMS.T_delta = 30 # s; half-life of delay-discounted value
DEFAULT_PARAMS.V_0_rel = 4 # scale of value squashing function, in multiples of V_free
DEFAULT_PARAMS.V_ny_rel = 0 # value function term for non-yielding, in multiples of V_free
DEFAULT_PARAMS.thetaDot_0 = 0.001 # rad/s; minimum value for looming to be aversive
DEFAULT_PARAMS.thetaDot_1 = 0.1 # rad/s; looming at which the magnitude of the negative looming utility equals V_free
DEFAULT_PARAMS.ctrl_deltas = np.array([-1, -0.5, 0, 0.5, 1]) # available speed/acc change actions, magnitudes in m/s or m/s^2 dep on agent type

# default gains for affordance-based value function
DEFAULT_PARAMS_K_VA = {}
# - speed-controlling agent
DEFAULT_PARAMS_K_VA[CtrlType.SPEED] = commotions.Parameters()
FREE_SPEED_SPEED_CTRL = 1.5
sc_scenario_helper.set_val_gains_for_free_speed(
    DEFAULT_PARAMS_K_VA[CtrlType.SPEED], FREE_SPEED_SPEED_CTRL)
DEFAULT_PARAMS_K_VA[CtrlType.SPEED]._da = 0.5 # gives sensible-looking acceleration from standstill
# - acceleration-controlling agent
DEFAULT_PARAMS_K_VA[CtrlType.ACCELERATION] = commotions.Parameters()
FREE_SPEED_ACC_CTRL = 10
sc_scenario_helper.set_val_gains_for_free_speed(
    DEFAULT_PARAMS_K_VA[CtrlType.ACCELERATION], FREE_SPEED_ACC_CTRL)
DEFAULT_PARAMS_K_VA[CtrlType.ACCELERATION]._da = 0.5 # gives sensible-looking acceleration from standstill

# default gains for original, non-affordance-based value function
DEFAULT_PARAMS_K_NVA = {}
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED] = commotions.Parameters()
sc_scenario_helper.set_val_gains_for_free_speed(
    DEFAULT_PARAMS_K_NVA[CtrlType.SPEED], FREE_SPEED_SPEED_CTRL)
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED]._c = 1   
DEFAULT_PARAMS_K_NVA[CtrlType.SPEED]._e = 0   
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION] = commotions.Parameters()
sc_scenario_helper.set_val_gains_for_free_speed(
    DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION], FREE_SPEED_ACC_CTRL)
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._da = 0.01
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._sc = 1    
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._sg = 0
DEFAULT_PARAMS_K_NVA[CtrlType.ACCELERATION]._e = 0   
    
def get_default_params(oVA):
    params = copy.copy(DEFAULT_PARAMS)
    if oVA:
        params_k = copy.deepcopy(DEFAULT_PARAMS_K_VA)
    else:
        params_k = copy.deepcopy(DEFAULT_PARAMS_K_NVA)
    return (params, params_k)

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
        # store a (correct) "image" of oneself 
        # [not quite generalisable to store the collision distance as part of 
        #  this image, since it is a function of the other agent also...]
        self.coll_dist = sc_scenario_helper.get_agent_coll_dist(
            self.length, self.other_agent.width)
        self.self_image = SCAgentImage(ctrl_type = self.ctrl_type, 
                                       params = self.params, 
                                       v_free = self.v_free,
                                       g_free = self.g_free,
                                       V_free = self.V_free,
                                       coll_dist = self.coll_dist,
                                       eff_width = self.eff_width)
        # store an "image" of the other agent, with parameters assumed same as 
        # own parameters (but for the appropriate ctrl type)
        oth_params = copy.copy(self.params)
        oth_params.V_ny_inf = 0 # not assuming other agent avoids entering conflict space
        oth_params.k = copy.copy(self.params.k_all[self.other_agent.ctrl_type])
        oth_v_free = sc_scenario_helper.get_agent_free_speed(oth_params.k)
        oth_g_free = self.other_agent.g_free
        oth_V_free = self.other_agent.V_free
        oth_coll_dist = sc_scenario_helper.get_agent_coll_dist(
            self.other_agent.length, self.width)
        self.oth_image = SCAgentImage(ctrl_type = self.other_agent.ctrl_type,
                                      params = oth_params, 
                                      v_free = oth_v_free,
                                      g_free = oth_g_free,
                                      V_free = oth_V_free,
                                      coll_dist = oth_coll_dist,
                                      eff_width = self.other_agent.eff_width)
        # allocate vectors for storing internal states
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
        self.states.action_evidence = \
            math.nan * np.ones((self.n_actions, n_time_steps)) # xi_a(t)
        self.states.action_triggered = \
            np.full((self.n_actions, n_time_steps), False) 
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
        self.states.thetaDot = math.nan * np.ones(n_time_steps)
        # set initial values for states that depend on the previous time step
        self.states.est_action_vals[:, -1] = 0
        self.states.action_evidence[:, -1] = 0
        self.states.beh_activ_V_given_actions[:, :, -1] = 0
        self.states.beh_activ_O[:, -1] = 0
        self.states.beh_activ_O[i_CONSTANT, -1] = 0
        # prepare a separate array for storing distances to the conflict point
        self.signed_CP_dists = np.full(n_time_steps, math.nan)
        
        # specify where the two agents' paths intersect, if it has not already 
        # been done
        if not hasattr(self.simulation, 'conflict_point'):
            self.simulation.conflict_point = np.array((0, 0))
            # make sure the SC scenario has been correctly set up
            apparent_conflict_point = \
                commotions.get_intersection_of_lines(
                    self.get_current_kinematic_state().pos, self.goal, 
                    self.other_agent.get_current_kinematic_state().pos, 
                    self.other_agent.goal)
            assert(np.linalg.norm(self.simulation.conflict_point
                                 - apparent_conflict_point) < 0.0001)

    
    def add_sc_state_info(self, state, coll_dist):
        state.signed_CP_dist = (
            sc_scenario_helper.get_signed_dist_to_conflict_pt(
                self.simulation.conflict_point, state))
        state = sc_scenario_helper.add_entry_exit_times_to_state(
                state, coll_dist)
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
        self.snapshot_axs[0].plot((state.pos[0], vect_pos[0]),
                              (state.pos[1], vect_pos[1]), '-',
                              color=plot_color, alpha=alpha_val)
        self.snapshot_axs[0].plot(state.pos[0], state.pos[1], 'o', 
                              color=plot_color, alpha=alpha_val)
        if speed_label:
            text_pos = state.pos + 2 * speed_vect
            self.snapshot_axs[0].text(text_pos[0], text_pos[1], 
                                  '(%.2f, %.2f)' % (state.long_speed,
                                                    state.long_acc),
                                  color=plot_color, alpha=alpha_val,
                                  ha='center', va='center', size=8)
            
            
    def add_snapshot_info(self, snapshot_str, snapshot_loc, snapshot_color):
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
        self.snapshot_axs[0].text(text_x, text_y, snapshot_str, 
                              transform=self.snapshot_axs[0].transAxes,
                              ha=halign, va=valign, fontsize=7,
                              color=snapshot_color)
    
    
    def prep_for_snapshot_details(self, is_act_val_details, 
                                  i_action=None, i_beh=None):
        self.curr_snapshot_is_act_val = is_act_val_details
        self.i_curr_snapshot_action = i_action
        self.i_curr_snapshot_beh = i_beh
        
        
    def store_snapshot_details(self, snapshot_details):
        if self.curr_snapshot_is_act_val:
            self.snapshot_act_val_details[self.i_curr_snapshot_action, 
                                          self.i_curr_snapshot_beh] = snapshot_details
    
    
    def noisy_lp_filter(self, T, sigma, prevXhat, currXtilde):
        f = self.simulation.settings.time_step / T
        currXhat = (1 - f) * prevXhat + f * currXtilde
        if sigma > 0:
            currXhat += self.rng.standard_normal(len(prevXhat)) * sigma * math.sqrt(
                self.simulation.settings.time_step)
        return currXhat
    
    
    def prepare_for_action_update(self):
        """ Override the base class method with some per-timestep 
            precalculation - done here so that both agents can access these
            for both agents in do_action_update().
        """
        self.curr_state = self.get_current_kinematic_state()
        self.curr_state.long_acc = \
            self.trajectory.long_acc[self.simulation.state.i_time_step-1]
        self.curr_state = self.add_sc_state_info(self.curr_state, self.coll_dist)
        self.signed_CP_dists[self.simulation.state.i_time_step] = \
            self.curr_state.signed_CP_dist
        if (self.inhibit_first_pass_before_time != None
            and self.simulation.state.time < self.inhibit_first_pass_before_time):
            self.params.V_ny_inf = -math.inf
        else:
            self.params.V_ny_inf = 0


    def do_action_update(self):
        """Do the action update for the agent. 
        """

        i_time_step = self.simulation.state.i_time_step
        time_step = self.simulation.settings.time_step
        time_stamp = self.simulation.time_stamps[i_time_step]
        
        # is this agent just supposed to keep constant speed?
        if self.const_acc != None:
            if type(self.const_acc) is tuple:
                # piecewise constant acceleration
                const_acc = 0
                for cmd in self.const_acc:
                    if self.simulation.state.time >= cmd[0]:
                        const_acc = cmd[1]
                    else:
                        break
                self.trajectory.long_acc[i_time_step] = const_acc
                return
            else:
                # single constant acceleration
                self.trajectory.long_acc[i_time_step] = self.const_acc
                return
        elif (self.zero_acc_after_exit 
              and self.curr_state.signed_CP_dist < -self.coll_dist):
            self.trajectory.long_acc[i_time_step] = 0
            return
        
        # update this agent's current perception of the other agent
        # - save the percept from the previous time step
        self.perception.prev_perc_oth_state = copy.deepcopy(
            self.perception.perc_oth_state)
        # - do the perception update
        self.perception.update(i_time_step, self.curr_state, 
                               self.other_agent.curr_state)
        # - shouldn't really be needed, but is now given the imperfections
        # - mentioned in .get_access_order_values_for_agent_v02()
        self.perception.perc_oth_state.long_acc = self.other_agent.curr_state.long_acc

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
            np.nonzero(proj_signed_dist_to_CP < self.coll_dist)[0]
        if len(i_time_steps_entered) == 0:
            # not currently projected to enter he conflict area within the simulation duration
            self.states.time_left_to_CA_entry[i_time_step] = math.inf
        else:
            self.states.time_left_to_CA_entry[i_time_step] = \
                i_time_steps_entered[0] * time_step
        # - exit
        i_time_steps_exited = \
            np.nonzero(proj_signed_dist_to_CP < -self.coll_dist)[0]
        if i_time_steps_exited.size == 0:
            # not currently projected to exit the conflict area within the simulation duration
            self.states.time_left_to_CA_exit[i_time_step] = math.inf
        else:
            self.states.time_left_to_CA_exit[i_time_step] = \
                i_time_steps_exited[0] * time_step
                
        if not self.oth_image.eff_width is None:
            # calculate and store looming state information - not used except for
            # visualisation
            self.states.thetaDot[i_time_step] = sc_scenario_helper.get_agent_optical_exp(
                self.curr_state, self.other_agent.curr_state, self.oth_image)

        # calculate the accelerations needed for the different behaviors of the 
        # other agent, as of the current time 
        # - constant behavior
        self.states.beh_long_accs[i_CONSTANT, i_time_step] = 0  
        # - use helper function to get other agent's expected accelerations to
        #   pass in front of or behind me, given my current position and speed
        #   (but not my current acceleration)
        (self.states.beh_long_accs[i_PASS1ST, i_time_step], 
         self.states.beh_long_accs[i_PASS2ND, i_time_step]) = \
             sc_scenario_helper.get_access_order_accs(
                     ego_image=self.oth_image, 
                     ego_state=self.perception.perc_oth_state, 
                     oth_image=self.self_image,
                     oth_state=self.curr_state, 
                     consider_oth_acc=self.assumptions[OptionalAssumption.oVAa] 
                     and self.ctrl_type == CtrlType.ACCELERATION)
             
        # determine which behaviours are valid at this time step
        # - the helper function above returns nan if behaviour is invalid for 
        # - this time step
        beh_is_valid = np.invert(np.isnan(
                self.states.beh_long_accs[:, i_time_step]))
        # - is the constant behaviour valid for this time step?
        if (self.assumptions[DerivedAssumption.dBE] and 
            (not self.assumptions[OptionalAssumption.oBEc])
            and (beh_is_valid[i_PASS1ST] or beh_is_valid[i_PASS2ND])):
            # no the constant behaviour is not valid, because we are 
            # estimating behaviours, oBEc is not enabled, and at least one of 
            # the other (non-constant-speed) behaviours is valid
            beh_is_valid[i_CONSTANT] = False
            self.states.beh_long_accs[i_CONSTANT, i_time_step] = math.nan
        # - are the pass 1st/2nd behaviours valid? (are we estimating behs?)
        elif not self.assumptions[DerivedAssumption.dBE]:
            for i_beh in (i_PASS1ST, i_PASS2ND):
                beh_is_valid[i_beh] = False
                self.states.beh_long_accs[i_beh, i_time_step] = math.nan
         
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
        self.do_snapshot_now = False
        if self.doing_snapshots:
            if np.amin(np.abs(np.array(self.snapshot_times) - time_stamp)) < 0.00001:
                self.do_snapshot_now = True
                # set up the figure(s)
                fig_axs = []
                # - which snapshots to do?
                snapshots = []
                snapshots.append('') # basic state info
                if self.detailed_snapshots:
                    snapshots.append('(kinematics) ') # kinematics details
                    if self.assumptions[OptionalAssumption.oVAl]:
                        snapshots.append('(looming) ') # looming details
                    # prepare for storing snapshot action value details
                    # (only supporting storing details for the last snapshot,
                    # if more than one snapshots in a simulation)
                    self.snapshot_act_val_details = np.full((self.n_actions, 
                                                             N_BEHAVIORS), None)
                for snapshot in snapshots:
                    fig_name = 'Snapshot %sfor %s at t = %.2f s' % (
                        snapshot, self.name, time_stamp)
                    fig = plt.figure(num=fig_name, figsize=(15, 10))
                    fig.clf()
                    fig_axs.append(fig.subplots(nrows=N_BEHAVIORS, 
                                                ncols=self.n_actions,
                                                sharex=True, sharey=True))
            
        # now loop over all combinations of own actions and other's behaviors, 
        # and get values from both agents' perspectives
        for i_action in range(self.n_actions):
            for i_beh in range(N_BEHAVIORS):
                # is this behaviour valid for this time step? 
                # (if not leave values as NaNs)
                if beh_is_valid[i_beh]:
                    # doing snapshot?
                    if self.do_snapshot_now:
                        # create a temporary list of the snapshot axes for
                        # the current action-behaviour combination
                        # (used also for more detailed snapshot plotting)
                        self.snapshot_axs = []
                        for fig_ax in fig_axs:
                            self.snapshot_axs.append(fig_ax[i_beh, i_action])
                        # plot basic state snapshot info
                        self.plot_state_snapshot(self.curr_state, 
                                                 self.plot_color, 
                                                 alpha=True)
                        self.plot_state_snapshot(self.other_agent.curr_state, 
                                                 'lightgray', 
                                                 alpha=True)
                        self.plot_state_snapshot(self.perception.perc_oth_state, 
                                                 self.other_agent.plot_color, 
                                                 alpha=True)
                        self.plot_state_snapshot(pred_own_states[i_action],
                                                 self.plot_color, 
                                                 speed_label = True)
                        self.plot_state_snapshot(pred_oth_states[i_beh],
                                                 self.other_agent.plot_color, 
                                                 speed_label = True)
                    # what type of value functions (affordance-based or not?)
                    if self.assumptions[OptionalAssumption.oVA]:
                        # affordance-based value functions
                        # get value for me of this action/behavior combination,
                        # storing both the per-access-order values and the max value
                        # of those    
                        if self.detailed_snapshots:
                            # prepare for receiving and storing snapshot 
                            # details about the action value estimates
                            self.prep_for_snapshot_details(True, i_action, i_beh)                    
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
                        if self.detailed_snapshots:
                            # not storing snapshot details for behaviour value estimates
                            self.prep_for_snapshot_details(False)    
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
                                            pred_own_states[self.i_no_action]) )
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
                                                             pred_own_states[self.i_no_action], 
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
                            self.params.T, self.params.sigma_V,
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
                        (1 - self.simulation.settings.time_step / self.params.T_Of) 
                        * self.states.beh_activ_O[i_beh, i_time_step-1])
                    if i_time_step > 0:
                        self.states.sensory_probs_given_behs[
                            i_beh, i_time_step] = \
                            self.get_prob_of_current_state_given_beh(i_beh)
                        self.states.beh_activ_O[i_beh, i_time_step] += \
                            (self.simulation.settings.time_step / self.params.T_O1) \
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
                        self.states.beh_activations_given_actions[
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

        # update the accumulative (low-pass filtered) estimates of action value
        self.states.est_action_vals[:, i_time_step] = self.noisy_lp_filter(
            self.params.T, self.params.sigma_V,
            self.states.est_action_vals[:, i_time_step-1],
            self.states.mom_action_vals[:, i_time_step])
        
        # get surplus action values (compared to non-action)
        self.states.est_action_surplus_vals[:, i_time_step] = \
            self.states.est_action_vals[:, i_time_step] \
            - self.states.est_action_vals[self.i_no_action, i_time_step]

        # should any action be taken?
        if self.assumptions[OptionalAssumption.oDA]:
            # update the accumulated evidence for actions
            dxidt = self.states.est_action_surplus_vals[:, i_time_step]
            self.states.action_evidence[:, i_time_step] = (
                self.states.action_evidence[:, i_time_step-1]
                + self.simulation.settings.time_step * dxidt)
            self.states.action_evidence[:, i_time_step] = np.maximum(
                0, self.states.action_evidence[:, i_time_step])
            # check if the highest action evidence is above threshold
            i_best_action = np.argmax(
                self.states.action_evidence[:, i_time_step])
            take_best_action = self.states.action_evidence[
                i_best_action, i_time_step] > self.params.xi_th
        else:
            # if not using decision evidence accumulation, check if the highest
            # estimated surplus action value is above threshold
            i_best_action = np.argmax(
                self.states.est_action_surplus_vals[:, i_time_step])
            take_best_action = self.states.est_action_surplus_vals[
                i_best_action, i_time_step] > self.params.DeltaV_th
                
        if take_best_action:
            # add action to the array of future acceleration values
            self.add_action_to_acc_array(self.action_long_accs, i_best_action, \
                self.simulation.state.i_time_step)
            # remember what action was taken when
            self.states.action_triggered[i_best_action, i_time_step] = True
            # reset accumulators if needed
            if self.assumptions[OptionalAssumption.oEA]:
                # reset the value accumulators
                self.states.est_action_vals[:, i_time_step] = 0
                self.states.beh_activ_V_given_actions[:, :, i_time_step] = 0
            if self.assumptions[OptionalAssumption.oDA]:
                # reset the decision evidence accumulators
                self.states.action_evidence[:, i_time_step] = 0

        # set long acc in actual trajectory
        self.trajectory.long_acc[i_time_step] = self.action_long_accs[i_time_step]
        
    def squash_value(self, input_values):
        return np.tanh(input_values / self.params.V_0)

    def get_access_order_values_for_agent_v02(self, 
                                              ego_image, 
                                              ego_curr_state, ego_pred_state, 
                                              oth_image, 
                                              oth_curr_state, oth_pred_state,
                                              snapshot_color, snapshot_loc,
                                              plot_snapshot_deets):
        
        # skipping goal stopping for now - I think the sensible way of adding
        # it back in later is to include an independent term for it, valuating
        # the needed manoeuvre for stopping just like this function valuates
        # the needed manoevure for achieving each access order
       
        # get the effective average acceleration or jerk (depending on agent 
        # type) in the action/prediction interval 
        # [NB: The below sort of assumes that the ego agent here is =self, which is
        # only true if the calling function is _for_me_v02(), not _for_other_v02()
        # - in the latter case the calculation for acceleration-controlling 
        # agents will not be entirely in line with the assumption of constant
        # behaviour acceleration during the prediction interval.]
        if ego_image.ctrl_type is CtrlType.SPEED:
            action_acc0 = (ego_pred_state.long_speed 
                          - ego_curr_state.long_speed) / ego_image.params.DeltaT
            action_jerk = 0
        else:
            action_acc0 = ego_curr_state.long_acc
            action_jerk = (ego_pred_state.long_acc
                           - ego_curr_state.long_acc) / ego_image.params.DeltaT
        
        # call helper function to get needed manoeuvring and delay times for
        # each access order, starting from this state
        implications = sc_scenario_helper.get_access_order_implications(
                ego_image=ego_image, ego_state=ego_pred_state, 
                oth_image=oth_image, oth_state=oth_pred_state, 
                consider_oth_acc=self.assumptions[OptionalAssumption.oVAa])
        
        # inhibit early entry (slightly inelegant solution for emulating the 
        # "first vehicle" in the HIKER scenarios)
        if (ego_image.params.V_ny_inf != 0 and ego_pred_state.signed_CP_dist 
            < ego_image.coll_dist + ego_image.params.D_s):
            early_entry_value = ego_image.params.V_ny_inf
        else:
            early_entry_value = 0
        
        # get values for the respective access orders
        access_order_values = np.full(N_ACCESS_ORDERS, math.nan)
        if NEW_AFF_VAL_CALCS:
            
            # get the assumed-constant accelerations of the other agent in the 
            # action/prediction interval as well as beyond it
            # - action/prediction interval: the acceleration that achieves the 
            # - correct speed at t + T_p
            oth_first_acc = ((oth_pred_state.long_speed 
                             - oth_curr_state.long_speed) 
                             / ego_image.params.DeltaT)
            # - after action/prediction interval - depends on model assumptions
            if self.assumptions[OptionalAssumption.oVAa]:
                oth_cont_acc = oth_pred_state.long_acc
            else:
                oth_cont_acc = 0
            
            # call helper function to get dict with unsquashed access order 
            # values (and any details needed for snapshots)
            access_ord_values_dict = sc_scenario_helper.get_access_order_values(
                ego_image = ego_image, ego_curr_state = ego_curr_state,  
                action_acc0 = action_acc0, action_jerk = action_jerk, 
                oth_image = oth_image, oth_curr_state = oth_curr_state,
                oth_first_acc = oth_first_acc, oth_cont_acc = oth_cont_acc, 
                access_ord_impls = implications, 
                consider_looming = self.assumptions[OptionalAssumption.oVAl],
                return_details = self.do_snapshot_now)     
            
            # do value squashing and store in output array
            for access_order in AccessOrder:
                value = self.squash_value(access_ord_values_dict[access_order].value 
                                          + early_entry_value)
                access_order_values[access_order.value] = value
            
            # doing a snapshot?
            if self.do_snapshot_now:
                if self.detailed_snapshots:
                    self.store_snapshot_details(access_ord_values_dict)
                snapshot_str = ''
                for access_order in AccessOrder:
                    snapshot_str += ('(%.1f m/s^2, %.2f s, %.1f s)' %
                                    (implications[access_order].acc,
                                     implications[access_order].T_acc,
                                     implications[access_order].T_dw) + '\n'
                                    + 'V = %.2f' % access_ord_values_dict[access_order].value 
                                    + '\n')
                    deets = access_ord_values_dict[access_order].details
                    if not (deets is None):
                        snapshot_str += '('
                        for i_phase, kinem_value in enumerate(deets.phase_kinem_values):
                            snapshot_str += f'{kinem_value:.1f}, '
                        looming_value = np.sum(deets.phase_looming_values)
                        snapshot_str += f'{deets.inh_access_value:.1f}, '
                        snapshot_str += f'{looming_value:.2f})\n'
                        if plot_snapshot_deets:
                            if access_order == AccessOrder.EGOFIRST:
                                color = 'g'
                            else:
                                color = 'r'
                            self.snapshot_axs[1].plot(deets.time_stamps, 
                                                      deets.speeds,
                                                      '--', lw=2, color=color, 
                                                      alpha=0.5)
                            self.snapshot_axs[1].plot(deets.time_stamps, 
                                                      deets.cp_dists,
                                                      '-', lw=2, color=color, 
                                                      alpha=0.5)
                            self.snapshot_axs[1].plot(deets.time_stamps, 
                                                      deets.oth_speeds,
                                                      '--', lw=1, color='gray', 
                                                      alpha=0.5)
                            self.snapshot_axs[1].plot(deets.time_stamps, 
                                                      deets.oth_cp_dists,
                                                      '-', lw=1, color='gray', 
                                                      alpha=0.5)
                            if self.assumptions[OptionalAssumption.oVAl]:
                                self.snapshot_axs[2].plot(deets.time_stamps,
                                                          deets.thetaDots,
                                                          '-', lw=2, color=color, 
                                                          alpha=0.5)
                                
        else:
            
            # get value of the action itself
            action_value = sc_scenario_helper.get_value_of_const_jerk_interval(
                v0 = ego_curr_state.long_speed, a0 = action_acc0, j = action_jerk, 
                T = ego_image.params.DeltaT, k = ego_image.params.k)
            
            # get value of any looming experienced at the predicted moment
            looming_value = 0
            if self.assumptions[OptionalAssumption.oVAl]:
                ttc = sc_scenario_helper.get_time_to_sc_agent_collision(
                            ego_pred_state, oth_pred_state, consider_acc=False)
                if ttc > 0 and ttc < math.inf:
                    # on a collision course, get looming and calculate value
                    thetaDot = sc_scenario_helper.get_agent_optical_exp(
                        ego_pred_state, oth_pred_state, oth_image)
                    if thetaDot > ego_image.params.thetaDot_0:
                        looming_value = (-ego_image.V_free
                                         * (thetaDot - ego_image.params.thetaDot_0)
                                         / (ego_image.params.thetaDot_1 
                                            - ego_image.params.thetaDot_0))  
            
            # get constant value for remainder of trip after action and possible
            # interaction, i.e., V_free
            post_value = ego_image.V_free
            
            # get the estimated time needed for the agent to regain free speed, 
            # if not already at it
            if ego_image.ctrl_type is CtrlType.SPEED:
                agent_time_to_v_free = ego_image.params.DeltaT
            else:
                agent_time_to_v_free = sc_scenario_helper.ACC_CTRL_REGAIN_SPD_TIME
            
            # loop over the access orders and get value for each
            if self.do_snapshot_now:
                snapshot_str = ''
            for access_order in AccessOrder:
                if np.isnan(implications[access_order].acc):
                    # access order not valid from this state
                    value = -math.inf
                else:
                    # valuation of the action/prediction time interval, and any
                    # looming experienced after the prediction interval
                    value = action_value + looming_value
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
                    inh_access_value = 0
                    if access_order is AccessOrder.EGOFIRST:
                        if ego_image.params.V_ny_inf != 0:
                            inh_access_value = ego_image.params.V_ny_inf
                        elif ego_image.ctrl_type is CtrlType.ACCELERATION:
                            inh_access_value = ego_image.params.V_ny
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
                                         +'(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f)' % 
                                         (action_value, looming_value, 
                                          ach_access_value, inh_access_value, 
                                          regain_value, final_value) + '\n')
                # squash the value with a sigmoid
                value = self.squash_value(value + early_entry_value)
                # store the value of this access order in the output numpy array
                access_order_values[access_order.value] = value
            
        # snapshot info
        if self.do_snapshot_now:
            self.add_snapshot_info(snapshot_str, snapshot_loc, snapshot_color)
            
        return access_order_values
        
    
    def get_access_order_values_for_me_v02(self, my_pred_state, oth_pred_state):
        access_order_values = self.get_access_order_values_for_agent_v02(
                ego_image = self.self_image, 
                ego_curr_state = self.curr_state, 
                ego_pred_state = my_pred_state, 
                oth_image = self.oth_image,
                oth_curr_state = self.perception.perc_oth_state,
                oth_pred_state = oth_pred_state,
                snapshot_color = self.plot_color,
                snapshot_loc = 'topleft',
                plot_snapshot_deets = self.detailed_snapshots)
        return access_order_values
        
    
    def get_access_order_values_for_other_v02(self, oth_pred_state, my_pred_state):
        access_order_values = self.get_access_order_values_for_agent_v02(
                ego_image = self.oth_image, 
                ego_curr_state = self.perception.perc_oth_state, 
                ego_pred_state = oth_pred_state,
                oth_image = self.self_image,
                oth_curr_state = self.curr_state,
                oth_pred_state = my_pred_state,
                snapshot_color = self.other_agent.plot_color,
                snapshot_loc = 'bottomright',
                plot_snapshot_deets = False)
        return access_order_values


    def get_value_of_state_for_agent(self, own_image, own_state, own_goal, 
                                     oth_image, oth_state, 
                                     snapshot_color, snapshot_loc):

        heading_vector = np.array([math.cos(own_state.yaw_angle), \
            math.sin(own_state.yaw_angle)])
        
        # reward for progress toward goal and speed discomfort cost
        vector_to_goal = own_goal - own_state.pos
        heading_to_goal_vector = vector_to_goal / np.linalg.norm(vector_to_goal)
        heading_toward_goal_component = \
            np.dot(heading_vector, heading_to_goal_vector)
        goal_distance_change_rate = \
            -heading_toward_goal_component * own_state.long_speed
        progress_value = -own_image.params.k._g * goal_distance_change_rate
        spd_disc_value = -own_image.params.k._dv * own_state.long_speed ** 2
        value = progress_value + spd_disc_value
        
        if own_image.ctrl_type is CtrlType.ACCELERATION:
            # acceleration discomfort cost
            acc_disc_value = -own_image.params.k._da * own_state.long_acc ** 2
            value += acc_disc_value
            # cost for acceleration required to stop at goal
            goal_distance = np.linalg.norm(vector_to_goal)
            req_acc_to_goal = -(own_state.long_speed ** 2 / (2 * goal_distance))
            goal_stop_value = -own_image.params.k._sg * req_acc_to_goal ** 2
            value += goal_stop_value
            # priority cost for vehicle of passing first
            pass1st_value = 0  
            if own_image.params.V_ny != 0:
                (__, acc_for_2nd) = sc_scenario_helper.get_access_order_accs(
                    own_image, own_state, oth_image, oth_state, 
                    consider_oth_acc = False)
                if own_state.long_acc > acc_for_2nd:
                    pass1st_value = (own_image.params.V_ny
                                     * (2 - own_state.long_acc / acc_for_2nd)) 
                    value += pass1st_value
        else:
            acc_disc_value = 0
            goal_stop_value = 0
            pass1st_value = 0

        # cost for being on collision course with the other agent
        time_to_agent_collision = \
            sc_scenario_helper.get_time_to_sc_agent_collision(own_state, 
                                                              oth_state,
                                                              consider_acc=False)
        if time_to_agent_collision == 0:
            time_to_agent_collision = TTC_FOR_COLLISION
        if time_to_agent_collision < math.inf:
            if own_image.ctrl_type is CtrlType.SPEED:
                coll_value = -own_image.params.k._c / time_to_agent_collision  
            elif own_image.ctrl_type is CtrlType.ACCELERATION:
                coll_value = -own_image.params.k._sc * (own_state.long_speed \
                    / (2 * time_to_agent_collision)) ** 2  
        else:
            coll_value = 0
        value += coll_value
        
        if self.do_snapshot_now:
            snapshot_str = (f'V = {value:.2f}\n'
                            f'({progress_value:.1f}, {spd_disc_value:.1f},'
                            f' {acc_disc_value:.1f}, {goal_stop_value:.1f},'
                            f' {pass1st_value:.1f}, {coll_value:.1f})')
            self.add_snapshot_info(snapshot_str, snapshot_loc, snapshot_color)
                    
        # squash the value with a sigmoid
        value = self.squash_value(value)
        
        return value


    def get_value_for_me(self, my_state, oth_state, i_action):            
        # cost for making a speed change, if any
        value = -self.params.k._e * self.params.ctrl_deltas[i_action] ** 2
        # add value of the state
        value += self.get_value_of_state_for_agent(
            self.self_image, my_state, self.goal, self.oth_image, oth_state,
            snapshot_color = self.plot_color, snapshot_loc = 'topleft')
        return value


    def get_value_for_other(self, oth_state, my_state, i_beh):
        # POSSIBLE TODO - add cost for the behavior itself
        # - disregarding acceleration discomfort for the other agent, for now at least
        #   (needs a bit of code restructuring, or storing the cost somewhere)
        value = 0
        # add value of the state
        value += self.get_value_of_state_for_agent(
                self.oth_image, oth_state, self.other_agent.goal, 
                self.self_image, my_state,
                snapshot_color = self.other_agent.plot_color,
                snapshot_loc = 'bottomright') 
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
        predicted_state = self.add_sc_state_info(predicted_state, self.coll_dist)
        return predicted_state


    def get_predicted_other_state(self, i_beh):
        # get the longitudinal acceleration for this behavior, if implemented at
        # the current time step
        long_acc_for_this_beh = \
            self.states.beh_long_accs[i_beh, self.simulation.state.i_time_step]
        if math.isnan(long_acc_for_this_beh):
            predicted_state = None
        else:
            # # let the other agent object calculate what its predicted state would
            # # be with this acceleration 
            # predicted_state = self.other_agent.get_future_kinematic_state(
            #         long_acc_for_this_beh, yaw_rate = 0, 
            #         n_time_steps_to_advance = self.n_prediction_time_steps)
            # predicted_state.long_acc = long_acc_for_this_beh
            # # add SC scenario specific state info
            # predicted_state = self.add_sc_state_info(predicted_state)
            pred_speed = (self.perception.perc_oth_state.long_speed 
                          + long_acc_for_this_beh * self.params.T_P)
            # make sure not to predict reversing
            if pred_speed >= 0:
                dist_pred_time = self.params.T_P
            else:
                pred_speed = 0
                assert(long_acc_for_this_beh < 0)
                dist_pred_time = (self.perception.perc_oth_state.long_speed 
                                  / (-long_acc_for_this_beh))
            pred_CP_dist = (self.perception.perc_oth_state.signed_CP_dist
                            - self.perception.perc_oth_state.long_speed 
                            * dist_pred_time
                            - (long_acc_for_this_beh * dist_pred_time ** 2) / 2)
            
            pred_yaw_angle = self.perception.perc_oth_state.yaw_angle
            predicted_state = commotions.KinematicState(pos=None, 
                                                        long_speed=pred_speed,
                                                        yaw_angle=pred_yaw_angle)
            predicted_state.signed_CP_dist = pred_CP_dist
            predicted_state.pos = (
                sc_scenario_helper.get_pos_from_signed_dist_to_conflict_pt(
                    self.simulation.conflict_point, predicted_state))
            predicted_state.long_acc = long_acc_for_this_beh
            sc_scenario_helper.add_entry_exit_times_to_state(
                predicted_state, self.oth_image.coll_dist)
        return predicted_state
        

    def get_prob_of_current_state_given_beh(self, i_beh):
        """ Return probability density for perceived position of other agent,
            given perceived position at last time step and the acceleration for
            behaviour i_beh as estimated on the previous time step. Should only
            be called when i_time_step > 0.

        """
        i_prev_time_step = self.simulation.state.i_time_step-1
        if math.isnan(self.states.beh_long_accs[i_beh, i_prev_time_step]):
            prob_density = 0
        else:
            # retrieve the longitudinal acceleration for this behavior, as estimated on
            # the previous time step
            prev_long_acc_for_this_beh = \
                self.states.beh_long_accs[i_beh, i_prev_time_step]
            # # let the other agent object calculate what its predicted state at the 
            # # current time step would be with this acceleration     
            # expected_curr_state = self.other_agent.get_future_kinematic_state(
            #         prev_long_acc_for_this_beh, yaw_rate = 0, 
            #         n_time_steps_to_advance = 1, 
            #         i_start_time_step = i_prev_time_step)
            # # get the distance between expected and observed position
            # pos_diff = np.linalg.norm(
            #         expected_curr_state.pos 
            #         - self.other_agent.get_current_kinematic_state().pos)
            pred_CP_dist = (self.perception.prev_perc_oth_state.signed_CP_dist
                            - self.perception.prev_perc_oth_state.long_speed 
                            * self.simulation.settings.time_step
                            - (prev_long_acc_for_this_beh 
                               * self.simulation.settings.time_step ** 2) / 2)
            pos_diff = self.perception.perc_oth_state.signed_CP_dist - pred_CP_dist
            # return the probability density for this observed difference
            prob_density = norm.pdf(pos_diff, scale = self.params.sigma_O)
        return max(prob_density, np.finfo(float).eps) # don't return zero probability
                
    
    def get_stop_accs(self):
        """ Return a vector of accelerations that would, for each time step
            in the (alreadly run) simulation have the agent stop
            at its safety distance (D_s).
        """
        stop_dists = (self.signed_CP_dists - self.coll_dist - self.params.D_s)
        stop_accs = -(self.trajectory.long_speed ** 2 / (2 * stop_dists))
        stop_accs[stop_dists <= 0] = np.nan
        return stop_accs
    

    def __init__(self, name, ctrl_type, width, length, simulation, goal_pos, 
                 initial_state, optional_assumptions = {}, 
                 params = None, params_k = None, 
                 noise_seed = None, kalman_prior = None, 
                 inhibit_first_pass_before_time = None, # NB: Currently only supported for oVA models
                 const_acc = None, zero_acc_after_exit = False, 
                 plot_color = 'k',  snapshot_times = None, 
                 detailed_snapshots = False):

        # set control type
        self.ctrl_type = ctrl_type
        
        # set initial state and call inherited init method
        # (no reversing, regardless of agent type)
        super().__init__(name, simulation, goal_pos, \
            initial_state, can_reverse = False, plot_color = plot_color)
            
        # inhibiting passing first before a certain time?
        self.inhibit_first_pass_before_time = inhibit_first_pass_before_time
            
        # is this agent to just keep a constant acceleration?
        # (throughout or after exiting conflict space)
        self.const_acc = const_acc
        self.zero_acc_after_exit = zero_acc_after_exit
        
        # doing any value function snapshots?
        self.snapshot_times = snapshot_times
        self.doing_snapshots = snapshot_times != None
        self.detailed_snapshots = detailed_snapshots
        
        # store any optional assumptions provided by the caller
        self.assumptions = get_assumptions_dict(False)
        for specified_assumption in optional_assumptions.keys():
            self.assumptions[specified_assumption] = \
                optional_assumptions[specified_assumption]

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
        
        # this implementation requires action time equals prediction time
        if self.params.DeltaT != self.params.T_P:
            raise Exception('The SCAgent implementation requires that action'
                            ' time DeltaT and prediction time T_P are equal.')

        # parse the optional assumptions
        if ((not self.assumptions[OptionalAssumption.oSNc]) and 
            (not self.assumptions[OptionalAssumption.oSNv])):
            # no sensory noise
            pos_obs_noise_stddev = 0
            if self.assumptions[OptionalAssumption.oPF]:
                raise Exception('Cannot enable oPF without oSNc or oSNv.')
        else:
            # some form of sensory noise included
            if self.assumptions[OptionalAssumption.oAN]:
                raise Exception('Simultaneous sensory noise (oSN*) and'
                                'accumulator noise (oAN) not supported.')
            # get sensory noise magnitude
            if self.assumptions[OptionalAssumption.oSNc]:
                pos_obs_noise_stddev = self.params.tau_d
            else:
                pos_obs_noise_stddev = self.params.tau_theta
            # assuming Bayesian perceptual filtering?
            if self.assumptions[OptionalAssumption.oPF]:
                if kalman_prior is None:
                    raise Exception('Need to supply a Kalman prior to enable oPF.')
            else:
                # scale down sensory noise if not filtering
                pos_obs_noise_stddev *= self.params.c_tau
        if not self.assumptions[OptionalAssumption.oEA]:
            # no evidence accumulation, implemented by value accumulation 
            # reaching input value in one time step...
            self.params.T = self.simulation.settings.time_step 
            #self.params.Tprime = self.simulation.settings.time_step 
            # ... and decision threshold at zero
            self.params.DeltaV_th_rel = 0
        if self.assumptions[OptionalAssumption.oAN]:
            # initialise the accumulator noise generator
            self.rng = default_rng(seed = noise_seed)
        else:
            self.params.sigma_V = 0
            #self.params.sigma_Vprime = 0
        if self.assumptions[OptionalAssumption.oDA]:
            # doing decision evidence accumulation, so we should not be 
            # thresholding surplus action values
            self.params.DeltaV_th_rel = math.nan
        else:
            # no decision evidence accumulation
            self.params.T_xi = math.nan
            self.params.C_xi = math.nan
        if not self.assumptions[OptionalAssumption.oBEo]:
            self.params.beta_O = 0
        if not self.assumptions[OptionalAssumption.oBEv]:
            self.params.beta_V = 0
        self.assumptions[DerivedAssumption.dBE] = \
            self.assumptions[OptionalAssumption.oBEo] \
            or self.assumptions[OptionalAssumption.oBEv]
        if not self.assumptions[DerivedAssumption.dBE]:
            if self.assumptions[OptionalAssumption.oBEc]:
                warnings.warn('Cannot have oBEc without oBEv or oBEo, so'
                              ' disabling oBEc.')
                self.assumptions[OptionalAssumption.oBEc] = False
        if not self.assumptions[OptionalAssumption.oVA]:
            if self.detailed_snapshots:
                warnings.warn('Detailed snapshots for non-oVA* models not'
                              ' supported, so disabling snapshot details.')
                self.detailed_snapshots = False
            if self.assumptions[OptionalAssumption.oVAa]:
                warnings.warn('Cannot have oVAa without oVA, so disabling oVAa.')
                self.assumptions[OptionalAssumption.oVAa] = False
                
        # get and store the number of actions, and the "non-action" action
        self.n_actions = len(self.params.ctrl_deltas)
        self.i_no_action = np.argwhere(self.params.ctrl_deltas == 0)[0][0]
        
        # store own width and length
        self.width = width
        self.length = length
        
        # store own "effective width" - for simplicity setting to actual width
        self.eff_width = width

        # get and store own free speed
        self.v_free = sc_scenario_helper.get_agent_free_speed(self.params.k)
        
        # get and store g_free and V_free, the value rate and total future 
        # value for the agent of just
        # proceeding at the free speed, without any interaction
        if self.assumptions[OptionalAssumption.oVA]:
            self.g_free = sc_scenario_helper.get_agent_free_value_rate(self.params)
            self.V_free = sc_scenario_helper.get_agent_free_value(self.params)
        else:
            self.g_free = None # not applicable for non-oVA agent
            self.V_free = sc_scenario_helper.get_agent_free_value_rate(self.params)
        
        # get derived parameters 
        self.params.V_0 = self.V_free * self.params.V_0_rel
        self.params.DeltaV_th = (self.squash_value(self.V_free) 
                                 * self.params.DeltaV_th_rel)
        self.params.V_ny = self.V_free * self.params.V_ny_rel
        
        # set up the object that implements perception of the other agent
        self.perception = sc_scenario_perception.Perception(
            simulation = simulation,
            pos_obs_noise_stddev = pos_obs_noise_stddev,
            noise_seed = noise_seed,
            angular_perception = self.assumptions[OptionalAssumption.oSNv],
            ego_eye_height = self.params.H_e,
            kalman_filter = self.assumptions[OptionalAssumption.oPF],
            prior = kalman_prior,
            spd_proc_noise_stddev = self.params.sigma_xdot,
            draw_from_estimate = True)

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

    def __init__(self, ctrl_types, widths, lengths,
                 goal_positions, initial_positions, 
                 initial_speeds = np.array((0, 0)), 
                 start_time = 0, end_time = 10, time_step = 0.1, 
                 optional_assumptions = get_assumptions_dict(False), 
                 params = None, params_k = None,  
                 noise_seeds = (None, None), kalman_priors = (None, None),
                 inhibit_first_pass_before_time = None,
                 const_accs = (None, None), zero_acc_after_exit = False, 
                 agent_names = ('A', 'B'), plot_colors = ('c', 'm'), 
                 snapshot_times = (None, None), detailed_snapshots = False,
                 stop_criteria = ()):

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
                    width = widths[i_agent],
                    length = lengths[i_agent],
                    noise_seed = noise_seeds[i_agent],
                    kalman_prior = kalman_priors[i_agent],
                    inhibit_first_pass_before_time = inhibit_first_pass_before_time,
                    const_acc = const_accs[i_agent],
                    zero_acc_after_exit = zero_acc_after_exit,
                    plot_color = plot_colors[i_agent],
                    snapshot_times = snapshot_times[i_agent],
                    detailed_snapshots = detailed_snapshots)
        
        self.stop_criteria = stop_criteria
            
            
    def after_time_step(self):
        for stop_crit in self.stop_criteria:
            
            if stop_crit == SimStopCriterion.ACTIVE_AGENT_HALFWAY_TO_CS:
                for agent in self.agents:
                    if agent.const_acc == None:
                        halfway_dist = \
                            sc_scenario_helper.get_agent_halfway_to_CS_CP_dist(agent)
                        if agent.curr_state.signed_CP_dist < halfway_dist:
                            self.stop_now = True
                            return
            
            elif stop_crit == SimStopCriterion.ACTIVE_AGENT_IN_CS:
                for agent in self.agents:
                    if agent.const_acc == None:
                        if agent.curr_state.signed_CP_dist < agent.coll_dist:
                            self.stop_now = True
                            return
                        
            elif stop_crit == SimStopCriterion.AGENT_IN_CS:
                for agent in self.agents:
                    if agent.curr_state.signed_CP_dist < agent.coll_dist:
                        self.stop_now = True
                        return
                    
            elif stop_crit == SimStopCriterion.BOTH_AGENTS_HAVE_MOVED:
                found_unmoved_agent = False
                for agent in self.agents:
                    if agent.curr_state.signed_CP_dist == agent.signed_CP_dists[0]:
                        found_unmoved_agent = True
                        break
                if not found_unmoved_agent:
                    self.stop_now = True
                    return
                    
            elif stop_crit == SimStopCriterion.BOTH_AGENTS_STOPPED:
                found_moving_agent = False
                for agent in self.agents:
                    if agent.curr_state.long_speed > 0:
                        found_moving_agent = True
                        break
                if not found_moving_agent:
                    self.stop_now = True
                    return
                
            elif stop_crit == SimStopCriterion.BOTH_AGENTS_EXITED_CS:
                found_non_exited_agent = False
                for agent in self.agents:
                    if agent.curr_state.signed_CP_dist >= -agent.coll_dist:
                        found_non_exited_agent = True
                        break
                if not found_non_exited_agent:
                    self.stop_now = True
                    return
                
            else:
                raise Exception(f'Unexpected simulation stop criterion: {stop_crit}')
    
    
    def after_simulation(self):
        for agent in self.agents:
            ## signed distances to conflict point
            #vectors_to_CP = self.conflict_point - agent.trajectory.pos.T
            #yaw_angle = agent.trajectory.yaw_angle[0] # constant throughout in SC scenario
            #yaw_vector = np.array((math.cos(yaw_angle), math.sin(yaw_angle)))
            #agent.signed_CP_dists = np.dot(vectors_to_CP, yaw_vector)
            # entry time/sample
            ca_entered = np.nonzero(np.linalg.norm(agent.trajectory.pos, axis = 0)
                                    <= agent.coll_dist)[0]
            if len(ca_entered) == 0:
                agent.ca_entry_sample = math.inf
                agent.ca_entry_time = math.inf
            else:
                agent.ca_entry_sample = ca_entered[0]
                agent.ca_entry_time = self.time_stamps[ca_entered[0]]
        # who passed first?
        i_first_passer = np.argmin((self.agents[0].ca_entry_time, 
                                    self.agents[1].ca_entry_time))
        if math.isinf(self.agents[i_first_passer].ca_entry_time):
            # neither of the agents entered the conflict area
            self.first_passer = None
        else:
            # at least one agent entered the conflict area
            self.first_passer = self.agents[i_first_passer]
            
    
    def do_kinem_states_plot(self, axs, veh_stop_dec=False, axis_labels=True,
                             alpha=1, i_plot_agents=range(N_AGENTS),
                             agent_alpha=np.ones(N_AGENTS),
                             plot_const_guides=True, plot_fill=True, 
                             fill_r = None, hlines=True):
        """
        Plot kinematic simulation states.

        Parameters
        ----------
        axs : array of Matplotlib Axes.
            The axes to plot into: (0) acceleration, (1) speed, (2) distance
            to conflict point, (3) apparent time to conflict space entry,
            (4) collision distance margin. Provide a shorter array or set any
            element to None to omit that plot.
        veh_stop_dec : bool, optional
            Include require yielding deceleration for cars in the acceleration
            plot. The default is False.

        Returns
        -------
        None.

        """     
        for idx_agent, i_agent in enumerate(i_plot_agents):
            agent = self.agents[i_agent]
            alpha = agent_alpha[idx_agent]
                
            if hasattr(agent, 'plot_dashes'):
                dashes = agent.plot_dashes
            else:
                dashes = (1, 0)
                
            if hasattr(agent, 'plot_lw'):
                lw = agent.plot_lw
            else:
                lw = 1
                
            # acceleration
            if not(axs[0] == None):
                if veh_stop_dec and agent.ctrl_type == CtrlType.ACCELERATION:
                    axs[0].plot(self.time_stamps, agent.get_stop_accs(), 
                                '--', c=agent.plot_color, alpha = alpha/2, 
                                lw=lw)
                axs[0].plot(self.time_stamps, agent.trajectory.long_acc, 
                         '-', c=agent.plot_color, alpha=alpha, lw=lw, 
                         dashes=dashes)
                axs[0].set_xlim(self.time_stamps[0], self.actual_end_time)
                if axis_labels:
                    axs[0].set_ylabel('a (m/s^2)') 
                
            # speed
            if len(axs) > 1 and (not(axs[1] == None)):
                axs[1].plot(self.time_stamps, 
                            agent.other_agent.perception.states.x_perceived[1, :], 
                            '-', c=agent.plot_color, lw=lw/2, alpha=alpha/3, 
                            dashes=dashes)
                axs[1].plot(self.time_stamps, agent.trajectory.long_speed, 
                         '-', c=agent.plot_color, alpha=alpha, lw=lw, 
                         dashes=dashes)
                axs[1].set_ylim(-1, 15)
                if axis_labels:
                    axs[1].set_ylabel('v (m/s)') 
            
            # distance to conflict point
            if len(axs) > 2 and (not(axs[2] == None)):
                # - get CS entry/exit times
                in_CS_idxs = np.nonzero(np.abs(agent.signed_CP_dists) 
                                        <= agent.coll_dist)[0]
                if len(in_CS_idxs) > 0:
                    t_en = self.time_stamps[in_CS_idxs[0]]
                    t_ex = self.time_stamps[in_CS_idxs[-1]]
                else:
                    t_en = math.nan
                    t_ex = math.nan
                if plot_fill:
                    # - illustrate when agent is in CS
                    if fill_r == None:
                        agent_fill_r = agent.coll_dist
                    else:
                        agent_fill_r = fill_r
                    axs[2].fill(np.array((t_en, t_ex, t_ex, t_en)), 
                             np.array((-1, -1, 1, 1)) * agent_fill_r, 
                             color = agent.plot_color, alpha = alpha/3,
                             edgecolor = None)
                if plot_const_guides and hlines:
                    # - horizontal lines
                    axs[2].axhline(agent.coll_dist, color=agent.plot_color, 
                                   linestyle='--', lw=lw/2, alpha=0.9)
                    axs[2].axhline(-agent.coll_dist, color=agent.plot_color, 
                                   linestyle='--', lw=lw/2, alpha=0.9)
                    if i_agent == i_plot_agents[0]:
                        axs[2].axhline(0, color='k', linestyle=':')
                # - plot the distance itself
                axs[2].plot(self.time_stamps, 
                            agent.other_agent.perception.states.x_perceived[0, :], 
                            '-', c=agent.plot_color, lw=lw, alpha=alpha/3, 
                            dashes=dashes)
                axs[2].plot(self.time_stamps, agent.signed_CP_dists, 
                         '-', c=agent.plot_color, alpha=alpha, lw=lw, 
                         dashes=dashes)
                axs[2].set_ylim(-5, 5)
                if axis_labels:
                    axs[2].set_ylabel('$d_{CP}$ (m)') 
            
            # apparent time to conflict space entry
            if len(axs) > 3 and (not(axs[3] == None)):
                if i_agent == i_plot_agents[0]:
                    axs[3].axhline(0, color='k', linestyle=':')
                with np.errstate(divide='ignore'):
                    axs[3].plot(self.time_stamps, 
                                (agent.signed_CP_dists - agent.coll_dist) 
                                / agent.trajectory.long_speed, '-', 
                                c=agent.plot_color, alpha=alpha, dashes=dashes)
                axs[3].set_ylim(-1, 8)
                if axis_labels:
                    axs[3].set_ylabel('$TTCS_{app}$ (s)')
            
        # distance margin to agent collision
        if len(axs) > 4 and (not(axs[4] == None)):
            if plot_const_guides:
                axs[4].axhline(0, color='k', linestyle=':')
            coll_margins, coll_idxs = \
                get_sc_agent_collision_margins(self.agents[0].signed_CP_dists, 
                                               self.agents[1].signed_CP_dists,
                                               self.agents[0].coll_dist, 
                                               self.agents[1].coll_dist)
            axs[4].plot(self.time_stamps, coll_margins, 'k-', alpha=alpha, 
                        lw=lw, dashes=dashes)
            axs[4].plot(self.time_stamps[coll_idxs], 
                     coll_margins[coll_idxs], 'r-', alpha=alpha, 
                     lw=lw, dashes=dashes)
            axs[4].set_ylim(-1, 10)
            if axis_labels:
                axs[4].set_ylabel('$d_{coll}$ (m)')
                axs[4].set_xlabel('t (s)')      
    
            
    def do_plots(self, trajs = False, action_vals = False, action_probs = False, 
                 action_val_ests = False, surplus_action_vals = False, 
                 beh_activs = False, beh_accs = False, beh_probs = False, 
                 sensory_prob_dens = False, kinem_states = False, 
                 veh_stop_dec = False, times_to_ca = False, looming = False,
                 action_evidences = False):

        n_plot_actions = max(self.agents[0].n_actions, self.agents[1].n_actions)
        
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
                for i_action, deltav in enumerate(agent.params.ctrl_deltas):
                    plt.subplot(n_plot_actions, N_AGENTS, \
                                i_action * N_AGENTS +  i_agent + 1)
                    plt.ylim(-1.1, 1.1)
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
                for i_action, deltav in enumerate(agent.params.ctrl_deltas):
                    plt.subplot(n_plot_actions, N_AGENTS, \
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
            figname = 'Action value estimates'
            plt.figure(figname)
            plt.clf()
            fig, axs = plt.subplots(nrows = n_plot_actions, ncols = N_AGENTS,
                                    sharex = 'col', sharey = 'col',
                                    num = figname, figsize = (10.0, 10.0))
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(agent.params.ctrl_deltas):
                    ax = axs[i_action, i_agent]
                    ax.plot(self.time_stamps, 
                             agent.states.mom_action_vals[i_action, :])
                    ax.plot(self.time_stamps, 
                             agent.states.est_action_vals[i_action, :])
                    #plt.ylim(-2, 2)
                    if i_action == 0:
                        ax.set_title('Agent %s' % agent.name)
                        if i_agent == 1:
                            plt.legend(('$\\tilde{V}_a$', '$\\hat{V}_a$'))
                    if i_agent == 0:
                        ax.set_ylabel('$V(\\Delta v=%.1f)$' % deltav)
                        
        def plot_thresh_and_trigs(assumption, thresh, i_action):
            if agent.assumptions[assumption]:
                ax.plot([self.time_stamps[0], self.time_stamps[-1]], 
                    [thresh, thresh], '--', color = 'gray')
                action_time_steps = self.time_stamps[
                    agent.states.action_triggered[i_action, :]]
                n_actions = len(action_time_steps)
                if n_actions > 0:
                    ax.plot(action_time_steps, np.full(n_actions, thresh), 
                            'k+', ms=10, markeredgewidth=2)

        if surplus_action_vals:
            # - surplus action values
            figname = 'Surplus action value estimates'
            plt.figure(figname)
            plt.clf()
            fig, axs = plt.subplots(nrows = n_plot_actions, ncols = N_AGENTS,
                                    sharex = 'col', sharey = 'col',
                                    num = figname, figsize = (6, 5))
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(agent.params.ctrl_deltas):
                    ax = axs[i_action, i_agent]
                    ax.plot(self.time_stamps, 
                             agent.states.est_action_surplus_vals[i_action, :])
                    plot_thresh_and_trigs(OptionalAssumption.oEA, 
                                          agent.params.DeltaV_th, i_action)
                    if i_action == 0:
                        ax.set_title('Agent %s' % agent.name)
                    if i_agent == 0:
                        ax.set_ylabel('$\\Delta V(%.1f)$' % deltav)
              
        if action_evidences:
            figname = 'Action evidence'
            plt.figure(figname)
            plt.clf()
            fig, axs = plt.subplots(nrows = n_plot_actions, ncols = N_AGENTS,
                                    sharex = 'col', sharey = 'col',
                                    num = figname, figsize = (6, 5))
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(agent.params.ctrl_deltas):
                    ax = axs[i_action, i_agent]
                    ax.plot(self.time_stamps, 
                             agent.states.action_evidence[i_action, :])
                    plot_thresh_and_trigs(OptionalAssumption.oDA, 
                                          agent.params.xi_th, i_action)
                    ax.set_ylim(-.1 * agent.params.xi_th, 1.1 * agent.params.xi_th)
                    if i_action == 0:
                        ax.set_title('Agent %s' % agent.name)
                    if i_agent == 0:
                        ax.set_ylabel('$\\xi(%.1f)$' % deltav)

        if beh_activs:
            # - behavior activations
            for i_agent, agent in enumerate(self.agents):
                figname = ('Behaviour activations - Agent %s (observing %s)' %
                           (agent.name, agent.other_agent.name))
                plt.figure(figname)
                plt.clf()
                fig, axs = plt.subplots(nrows = agent.n_actions+1, 
                                        ncols = n_plot_behaviors,
                                        sharex = 'col', sharey = 'col',
                                        num = figname, figsize = (7, 7),
                                        squeeze = False)
                for i_beh in range(n_plot_behaviors):
                    # action observation contribution
                    ax = axs[0, i_beh]
                    ax.plot(self.time_stamps, agent.params.beta_O 
                             * agent.states.beh_activ_O[i_beh, :])
                    ax.set_title(BEHAVIORS[i_beh])
                    if i_beh == n_plot_behaviors-1:
                        ax.legend(('$\\beta_O A_{O,b}$',))
                    # value contribution and total activation - both per action
                    for i_action, deltav in enumerate(agent.params.ctrl_deltas):
                        ax = axs[i_action+1, i_beh]
                        ax.plot(self.time_stamps, 
                                 agent.params.beta_V 
                                 * agent.states.beh_activ_V_given_actions[
                                         i_beh, i_action,  :])
                        ax.plot(self.time_stamps, 
                                 agent.states.beh_activations_given_actions[
                                         i_beh, i_action, :])
                        #plt.ylim(-2, 5)
                        if i_beh == n_plot_behaviors-1 and i_action == 0:
                            ax.legend(('$\\beta_V A_{V,b|a}$', '$A_{b|a}$'))
                        if i_beh == 0:
                            ax.set_ylabel('$\\Delta v=%.1f$' % deltav)

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
            figname = 'Behaviour probabilities'
            plt.figure(figname)
            plt.clf()
            fig, axs = plt.subplots(nrows = n_plot_actions, ncols = N_AGENTS,
                                    sharex = 'col', sharey = 'col',
                                    num = figname, figsize = (8, 7))
            for i_agent, agent in enumerate(self.agents):
                for i_action, deltav in enumerate(agent.params.ctrl_deltas):
                    ax = axs[i_action, i_agent]
                    for i_beh in range(n_plot_behaviors):
                        # plt.subplot(N_BEHAVIORS, N_AGENTS, i_beh * N_AGENTS +  i_agent + 1)
                        ax.plot(self.time_stamps, 
                                 agent.states.beh_probs_given_actions[
                                         i_beh, i_action, :])
                        ax.set_ylim(-.1, 1.1)
                        if i_action == 0:
                            ax.set_title('Agent %s (observing %s)' % 
                                      (agent.name, agent.other_agent.name))
                    if i_agent == 0:
                        ax.set_ylabel('$P_{b|\\Delta v=%.1f}$ (-)' % deltav)
                    elif i_action == 0:
                        ax.legend(plot_behaviors)
                    
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
            fig = plt.figure('Kinematic and action states', figsize = (5, 6))
            plt.clf()
            N_PLOTROWS = 5
            axs = fig.subplots(N_PLOTROWS, 1, sharex=True)
            self.do_kinem_states_plot(axs, veh_stop_dec)
            fig.tight_layout()
            

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
                          
        if looming:
            plt.figure('Visual looming')
            plt.clf()
            legend_strs = []
            for i_agent, agent in enumerate(self.agents):
                plt.plot(self.time_stamps, agent.states.thetaDot, 
                         color=agent.plot_color)
                legend_strs.append(f'Seen by {agent.name}')
            plt.legend(legend_strs)
            plt.xlabel('Time (s)')
            plt.ylabel(r'$\dot{\theta}$ (rad/s)')
            
            
        plt.show()
        
        
def run_test_scenarios(optional_assumptions = None, params = None, 
                       dist0s = (30, 40, 50),
                       plot_beh_probs = False, plot_beh_activs = False, 
                       plot_beh_accs = False, plot_looming = False, 
                       plot_surpl_act_vals = False, plot_act_evidences = False,
                       ped_snaps = None, veh_snaps = None, 
                       print_dist = True):
    if optional_assumptions is None:
        optional_assumptions = get_assumptions_dict(default_value=False)
    if params is None:
        (params, params_k) = get_default_params(
            oVA = optional_assumptions[OptionalAssumption.oVA])
    NAMES = ('P', 'V')
    WIDTHS = (2, 2) 
    LENGTHS = (2, 2) 
    CTRL_TYPES = (CtrlType.SPEED, CtrlType.ACCELERATION) 
    GOALS = np.array([[0, 5], [-50, 0]])
    SPEEDS = np.array((0, 10))
    PED_Y0 = -5
    sims = []
    for dist0 in dist0s:
        INITIAL_POSITIONS = np.array([[0, PED_Y0], [dist0, 0]])
        sc_simulation = SCSimulation(
                CTRL_TYPES, WIDTHS, LENGTHS, GOALS, INITIAL_POSITIONS, 
                initial_speeds = SPEEDS, 
                end_time = 10, optional_assumptions = optional_assumptions,
                agent_names = NAMES, params = params,
                snapshot_times = (ped_snaps, veh_snaps))
        sc_simulation.run()
        sims.append(sc_simulation)
        if print_dist:
            print('\nInitial car distance %d m:' % dist0)
        sc_simulation.do_plots(kinem_states = True, 
                               beh_probs = plot_beh_probs,
                               beh_activs = plot_beh_activs, 
                               beh_accs = plot_beh_accs, 
                               surplus_action_vals = plot_surpl_act_vals,
                               looming = plot_looming,
                               action_evidences = plot_act_evidences)
    return sims



""" Some "unit tests" - not with pass/fail criteria but rather to provide a
    means of seeing if e.g. code changes that shouldn't make a difference don't,
    etc. The specific tests here are a subset of those I ran the last time I did
    this in a diary notebook (2022-01-11b), plus some later additions.
"""
if __name__ == "__main__":
    
    
    def print_test_heading(heading):
        print(f'\n\n***** {heading} *****')
    
    
    # use affordance-based value estimation for all tests
    AFF_VAL_FCN = True
    
    
    # Startup behaviour of the base model
    print_test_heading('Base model, just startup')
    NAMES = ('P', 'V')
    WIDTHS = (2, 2) 
    LENGTHS = (2, 2) 
    CTRL_TYPES = (CtrlType.SPEED, CtrlType.ACCELERATION) 
    GOALS = np.array([[0, 5], [-50, 0]])
    INITIAL_POSITIONS = np.array([[0,-5], [400, 0]])
    SPEEDS = np.array((0, 0))
    optional_assumptions = get_assumptions_dict(
        default_value = False, oVA = AFF_VAL_FCN)
    sc_simulation = SCSimulation(
            CTRL_TYPES, WIDTHS, LENGTHS, GOALS, INITIAL_POSITIONS, 
            initial_speeds = SPEEDS, end_time = 15, 
            optional_assumptions = optional_assumptions, agent_names = NAMES)
    sc_simulation.run()
    sc_simulation.do_plots(kinem_states = True)

            
    # Base model with the "baseline kinematics" 
    print_test_heading('Base model, baseline kinematics')
    run_test_scenarios(optional_assumptions = optional_assumptions)
    
    
    # a sequence of turning on the behaviour observation-related assumptions
    print_test_heading('Enabling behaviour observation assumptions')
    def run_beh_est_test(test_name, oBEo=False, oBEv=False, oAI=False):
        print(f'\n{test_name}:')
        global optional_assumptions
        optional_assumptions = get_assumptions_dict(
                default_value = False, oVA = AFF_VAL_FCN, 
                oBEo = oBEo, oBEv = oBEv, oAI = oAI)
        run_test_scenarios(optional_assumptions = optional_assumptions,
                           dist0s = (40,), plot_beh_probs = True,
                           print_dist = False)
    run_beh_est_test('Turning on oBEo', oBEo=True)
    run_beh_est_test('Turning on oBEv', oBEo=True, oBEv=True)
    run_beh_est_test('Turning on oAI', oBEo=True, oBEv=True, oAI=True)
    
    
    # testing oEA
    print_test_heading('Enabling value accumulation')
    optional_assumptions = get_assumptions_dict(
        default_value = False, oVA = AFF_VAL_FCN, oEA = True)
    run_test_scenarios(optional_assumptions = optional_assumptions,
                       dist0s = (30, 50), plot_surpl_act_vals=True)
    
    
    # testing oDA
    print_test_heading('Enabling decision evidence accumulation')
    optional_assumptions = get_assumptions_dict(
        default_value = False, oVA = AFF_VAL_FCN, oDA = True)
    run_test_scenarios(optional_assumptions = optional_assumptions,
                       dist0s = (30, 50), plot_surpl_act_vals=True, 
                       plot_act_evidences=True)