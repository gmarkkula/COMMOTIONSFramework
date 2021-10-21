import warnings
import collections
import math
from enum import Enum
import numpy as np

import commotions

EPSILON = np.finfo(float).eps

SMALL_NEG_SPEED = -0.01 # m/s - used as threshold for functions that don't accept negative speeds - but to allow minor speed imprecisions at stopping
ACC_CTRL_REGAIN_SPD_TIME = 10 # s - assumed time needed for acceleration-controlling agent to regain free speed (regardless of starting speed)

class CtrlType(Enum):
    SPEED = 0
    ACCELERATION = 1
    
class AccessOrder(Enum):
    EGOFIRST = 0
    EGOSECOND = 1
i_EGOFIRST = AccessOrder.EGOFIRST.value
i_EGOSECOND = AccessOrder.EGOSECOND.value
N_ACCESS_ORDERS = len(AccessOrder)

class AnticipationPhase(Enum):
    ACTION = 0
    ACH_ACCESS = 1
    WAIT = 2
    REGAIN_SPD = 3
    CONTINUE = 4
i_ACTION = AnticipationPhase.ACTION.value
i_ACH_ACCESS = AnticipationPhase.ACH_ACCESS.value
i_WAIT = AnticipationPhase.WAIT.value
i_REGAIN_SPD = AnticipationPhase.REGAIN_SPD.value
i_CONTINUE = AnticipationPhase.CONTINUE.value
N_ANTICIPATION_PHASES = len(AnticipationPhase)
ANTICIPATION_TIME_STEP = 0.2
ANTICIPATION_LIMIT = 6 # in units of T_delta (2^-6 = 0.016)
    
AccessOrderImplication = collections.namedtuple('AccessOrderImplication',
                                                ['acc', 'T_acc', 'T_dw'])

#ANTICIPATION_PHASES = ('action', 'ach_access', 'wait', 'regain_spd', 'continue')
#N_ANTICIPATION_PHASES = len(ANTICIPATION_PHASES)

AccessOrderValue = collections.namedtuple('AccessOrderValue',
                                          ['value', 'details'])

AccessOrderValueDetails = collections.namedtuple('AccessOrderValueDetails',
                                                 ['time_stamps', 
                                                  'speeds',
                                                  'accelerations',
                                                  'idx_phase_starts',
                                                  'kinematics_values',
                                                  'looming_values',
                                                  'discount_factors',
                                                  'phase_kinem_values',
                                                  'phase_looming_values'])

SCAgentImage = collections.namedtuple('SCAgentImage', 
                                      ['ctrl_type', 'params', 'v_free', 
                                       'V_free', 'eff_width'])



def get_agent_free_speed(k):
    return k._g / (2 * k._dv)

def set_val_gains_for_free_speed(k, v_free):
    """ Set properties _g and _dv of k to yield the free speed v_free, and 
        a normalised value at free speed equal to 1. 
        (See handwritten notes from 2021-01-16.)

    """
    k._g = 2 / v_free
    k._dv = 1 / v_free ** 2


def get_acc_to_be_at_dist_at_time(speed, target_dist, target_time, consider_stop):
    """ Return acceleration required to travel a further distance target_dist
        in time target_time if starting at speed speed, as well as time during
        which the acceleration will be applied. The time will be equal to
        target_time unless full stop is needed - if so the returned time is
        the stopping time. Handle infinite target_time by returning machine 
        epsilon with correct sign.
    """
        
    #assert target_time > 0    
    if target_time <= 0:
        return math.nan, math.nan
    
    # time horizon finite or not?
    if target_time == math.inf:
        # infinite time horizon - some special cases with near-zero acceleration
        if target_dist == 0:
            # at target, so just reverse any speed
            return -np.sign(speed) * EPSILON, math.inf
        elif speed == 0:
            # at zero speed, so remove remaining distance
            return np.sign(target_dist) * EPSILON, math.inf
        else:  
            # neither distance nor speed are zero, so sign of acceleration
            # depends on combination of both - and we need to do the stopping 
            # check below, so can't return straight away
            target_acc = -np.sign(speed * target_dist) * EPSILON
    else:
        # finite time horizon --> general answer for most situations
        target_acc = 2 * (target_dist - speed * target_time) / (target_time ** 2)
    
    # might this be a stopping situation?
    if consider_stop:
        # check whether this is a stopping situation - if so we should apply 
        # stopping deceleration instead (to avoid passing beyond target distance 
        # and reversing back toward it again)
        if target_dist > 0 and speed > 0 and target_acc < 0:
            stop_acc = -speed ** 2 / (2 * target_dist)
            stop_time = speed / (-stop_acc)
            if stop_time < target_time:
                # stopping time is shorter than target time, so we will need to
                # stop at target and wait
                return stop_acc, stop_time
    
    # not a stopping situation, so return what was calculated above
    return target_acc, target_time
        


def get_time_to_dist_with_acc(state, target_dist):
    # already past the target distance?
    if target_dist <= 0:
        return -math.inf
    # standing still?
    if state.long_speed == 0 and state.long_acc <= 0:
        return math.inf
    # acceleration status?
    if state.long_acc == 0:
        # no acceleration
        return target_dist / state.long_speed
    elif state.long_acc < 0:
        # decelerating
        # will stop before target distance?
        stop_dist = state.long_speed ** 2 / (-2 * state.long_acc)
        if stop_dist <= target_dist:
            return math.inf
        # no, so get time of passing target distance as first root of the
        # quadratic equation
    # accelerating (or decelerating) past, so get second (or first - controlled
    # by the sign of acceleration in denominator) root of the quadratic equation
    return (-state.long_speed + 
            math.sqrt(state.long_speed ** 2 + 2 * state.long_acc * target_dist)
            ) / state.long_acc
    
    
    
def get_entry_exit_times(state, coll_dist, consider_acc = True):
    """ Return tuple of times left to entry and exit into conflict zone defined by 
        +/-coll_dist around conflict point.
        
        If consider_acc = True: Consider agent at 
        state.signed_CP_dist to conflict point and travelling toward it at 
        speed state.long_speed and acceleration state.long_acc. Return zero
        if entry/exit has happened already. Return inf if speed is zero, or 
        if the acceleration will have the agent stop before entry/exit.
        
        If consider_acc = False: As above, but disregard
        acceleration, and give entry/exit apparently in the past returned
        as negative times. Return inf if speed is zero. 
    """
    # is the agent moving?
    if state.long_speed == 0:
        if abs(state.signed_CP_dist) < coll_dist:
            return (-math.inf, math.inf)
        else:
            return (math.inf, math.inf)
    else:
        # get the time to reach the entry/exit edge of the conflict space
        if consider_acc:
            entry_time = get_time_to_dist_with_acc(
                state, state.signed_CP_dist - coll_dist)
            exit_time = get_time_to_dist_with_acc(
                state, state.signed_CP_dist + coll_dist)
        else:
            # taking into account the speed when defining which edge is entry vs exit
            sp_sign = np.sign(state.long_speed)
            entry_time = (state.signed_CP_dist 
                          - sp_sign * coll_dist) / state.long_speed
            exit_time = (state.signed_CP_dist 
                         + sp_sign * coll_dist) / state.long_speed
        return (entry_time, exit_time)
    
    
def add_entry_exit_times_to_state(state, coll_dist):
    state.cs_entry_time = {}
    state.cs_exit_time = {}
    (state.cs_entry_time[CtrlType.SPEED], 
     state.cs_exit_time[CtrlType.SPEED]
     ) = get_entry_exit_times(state, coll_dist, consider_acc = False)
    (state.cs_entry_time[CtrlType.ACCELERATION], 
     state.cs_exit_time[CtrlType.ACCELERATION]
     ) = get_entry_exit_times(state, coll_dist, consider_acc = True)
    return state
      

def get_delay_discount(T, T_delta):
    return 2**(-T / T_delta)


def get_const_value_rate(v, a, k):
    """ Return value accrual rate for an agent with value function gains k, if
        travelling at speed v toward goal and acceleration a. Inputs v and a
        can be numpy arrays.
    """
    return (k._g * v - k._dv * v ** 2 - k._da * a ** 2)


def get_value_of_const_jerk_interval(v0, a0, j, T, k):
    """ Return value for an agent with value function gains k of applying
        constant jerk j during a time interval T, from initial speed and 
        acceleration v0 and a0.
        
        See hand written notes from 2020-12-21.
    """
#    av_a = (a0 + (a0 + j * T)) / 2
#    av_v = (v0 + (v0 + a0 * T + j * T**2 / 2)) / 2
#    av_value_rate = get_const_value_rate(av_v, av_a, k) 
    av_value_rate = ( get_const_value_rate(v0, a0, k) 
        + (1/2) * (k._g * a0 - 2 * k._dv * v0 * a0 - 2 * k._da * a0 * j) * T 
        + (1/3) * (k._g * j / 2 - k._dv * (a0**2 + v0 * j) - k._da * j**2) * T**2 
        - (k._dv * j / 4) * (a0 + j * T / 5) * T**3 )
    return T * av_value_rate
    

def set_phase_acceleration(accelerations, idx_phase_starts, i_phase, phase_acc):
    accelerations[idx_phase_starts[i_phase]:idx_phase_starts[i_phase+1]] = phase_acc


def get_access_order_values(ego_image, oth_image, 
                            v0, action_acc0, action_jerk, 
                            access_ord_impls, return_details = False):
    
    access_ord_values = {}
    for access_ord in AccessOrder:
        
        if np.isnan(access_ord_impls[access_ord].acc):
            # access order not valid from this state
            access_ord_values[access_ord] = AccessOrderValue(
            value = -math.inf, details = None)
            continue
        
        # get the duration of the anticipation phases 
        # (except the final "continue" phase)
        phase_durations = np.full(N_ANTICIPATION_PHASES-1, math.nan)
        phase_durations[i_ACTION] = ego_image.params.DeltaT
        phase_durations[i_ACH_ACCESS] = access_ord_impls[access_ord].T_acc
        phase_durations[i_WAIT] = access_ord_impls[access_ord].T_dw
        if ego_image.ctrl_type is CtrlType.SPEED:
            phase_durations[i_REGAIN_SPD] = ego_image.params.DeltaT
        else:
            phase_durations[i_REGAIN_SPD] = ACC_CTRL_REGAIN_SPD_TIME
        cont_phase_start_time = np.sum(phase_durations)
            
        # # are any of the phase durations of infinite length?
        # inf_duration_phases = np.nonzero(np.isinf(phase_durations))[0]
        # if len(inf_duration_phases) > 0:
        #     i_final_integr_phase = inf_duration_phases[0] - 1
        # else:
        #     i_final_integr_phase = i_REGAIN_SPD
            
        # get various vectors concerning time stamps and phase starts
        phase_start_times = np.concatenate((np.array((0.,)), np.cumsum(phase_durations)))
        distant_phases = np.nonzero(
            phase_start_times > ANTICIPATION_LIMIT * ego_image.params.T_delta)[0]
        if len(distant_phases) > 0:
            i_final_integr_phase = distant_phases[0] - 2
        else:
            i_final_integr_phase = i_REGAIN_SPD
        idx_phase_starts = np.floor(
            phase_start_times[:i_final_integr_phase+2] / ANTICIPATION_TIME_STEP).astype(int)
        integr_end_time = phase_start_times[i_final_integr_phase + 1]
        time_stamps = np.arange(0, integr_end_time, ANTICIPATION_TIME_STEP)
        n_time_steps = len(time_stamps)
        
        # get vector of accelerations up until just before the "continue" phase
        accelerations = np.zeros(n_time_steps)
        # - action phase
        if action_jerk == 0:
            action_acc = action_acc0
        else:
            action_acc = action_acc0 + action_jerk * time_stamps[
                :idx_phase_starts[i_ACTION+1]]
        set_phase_acceleration(accelerations, idx_phase_starts, 
                               i_ACTION, action_acc)
        # - achieving access order phase
        if i_final_integr_phase >= i_ACH_ACCESS:
            set_phase_acceleration(accelerations, idx_phase_starts, 
                                   i_ACH_ACCESS, access_ord_impls[access_ord].acc)
            # - waiting phase
            if i_final_integr_phase >= i_WAIT:
                if phase_durations[i_WAIT] > 0:
                    set_phase_acceleration(accelerations, idx_phase_starts, 
                                           i_WAIT, 0)
                    vprime = 0
                else:
                    vprime = v0 + np.sum(accelerations[:idx_phase_starts[i_WAIT]] 
                                         * ANTICIPATION_TIME_STEP)
                    #assert(abs(vprime) < 1)
                # - regain free speed phase
                if i_final_integr_phase >= i_REGAIN_SPD:
                    regain_spd_acc = ((ego_image.v_free - vprime) 
                                      / phase_durations[i_REGAIN_SPD])
                    set_phase_acceleration(accelerations, idx_phase_starts, 
                                           i_REGAIN_SPD, regain_spd_acc)
        
        # integrate to get corresponding vector of speeds
        speeds = v0 + np.cumsum(accelerations * ANTICIPATION_TIME_STEP)
        if i_final_integr_phase >= i_REGAIN_SPD and phase_durations[i_WAIT] > 0:
            # make sure to adjust so that the speed while waiting is exactly zero
            vprime_actual = speeds[idx_phase_starts[i_WAIT]]
            speeds[idx_phase_starts[i_WAIT]:] -= vprime_actual
        
        # get the kinematics value contributions
        kinematics_values = ((ego_image.params.k._g * speeds
                              - ego_image.params.k._dv * speeds ** 2
                              - ego_image.params.k._da * accelerations ** 2)
                             * ANTICIPATION_TIME_STEP)
        post_value_raw = ego_image.V_free
        
        # get the looming value contributions
        looming_values = np.zeros(n_time_steps)
        
        # apply time discounting and get the total value
        discount_factors = get_delay_discount(time_stamps, 
                                              ego_image.params.T_delta)
        discounted_values = (kinematics_values 
                             + looming_values) * discount_factors
        post_value_discounted = (get_delay_discount(cont_phase_start_time, 
                                                   ego_image.params.T_delta)
                                 * post_value_raw)
        total_value = np.sum(discounted_values) + post_value_discounted
        
        # should we be returning detailed information?
        if return_details:
            # get the contribution 
            phase_kinem_values = np.zeros(N_ANTICIPATION_PHASES)
            phase_looming_values = np.zeros(N_ANTICIPATION_PHASES)
            for i_phase in range(i_final_integr_phase+1):
                phase_idxs = np.arange(idx_phase_starts[i_phase],
                                      idx_phase_starts[i_phase+1])
                phase_kinem_values[i_phase] = np.sum(kinematics_values[phase_idxs] 
                                                     * discount_factors[phase_idxs])
                phase_looming_values[i_phase] = np.sum(looming_values[phase_idxs] 
                                                       * discount_factors[phase_idxs])
            phase_kinem_values[i_CONTINUE] = post_value_discounted
            details = AccessOrderValueDetails(
                time_stamps = time_stamps,
                speeds = speeds,
                accelerations = accelerations,
                idx_phase_starts = idx_phase_starts,
                kinematics_values = kinematics_values,
                looming_values = looming_values,
                discount_factors = discount_factors,
                phase_kinem_values = phase_kinem_values,
                phase_looming_values = phase_looming_values)
        else:
            details = None
            
        # create the named tuple object for this access order
        access_ord_values[access_ord] = AccessOrderValue(
            value = total_value, details = details)
        
    # return the results
    return access_ord_values


def get_access_order_implications(ego_image, ego_state, oth_state, coll_dist,
                                  consider_oth_acc = True, return_nans = True):
    """ Return a dict over AccessOrder with an AccessOrderImplication for each, 
        for the ego agent described by ego_image, with state ego_state (using 
        fields signed_CP_dist, long_speed), to pass respectively before or 
        after, respecting collision distance coll_dist, the other agent 
        described by oth_state (same fields as above, but also 
        cs_times), assumed to keep either a constant acceleration 
        (consider_oth_acc = True) or constant speed (consider_oth_acc = False) 
        from the current time. 
        
        More specifically, each AccessOrderImplication contains:
            
        * An acceleration acc and an associated duration of acceleration T_acc:
        
            - Accelerations for passing first: Either just normal acceleration 
              toward free speed if this is enough (with T_acc being the time 
              needed to reach the free speed), otherwise the acceleration
              needed to exit the conflict space just as the other agent enters it 
              (with T_acc the time until that exit).
            
            - Accelerations for passing second: Either just normal acceleration
              toward free speed if this is enough (with T_acc being the time 
              needed to reach the free speed), otherwise the acceleration
              needed to just enter the conflict space as the other agent exits it 
              (with T_acc the time until that entry), if possible, otherwise the
              acceleration needed to stop just at the entrance to the conflict 
              space (to wait there until the other agent passes; with T_acc the 
              time until full stop).
        
        * A waiting delay duration T_dw which is zero in all cases except the 
          full stop cases mentioned above; in these cases T_dw is the time 
          from full stop until the other agent exits the conflict space.
            
        If there is an ongoing collision, outputs for both access orders will 
        be undefined (math.nan if return_nans is True, otherwise using math.inf)
        
        If the other agent has already entered the conflict space, outputs for
        the AccessOrder.EGOFIRST will be undefined.
        
        If the ego agent has already exited the conflict space, outputs for 
        both access orders will be just acceleration to free speed.
    """

    # are both agents currently in the conflict space (i.e., a collision)?
    if (abs(ego_state.signed_CP_dist) < coll_dist and
        abs(oth_state.signed_CP_dist) < coll_dist):
        implications = {}
        if return_nans:
            for access_order in AccessOrder:
                implications[access_order] = AccessOrderImplication(
                        math.nan, math.nan, math.nan)
        else:
            implications[AccessOrder.EGOFIRST] = AccessOrderImplication(
                    acc = math.inf, T_acc = math.inf, T_dw = 0)
            implications[AccessOrder.EGOSECOND] = AccessOrderImplication(
                    acc = -math.inf, T_acc = math.inf, T_dw = 0)
        return implications
    

    # get ego agent's acceleration if just aiming for free speed
    dev_from_v_free = ego_state.long_speed - ego_image.v_free
    if ego_image.ctrl_type is CtrlType.SPEED:
        # assuming straight acc to free speed
        agent_time_to_v_free = ego_image.params.DeltaT
    else:
        # simplified approach for acc-controlling agents (the calculations
        # commented out below were also relying on lots of simplifying
        # assumptions, most notably constant acceleration while regaining speed,
        # which is not close to the truth)
        agent_time_to_v_free = ACC_CTRL_REGAIN_SPD_TIME
    ego_free_acc = -dev_from_v_free / agent_time_to_v_free
                
    # get time to reach free speed if applying free acceleration
    if dev_from_v_free == 0:
        T_acc_free = 0
    else:
        T_acc_free = agent_time_to_v_free
        
    # has the ego agent already exited the conflict space?
    if ego_state.signed_CP_dist <= -coll_dist:
        # yes - so no interaction left to do - return just the acceleration 
        # to free speed for both outcomes
        implications = {}
        for access_order in AccessOrder:
            implications[access_order] = AccessOrderImplication(
                    acc = ego_free_acc, T_acc = T_acc_free, T_dw = 0)
        return implications
    
    # which type of prediction for the other agent
    if consider_oth_acc:
        oth_pred = CtrlType.ACCELERATION
    else:
        oth_pred = CtrlType.SPEED 
        
    # prepare dicts
    accs = {}
    T_accs = {}
    T_dws = {}
    
    # never any waiting time involved in passing first
    T_dws[AccessOrder.EGOFIRST] = 0
    
    # get acceleration needed to pass first
    # - other agent already entered conflict space?
    if oth_state.signed_CP_dist <= coll_dist:
        # yes, so not possible for ego agent to pass first
        if return_nans:
            accs[AccessOrder.EGOFIRST] = math.nan
            T_accs[AccessOrder.EGOFIRST] = math.nan
        else:
            accs[AccessOrder.EGOFIRST] = math.inf
            T_accs[AccessOrder.EGOFIRST] = math.inf
    else:
        # no, so it is theoretically possible to pass in front of it
        # --> get acceleration that has the ego agent be at exit of the conflict 
        # space at the same time as the other agent enters it
        # (need to consider stop as possibility here, in case there is a long
        # time left until the other agent reaches the conflict space)
        ego_dist_to_exit = ego_state.signed_CP_dist + coll_dist
        accs[AccessOrder.EGOFIRST], T_accs[AccessOrder.EGOFIRST] = \
            get_acc_to_be_at_dist_at_time(
                ego_state.long_speed, ego_dist_to_exit + ego_image.params.D_s, 
                oth_state.cs_entry_time[oth_pred] - ego_image.params.T_s, 
                consider_stop=True)
        # if the acceleration to free speed is higher than the acceleration
        # needed to exit just as the other agent enters, there is no need to
        # assume that the agent will move slower than its free speed
        if ego_free_acc > accs[AccessOrder.EGOFIRST]:
            accs[AccessOrder.EGOFIRST] = ego_free_acc
            T_accs[AccessOrder.EGOFIRST] = T_acc_free
            
        
    # get acceleration needed to pass second
    # - has the other agent already exited the conflict space?
    if oth_state.signed_CP_dist <= -coll_dist:
        # yes, so just accelerate to free speed
        accs[AccessOrder.EGOSECOND] = ego_free_acc
        T_accs[AccessOrder.EGOSECOND] = T_acc_free
        T_dws[AccessOrder.EGOSECOND] = 0
    else:
        # no, the other agent hasn't already exited the conflict space
        # has the ego agent already entered conflict space?
        if ego_state.signed_CP_dist <= coll_dist:
            # yes, so passing in second is no longer an option
            if return_nans:
                accs[AccessOrder.EGOSECOND] = math.nan
                T_accs[AccessOrder.EGOSECOND] = math.nan
                T_dws[AccessOrder.EGOSECOND] = math.nan
            else:
                accs[AccessOrder.EGOSECOND] = -math.inf
                T_accs[AccessOrder.EGOSECOND] = math.inf
                T_dws[AccessOrder.EGOSECOND] = 0
        else:
            # not yet reached conflict space, so get acceleration that has the ego 
            # agent be at entrance to the conflict space at the same time as the 
            # other agent exits it (possibly by stopping completely, possibly
            # even before the other agent reaches the conflict space)
            ego_dist_to_entry = ego_state.signed_CP_dist - coll_dist
            oth_exit_time = oth_state.cs_exit_time[oth_pred]
            accs[AccessOrder.EGOSECOND], T_accs[AccessOrder.EGOSECOND] = \
                get_acc_to_be_at_dist_at_time(
                    ego_state.long_speed, ego_dist_to_entry - ego_image.params.D_s,
                    oth_exit_time + ego_image.params.T_s, consider_stop=True)
            if math.isinf(oth_exit_time):
                T_dws[AccessOrder.EGOSECOND] = math.inf
            else:
                T_dws[AccessOrder.EGOSECOND] = max(0, oth_exit_time
                                                - T_accs[AccessOrder.EGOSECOND])
            # if the acceleration to free speed is lower than the acceleration
            # needed to enter just as the other agent exits, there is no need to
            # assume that the agent will move faster than its free speed
            if ego_free_acc < accs[AccessOrder.EGOSECOND]:
                accs[AccessOrder.EGOSECOND] = ego_free_acc
                T_accs[AccessOrder.EGOSECOND] = T_acc_free
                T_dws[AccessOrder.EGOSECOND] = 0
        
            
    # return dict with the full results
    implications = {}
    for access_order in AccessOrder:
        implications[access_order] = AccessOrderImplication(
                acc = accs[access_order], T_acc = T_accs[access_order], 
                T_dw = T_dws[access_order])
    return implications


def get_access_order_accs(ego_image, ego_state, oth_state, coll_dist, 
                          consider_oth_acc = True):
    """ Return a tuple (acc_1st, acc_2nd) of expected accelerations 
        for the ego agent to pass before or after the other agent. A wrapper
        for get_access_order_implications() - see that function for 
        calling details.
    """
    implications = get_access_order_implications(ego_image, ego_state, 
                                                 oth_state, coll_dist,
                                                 consider_oth_acc, 
                                                 return_nans = True)
    return (implications[AccessOrder.EGOFIRST].acc, 
            implications[AccessOrder.EGOSECOND].acc)
 

def get_time_to_sc_agent_collision(state1, state2, consider_acc=False):
    """ Return the time left until the two agents with states state1 and state2
        are within the conflict space at the same time, or math.inf if this is 
        not projected to happen, or zero if it is already happening. This uses
        the properties cs_entry/exit_time of state1/2, specifically 
        cs_entry/exit_time[CtrlType.SPEED] if consider_acc is False, otherwise
        cs_entry/exit_time[CtrlType.ACCELERATION] is used.
    """
    # consider acceleration?
    if consider_acc:
        ctrl_type = CtrlType.ACCELERATION
    else:
        ctrl_type = CtrlType.SPEED
    # has at least one of the agents already left the CS?
    if state1.cs_exit_time[ctrl_type] < 0 or state2.cs_exit_time[ctrl_type] < 0:
        # yes - so no collision projected
        return math.inf
    # no, so both agents' exits are in future - are both agents' entries in past?
    if state1.cs_entry_time[ctrl_type] < 0 and state2.cs_entry_time[ctrl_type] < 0:
        # yes - so collision is ongoing
        return 0
    # no collision yet, so check who enters first
    if state1.cs_entry_time[ctrl_type] <= state2.cs_entry_time[ctrl_type]:
        # agent 1 entering CS first, is it staying long enough for agent 2 to enter
        if state1.cs_exit_time[ctrl_type] >= state2.cs_entry_time[ctrl_type]:
            # collision occurs when agent 2 enters CS - we know this is positive
            # (in future) since at least one of the entry times is positive and
            # agent 2's entry time is the larger one
            return state2.cs_entry_time[ctrl_type]
    else:
        # agent 2 entering first, so same logic as above but reversed
        if state2.cs_exit_time[ctrl_type] >= state1.cs_entry_time[ctrl_type]:
            return state1.cs_entry_time[ctrl_type]
    # no future overlap in CS occupancy detected - i.e., no collision projected
    return math.inf


def get_sc_agent_collision_margins(ag1_ds, ag2_ds, coll_dist):
    """
    Return the distance margins between agents at signed distances from conflict
    point ag1_ds and ag2_ds (can be numpy arrays), given the collision distance 
    coll_dist.
    """
    ag1_margins_to_CS = np.abs(ag1_ds) - coll_dist
    ag2_margins_to_CS = np.abs(ag2_ds) - coll_dist
    ag1_in_CS = ag1_margins_to_CS < 0
    ag2_in_CS = ag2_margins_to_CS < 0
    collision_margins = (np.maximum(ag1_margins_to_CS, 0)
                         + np.maximum(ag2_margins_to_CS, 0))
    collision_idxs = np.nonzero(np.logical_and(ag1_in_CS, ag2_in_CS))[0]
    return (collision_margins, collision_idxs)


def get_agent_optical_size(ego_state, oth_state, oth_image):
    """
    Return the optical size, in radians, of the agent described by
    oth_state and oth_image, as seen by an agent with state ego_state.

    """
    dist = math.sqrt(ego_state.signed_CP_dist ** 2 
                     + oth_state.signed_CP_dist ** 2)
    theta = 2 * math.atan(oth_image.eff_width / (2 * dist))
    return theta


def get_agent_optical_exp(ego_state, oth_state, oth_image):
    """
    Return the visual looming, in rad/s, of the agent described by
    oth_state and oth_image, as seen by an agent with state ego_state.

    """
    # see handwritten notes dated 2021-10-12
    dist = math.sqrt(ego_state.signed_CP_dist ** 2 
                     + oth_state.signed_CP_dist ** 2)
    thetaDot = -((oth_image.eff_width / dist)
                 * (ego_state.signed_CP_dist * (-ego_state.long_speed)
                    + oth_state.signed_CP_dist * (-oth_state.long_speed))
                 / (dist ** 2 + oth_image.eff_width ** 2 / 4))
    return thetaDot
        

# "unit tests"

if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.close('all')
    
    TEST_ACCESS_ORDER_ACCS = True
    TEST_TTCS_AND_LOOMING = True

# =============================================================================
#   Testing get_access_order_accs()
# =============================================================================
    
    if TEST_ACCESS_ORDER_ACCS:
    
        # defining the tests
        TESTS = ('Stationary ped., moving veh.', 'Moving ped., moving veh.',
                 'Ped. moving backwards, moving veh.', 
                 'Stationary veh., moving ped.', 'Moving veh., moving ped.', 
                 'Moving veh., stationary ped.', 'Stationary veh., stationary ped.',
                 'Stationary ped., accelerating veh.',
                 'Stationary ped., stopping veh.',
                 'Stationary ped., veh. decelerating past')
        EGO_CTRL_TYPES = (CtrlType.SPEED, CtrlType.SPEED, 
                          CtrlType.SPEED,
                          CtrlType.ACCELERATION, CtrlType.ACCELERATION, 
                          CtrlType.ACCELERATION, CtrlType.ACCELERATION,
                          CtrlType.SPEED,
                          CtrlType.SPEED,
                          CtrlType.SPEED)
        EGO_VS = (0, 1.5, 
                  -1.5,
                  0, 10, 
                  10, 0,
                  0,
                  0,
                  0)
        OTH_VS = (10, 10, 
                  10,
                  1.5, 1.5, 
                  0, 0,
                  10,
                  10,
                  10)
        OTH_AS = (0, 0,
                  0,
                  0, 0,
                  0, 0,
                  2,
                  -2,
                  -1)
        
        # constants depending on ego agent control type
        EGO_D_MAX = {}
        EGO_D_MAX[CtrlType.SPEED] = 40
        EGO_D_MAX[CtrlType.ACCELERATION] = 100
        EGO_V_FREE = {}
        EGO_V_FREE[CtrlType.SPEED] = 1.5
        EGO_V_FREE[CtrlType.ACCELERATION] = 10
        EGO_ACTION_DUR = 0.5
        EGO_PARAMS = {}
        for ctrl_type in CtrlType:
            EGO_PARAMS[ctrl_type] = commotions.Parameters() 
            EGO_PARAMS[ctrl_type].DeltaT = EGO_ACTION_DUR
            EGO_PARAMS[ctrl_type].T_s = 0.5
            EGO_PARAMS[ctrl_type].D_s = 0.5
        EGO_PARAMS[CtrlType.ACCELERATION].k = commotions.Parameters()
        EGO_PARAMS[CtrlType.ACCELERATION].k._g = 1 
        EGO_PARAMS[CtrlType.ACCELERATION].k._dv = 0.05
        EGO_PARAMS[CtrlType.ACCELERATION].k._da = 0.01
        OTH_D = {}
        OTH_D[CtrlType.SPEED] = 40
        OTH_D[CtrlType.ACCELERATION] = 6
        # other constants
        COLL_DIST = 2
        END_TIME = 10 # s
        TIME_STEP = 0.005 # s
        TIME_STAMPS = np.arange(0, END_TIME, TIME_STEP)
        
        # plot fcns
        def plot_conflict_window(ax, oth_state):
            ax.axhline(COLL_DIST, color='r', linestyle='--', lw=0.5)
            ax.axhline(-COLL_DIST, color='r', linestyle='--', lw=0.5)
            ax.axvline(oth_state.cs_entry_time[CtrlType.ACCELERATION], 
                       color='r', linestyle='--', lw=0.5)
            ax.axvline(oth_state.cs_exit_time[CtrlType.ACCELERATION], 
                       color='r', linestyle='--', lw=0.5)
        def plot_pred_ds(ax, ego_state, ego_acc):
            vs = ego_state.long_speed + ego_acc * TIME_STAMPS
            ds = ego_state.signed_CP_dist - np.cumsum(vs * TIME_STEP)
            ax.plot(TIME_STAMPS, ds, 'k-', lw=0.5)
        
        # loop through the test cases
        for i_test, test_name in enumerate(TESTS):
            
            # get test settings
            ego_ctrl_type = EGO_CTRL_TYPES[i_test]
            ego_image = SCAgentImage(ego_ctrl_type, 
                                     EGO_PARAMS[ego_ctrl_type], 
                                     v_free = EGO_V_FREE[ego_ctrl_type],
                                     V_free = None, eff_width = None)
            ego_state = commotions.KinematicState(long_speed = EGO_VS[i_test])
            oth_state = commotions.KinematicState(long_speed = OTH_VS[i_test])
            oth_state.long_acc = OTH_AS[i_test]
            oth_state.signed_CP_dist = OTH_D[ego_ctrl_type]
            ego_ds = np.linspace(-5, EGO_D_MAX[ego_ctrl_type], 20)
            
            # get other agent's movement
            oth_speeds = oth_state.long_speed + oth_state.long_acc * TIME_STAMPS
            oth_distances = (oth_state.signed_CP_dist 
                             - np.cumsum(oth_speeds * TIME_STEP))
            oth_state = add_entry_exit_times_to_state(oth_state, COLL_DIST)
                
            # get figure window for test results
            fig = plt.figure(test_name, figsize = (10, 5))
            axs = fig.subplots(1, 3)
            
            # plot other agent's path
            axs[0].plot(TIME_STAMPS, oth_distances, 'k-')
            plot_conflict_window(axs[0], oth_state)
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('$d_{oth}$ (m)')
            axs[0].set_title("Other agent")
            
            # prepare subplots for passing first/second plots
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('$d_{ego}$ (m)')
            axs[1].set_title('Ego agent passing first')
            plot_conflict_window(axs[1], oth_state)
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('$d_{ego}$ (m)')
            axs[2].set_title('Ego agent passing second')
            plot_conflict_window(axs[2], oth_state)
                
            # loop through initial distances
            for ego_d in ego_ds:
                ego_state.signed_CP_dist = ego_d
                (acc_1st, acc_2nd) = get_access_order_accs(
                        ego_image, ego_state, oth_state, COLL_DIST)
                plot_pred_ds(axs[1], ego_state, acc_1st)
                plot_pred_ds(axs[2], ego_state, acc_2nd)
                
            plt.tight_layout()
            plt.show()
    
    
# =============================================================================
#   Testing get_time_to_sc_agent_collision() and get_agent_optical_exp()
# =============================================================================
            
    if TEST_TTCS_AND_LOOMING:
        
        # defining the scenarios
        TESTS = ('Ped const - Colliding', 
                 'Ped const - Not colliding', 'Ped const - Stopping', 
                 'Ped stat - Passing', 'Ped stat - Stopping')
        N_TESTS = len(TESTS)
        COLL_DIST = 2
        AG_EFF_WIDTH = [1.8, 0.8] # car, pedestrian
        AG1_V0S = (10, 10, 10, 10, 10)
        AG1_D0S = (40, (4+COLL_DIST)*10+COLL_DIST+0.1, 40, 40, 40)
        STOP_ACC = -10**2/(2*(40-COLL_DIST))
        AG1_ACCS = (0, 0, STOP_ACC, 0, STOP_ACC)
        AG2_V0S = (1, 1, 1, 0, 0)
        AG2_D0S = (4, 4, 4, 2.5, 2.5)
        AG_COLORS = ('c', 'm')
        END_TIME = 10 # s
        TIME_STEP = 0.005 # s
        TIME_STAMPS = np.arange(0, END_TIME, TIME_STEP)
        N_TIME_STEPS = len(TIME_STAMPS)
        
        # helper fcn
        def get_ds_and_vs(d0, v0, acc):
            vs = v0 + TIME_STAMPS * acc
            vs = np.maximum(vs, 0)
            ds = d0 - np.cumsum(TIME_STEP * vs)
            return (ds, vs)
        
        # vectors for agent distances and speeds
        ag_ds = np.full((2, N_TIME_STEPS), math.nan)
        ag_vs = np.full((2, N_TIME_STEPS), math.nan)
        
        # agent image and state objects
        ag_image = []
        ag_state = [] 
        for i in range(2):
            ag_image.append(SCAgentImage(None, None, None, None, AG_EFF_WIDTH[i]))
            ag_state.append(commotions.KinematicState())

        # loop throgh scenarios
        for i_test, test_name in enumerate(TESTS):
            
            fig = plt.figure(test_name, figsize = (8, 9))
            axs = fig.subplots(nrows = 6, sharex = True)
            
            # get agent distances and speeds for this scenario
            (ag_ds[0,:], ag_vs[0,:]) = get_ds_and_vs(AG1_D0S[i_test], 
                                                     AG1_V0S[i_test], 
                                                     AG1_ACCS[i_test])
            (ag_ds[1,:], ag_vs[1,:]) = get_ds_and_vs(AG2_D0S[i_test], 
                                                     AG2_V0S[i_test], 0)
            
            # get and plot TTCs, both without and with consideration of accelerations
            for consider_acc in (True, False):
                ttcs = np.full(N_TIME_STEPS, math.nan)
                thetas = np.full((2, N_TIME_STEPS), math.nan)
                thetaDots = np.full((2, N_TIME_STEPS), math.nan)
                thetaDotsNum = np.full((2, N_TIME_STEPS), math.nan)
                for i, time in enumerate(TIME_STAMPS):
                    
                    # loop through agents and set their current states
                    for i_agent in range(2):
                        ag_state[i_agent].signed_CP_dist = ag_ds[i_agent, i]
                        ag_state[i_agent].long_speed = ag_vs[i_agent, i]
                        if i_agent == 0:
                            ag_state[i_agent].long_acc = AG1_ACCS[i_test]
                        else:
                            ag_state[i_agent].long_acc = 0
                        ag_state[i_agent] = add_entry_exit_times_to_state(
                                ag_state[i_agent], COLL_DIST)
                     
                    # loop through agent and get theoretical optical size and
                    # expansion
                    for i_agent in range(2):
                        thetas[i_agent, i] = get_agent_optical_size(
                            ag_state[i_agent], ag_state[1-i_agent], ag_image[1-i_agent])
                        thetaDots[i_agent, i] = get_agent_optical_exp(
                            ag_state[i_agent], ag_state[1-i_agent], ag_image[1-i_agent])
                        if i == 0:
                            thetaDotsNum[i_agent, i] = math.nan
                        else:
                            thetaDotsNum[i_agent, i] = (
                                (thetas[i_agent, i] - thetas[i_agent, i-1])
                                / TIME_STEP)
                    
                    # get TTC
                    ttcs[i] = get_time_to_sc_agent_collision(ag_state[0], ag_state[1], 
                                                             consider_acc)
                    
                if consider_acc:
                    lw = 4
                    color = 'lightgreen'
                else:
                    lw = 1
                    color = 'k'
                axs[2].plot(TIME_STAMPS, ttcs, '-', color=color, lw=lw)
            
            # do the rest of the plotting
            for i_agent in range(2):
                axs[0].plot(TIME_STAMPS, ag_vs[i_agent,:], 
                            color = AG_COLORS[i_agent])
                in_CS_idxs = np.nonzero(np.abs(ag_ds[i_agent,:]) <= COLL_DIST)[0]
                if len(in_CS_idxs) > 0:
                    t_en = TIME_STAMPS[in_CS_idxs[0]]
                    t_ex = TIME_STAMPS[in_CS_idxs[-1]]
                else:
                    t_en = math.nan
                    t_ex = math.nan
                axs[1].fill(np.array((t_en, t_ex, t_ex, t_en)), 
                            np.array((-1, -1, 1, 1)) * COLL_DIST, 
                            color = AG_COLORS[i_agent], alpha = 0.3,
                            edgecolor = None)
                axs[1].plot(TIME_STAMPS, ag_ds[i_agent,:], 
                                color = AG_COLORS[i_agent])
                axs[4].plot(TIME_STAMPS, thetas[i_agent,:], 
                            color = AG_COLORS[i_agent], lw = 1)
                axs[5].plot(TIME_STAMPS, thetaDotsNum[i_agent,:], 
                            color = AG_COLORS[i_agent], lw = 4, 
                            alpha = 0.3)
                axs[5].plot(TIME_STAMPS, thetaDots[i_agent,:], 
                            color = AG_COLORS[i_agent], lw = 1)
            axs[0].set_ylabel('v (m)')
            axs[1].axhline(COLL_DIST, color='r', linestyle='--', lw=0.5)
            axs[1].axhline(-COLL_DIST, color='r', linestyle='--', lw=0.5)
            axs[1].set_ylabel('d (m)')
            axs[1].set_ylim(np.array((-1, 1)) * COLL_DIST * 3)
            axs[2].set_ylabel('TTC (s)')
            (coll_margins, coll_idxs) = \
                get_sc_agent_collision_margins(ag_ds[0,:], ag_ds[1,:], COLL_DIST) 
            axs[3].plot(TIME_STAMPS, coll_margins, 'k-')
            axs[3].plot(TIME_STAMPS[coll_idxs], coll_margins[coll_idxs], 'r.')
            axs[3].set_ylabel('$d_{margin}$ (m)')
            axs[3].set_ylim(np.array((-.1, 1)) * COLL_DIST * 3)
            axs[4].set_ylabel(r'$\theta$ (rad)')
            axs[5].set_ylim(np.array((-0.2, 0.2)))
            axs[5].set_ylabel(r'$\dot{\theta}$ (rad/s)')
            axs[5].set_xlabel('t (s)')
            

        