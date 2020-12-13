
import math
from enum import Enum

import commotions

EPSILON = np.finfo(float).eps

class CtrlType(Enum):
    SPEED = 0
    ACCELERATION = 1


def get_acc_to_be_at_dist_at_time(speed, target_dist, target_time):
    """ Return acceleration required to travel a further distance target_dist
        in time target_time if starting at speed speed. Handle infinite
        target_time by returning machine epsilon with correct sign.
    """
    
    assert target_time > 0    
    
    # time horizon finite or not?
    if target_time == math.inf:
        # infinite time horizon - some special cases with near-zero acceleration
        if target_dist == 0:
            # at target, so just reverse any speed
            return -np.sign(speed) * EPSILON
        elif speed == 0:
            # at zero speed, so remove remaining distance
            return np.sign(target_dist) * EPSILON
        else:  
            # neither distance nor speed are zero, so sign of acceleration
            # depends on combination of both - and we need to do the stopping 
            # check below, so can't return straight away
            target_acc = -np.sign(speed * target_dist) * EPSILON
    else:
        # finite time horizon --> general answer for most situations
        target_acc = 2 * (target_dist - speed * target_time) / (target_time ** 2)
    
    # check whether this is a stopping situation - if so we should apply 
    # stopping deceleration instead (to avoid passing beyond target distance 
    # and reversing back toward it again)
    if target_dist > 0 and speed > 0 and target_acc < 0:
        stop_acc = -speed ** 2 / (2 * target_dist)
        stop_time = speed / (-stop_acc)
        if stop_time < target_time:
            # stopping time is shorter than target time, so we will need to
            # stop at target and wait
            return stop_acc
    
    # not a stopping situation, so return what was calculated above
    return target_acc
        

def get_entry_exit_times(d, v, coll_dist):
    """ Return tuple of entry and exit times into conflict zone defined by +/-coll_dist
        around conflict point, for agent currently at distance d to conflict 
        point and travelling toward it at speed v. Returning inf for zero speeds,
        but not checking for negative distances or speeds.
    """
    if v == 0:
        return (math.inf, math.inf)
    else:
        entry_time = (d - coll_dist) / v
        exit_time = (d + coll_dist) / v
        return (entry_time, exit_time)


def get_access_order_accs(ego_ctrl_type, ego_action_dur, ego_k, ego_v_free,
                          ego_d, ego_v, oth_d, oth_v, coll_dist):
    """ Return a tuple (acc_1st, acc_2nd) of minimally effortful accelerations 
        for the ego agent of CtrlType ego_ctrl_type with action duration 
        ego_action_dur and cost function gains ego_k, igned distance ego_d to 
        conflict point (positive sign 
        meaning not having reached conflict 
        point yet), speed ego_v, and free speed ego_v_free, to pass 
        respectively before or after, respecting collision distance coll_dist, 
        the other agent described by oth_d and oth_v, assumed to maintain 
        zero acceleration from the current time. If the ego agent has already 
        exited the conflict space, both outputs will be math.nan. If the other 
        agent has already entered the conflict space, acc_1st will be math.nan.
    """
    # not supporting special cases with negative initial speeds
    assert ego_v >= 0 and oth_v >= 0
    
    # has the ego agent already exited the conflict space?
    if ego_d <= -coll_dist:
        # yes, so nothing to do here
        return (math.nan, math.nan)
    
    # get other agent's entry/exit times into to the conflict space
    (oth_entry_time, oth_exit_time) = get_entry_exit_times(oth_d, oth_v, coll_dist)
    
    # get ego agent's acceleration if just aiming for free speed
    if ego_ctrl_type is CtrlType.SPEED:
        # assuming straight acc to free speed
        ego_free_acc = (ego_v_free - ego_v) / ego_action_dur
    else:
        # calculate the expected acceleration given the current deviation
        # from the free speed (see hand written notes from 2020-07-08)
        dev_from_v_free = ego_v - ego_v_free
        ego_free_acc = (
                - ego_k._dv * dev_from_v_free * ego_action_dur 
                / (0.5 * ego_k._dv * ego_action_dur ** 2 
                   + 2 * ego_k._da)
                )
    
    # get acceleration needed to pass first
    # - other agent already entered conflict space?
    if oth_d <= coll_dist:
        ego_acc_1st = math.nan
    else:
        # no, so it is theoretically possible to pass in front of it
        # --> get acceleration that has the ego agent be at exit of the conflict 
        # space at the same time as the other agent enters it
        ego_dist_to_exit = ego_d + coll_dist
        ego_acc_1st = get_acc_to_be_at_dist_at_time(ego_v, ego_dist_to_exit,
                                                    oth_entry_time)
        # if the acceleration to free speed is higher than the acceleration
        # needed to exit just as the other agent enters, there is no need to
        # assume that the agent will move slower than its free speed
        ego_acc_1st = max(ego_acc_1st, ego_free_acc)
        
    # get acceleration needed to pass second
    # - ego agent already entered conflict space?
    if ego_d <= coll_dist:
        # yes, so need to move back out of it again before the other agent 
        # arrives
        ego_dist_to_entry = ego_d - coll_dist
        ego_acc_2nd = get_acc_to_be_at_dist_at_time(ego_v, ego_dist_to_entry,
                                                    oth_entry_time)
    else:
        # not yet reached conflict space, so get acceleration that has the ego 
        # agent be at entrance to the conflict space at the same time as the 
        # other agent exits it
        ego_dist_to_entry = ego_d - coll_dist
        ego_acc_2nd = get_acc_to_be_at_dist_at_time(ego_v, ego_dist_to_entry,
                                                    oth_exit_time)
        # if the acceleration to free speed is lower than the acceleration
        # needed to enter just as the other agent exits, there is no need to
        # assume that the agent will move faster than its free speed
        ego_acc_2nd = min(ego_acc_2nd, ego_free_acc)
    
    return (ego_acc_1st, ego_acc_2nd)
        
        

# "unit tests"

if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.close('all')
    

# =============================================================================
#   Testing get_access_order_accs()
# =============================================================================

    # defining the tests
    TESTS = ('Stationary ped., moving veh.', 'Moving ped., moving veh.', 
             'Stationary veh., moving ped.', 'Moving veh., moving ped.', 
             'Moving veh., stationary ped.', 'Stationary veh., stationary ped.')
    N_TESTS = len(TESTS)
    EGO_CTRL_TYPES = (CtrlType.SPEED, CtrlType.SPEED, CtrlType.ACCELERATION, 
                      CtrlType.ACCELERATION, CtrlType.ACCELERATION, 
                      CtrlType.ACCELERATION)
    EGO_VS = (0, 1.5, 0, 10, 10, 0)
    OTH_VS = (10, 10, 1.5, 1.5, 0, 0)
    
    # constants depending on ego agent control type
    EGO_D_MAX = {}
    EGO_D_MAX[CtrlType.SPEED] = 40
    EGO_D_MAX[CtrlType.ACCELERATION] = 100
    EGO_V_FREE = {}
    EGO_V_FREE[CtrlType.SPEED] = 1.5
    EGO_V_FREE[CtrlType.ACCELERATION] = 10
    EGO_K = {}
    EGO_K[CtrlType.SPEED] = commotions.Parameters() # not used
    EGO_K[CtrlType.ACCELERATION] = commotions.Parameters()
    EGO_K[CtrlType.ACCELERATION]._g = 1 
    EGO_K[CtrlType.ACCELERATION]._dv = 0.05
    EGO_K[CtrlType.ACCELERATION]._da = 0.01
    OTH_D = {}
    OTH_D[CtrlType.SPEED] = 40
    OTH_D[CtrlType.ACCELERATION] = 6
    # other constants
    EGO_ACTION_DUR = 0.5
    COLL_DIST = 2
    END_TIME = 10 # s
    TIME_STEP = 0.005 # s
    TIME_STAMPS = np.arange(0, END_TIME, TIME_STEP)
    
    # plot fcns
    def plot_conflict_window(ax, oth_entry_time, oth_exit_time):
        ax.axhline(COLL_DIST, color='r', linestyle='--', lw=0.5)
        ax.axhline(-COLL_DIST, color='r', linestyle='--', lw=0.5)
        ax.axvline(oth_entry_time, color='r', linestyle='--', lw=0.5)
        ax.axvline(oth_exit_time, color='r', linestyle='--', lw=0.5)
    def plot_pred_ds(ax, d, v, a):
        vs = v + a * TIME_STAMPS
        ds = d - np.cumsum(vs * TIME_STEP)
        ax.plot(TIME_STAMPS, ds, 'k-', lw=0.5)
    
    # loop through the test cases
    for i_test, test_name in enumerate(TESTS):
        
        # get test settings
        ego_ctrl_type = EGO_CTRL_TYPES[i_test]
        ego_v = EGO_VS[i_test]
        oth_v = OTH_VS[i_test]
        ego_ds = np.linspace(-5, EGO_D_MAX[ego_ctrl_type], 20)
        ego_v_free = EGO_V_FREE[ego_ctrl_type]
        ego_k = EGO_K[ego_ctrl_type]
        oth_d = OTH_D[ego_ctrl_type]
        
        # get other agent's movement
        oth_distances = oth_d - oth_v * TIME_STAMPS
        (oth_entry_time, oth_exit_time) = get_entry_exit_times(
                oth_d, oth_v, COLL_DIST)
            
        # get figure window for test results
        fig = plt.figure(test_name, figsize = (10, 5))
        axs = fig.subplots(1, 3)
        
        # plot other agent's path
        axs[0].plot(TIME_STAMPS, oth_distances, 'k-')
        plot_conflict_window(axs[0], oth_entry_time, oth_exit_time)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('$d_{oth}$ (m)')
        axs[0].set_title("Other agent")
        
        # prepare subplots for passing first/second plots
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('$d_{ego}$ (m)')
        axs[1].set_title('Ego agent passing first')
        plot_conflict_window(axs[1], oth_entry_time, oth_exit_time)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('$d_{ego}$ (m)')
        axs[2].set_title('Ego agent passing second')
        plot_conflict_window(axs[2], oth_entry_time, oth_exit_time)
            
        # loop through initial distances
        for ego_d in ego_ds:
            (acc_1st, acc_2nd) = get_access_order_accs(
                    ego_ctrl_type, EGO_ACTION_DUR, ego_k, ego_v_free,
                    ego_d, ego_v, oth_d, oth_v, COLL_DIST)
            plot_pred_ds(axs[1], ego_d, ego_v, acc_1st)
            plot_pred_ds(axs[2], ego_d, ego_v, acc_2nd)
            
        plt.tight_layout()
        plt.show()
    
    
    