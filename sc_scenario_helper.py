
import math
from enum import Enum

import commotions


class CtrlType(Enum):
    SPEED = 0
    ACCELERATION = 1

def get_acc_to_be_at_dist_at_time(speed, target_dist, target_time):
    
    assert target_time > 0    
    
    # general answer
    target_acc = 2 * (target_dist - speed * target_time) / (target_time ** 2)
    
    # we have to check whether this is in fact a stopping situation - if so
    # should apply stopping deceleration instead (to avoid passing beyond 
    # target distance and reversing back toward it again)
    if target_dist > 0 and speed > 0 and target_acc < 0:
        stop_acc = -speed ** 2 / (2 * target_dist)
        stop_time = speed / (-stop_acc)
        if stop_time < target_time:
            # stopping time is shorter than target time, so we will need to
            # stop at target and wait
            return stop_acc
        
    # in all other cases the general answer above should be the correct solution
    return target_acc
        
        

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
        oth_entry_time = (oth_d - coll_dist) / oth_v
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
        oth_entry_time = (oth_d - coll_dist) / oth_v
        ego_dist_to_entry = ego_d - coll_dist
        ego_acc_2nd = get_acc_to_be_at_dist_at_time(ego_v, ego_dist_to_entry,
                                                    oth_entry_time)
    else:
        # not yet reached conflict space, so get acceleration that has the ego 
        # agent be at entrance to the conflict space at the same time as the 
        # other agent exits it
        oth_exit_time = (oth_d + coll_dist) / oth_v
        ego_dist_to_entry = ego_d - coll_dist
        ego_acc_2nd = get_acc_to_be_at_dist_at_time(ego_v, ego_dist_to_entry,
                                                    oth_exit_time)
        # if the acceleration to free speed is lower than the acceleration
        # needed to enter just as the other agent exits, there is no need to
        # assume that the agent will move faster than its free speed
        ego_acc_2nd = min(ego_acc_2nd, ego_free_acc)
    
    return (ego_acc_1st, ego_acc_2nd)
        
        

# code for testing the helper functions

if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.close('all')
    
    
    ## TODO : Test case with acceleration-controlling agent
    
    
    
    EGO_CTRL_TYPE = CtrlType.SPEED
    EGO_ACTION_DUR = 0.5
    EGO_K = commotions.Parameters()
    EGO_V_FREE = 1.5
    EGO_Ds = np.arange(40, -5, -1)
    EGO_V = 1.5
    OTH_D = 40
    OTH_V = 10
    COLL_DIST = 2
    
    END_TIME = 10 # s
    TIME_STEP = 0.005 # s
    TIME_STAMPS = np.arange(0, END_TIME, TIME_STEP)
    
    # get other agent's movement
    oth_distances = OTH_D - OTH_V * TIME_STAMPS
    oth_entry_time = (OTH_D - COLL_DIST) / OTH_V
    oth_exit_time = (OTH_D + COLL_DIST) / OTH_V
    
    def plot_conflict_window(ax):
        ax.axhline(COLL_DIST, color='r', linestyle='--', lw=0.5)
        ax.axhline(-COLL_DIST, color='r', linestyle='--', lw=0.5)
        ax.axvline(oth_entry_time, color='r', linestyle='--', lw=0.5)
        ax.axvline(oth_exit_time, color='r', linestyle='--', lw=0.5)
        
    # plot other agent's path
    fig = plt.figure('Other agent')
    ax = fig.subplots()
    ax.plot(TIME_STAMPS, oth_distances, 'k-')
    plot_conflict_window(ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to conflict point (m)')
    
    fig_1st = plt.figure('Passing first')
    ax_1st = fig_1st.subplots()
    plot_conflict_window(ax_1st)
    fig_2nd = plt.figure('Passing second')
    ax_2nd = fig_2nd.subplots()
    plot_conflict_window(ax_2nd)
    
    def plot_pred_ds(ax, d, v, a):
        vs = v + a * TIME_STAMPS
        ds = d - np.cumsum(vs * TIME_STEP)
        ax.plot(TIME_STAMPS, ds, 'k-', lw=0.5)
        
    for ego_d in EGO_Ds:
        (acc_1st, acc_2nd) = get_access_order_accs(
                EGO_CTRL_TYPE, EGO_ACTION_DUR, EGO_K, EGO_V_FREE,
                ego_d, EGO_V, OTH_D, OTH_V, COLL_DIST)
        plot_pred_ds(ax_1st, ego_d, EGO_V, acc_1st)
        plot_pred_ds(ax_2nd, ego_d, EGO_V, acc_2nd)
    
    
    