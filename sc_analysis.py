"""
Functions for analysing completed sc_scenario.SCSimulation simulations of
interactions between two agents on perpendicular paths: interaction safety
metrics (post-encroachment time, time to collision, distance margins), and
export of the simulated time series to tidy pandas DataFrames / CSV files.

Only numpy and the framework's own modules are imported at module level;
pandas is imported lazily, i.e., only when calling the DataFrame/CSV
functions.

Conventions used throughout (consistent with SCPaper/sc_fitting.py):

    - The conflict space (CS) is the region where a collision between the
      two agents is possible; agent i is inside it when
      abs(sim.agents[i].signed_CP_dists) < sim.agents[i].coll_dist (with the
      signed distance to the conflict point positive before the conflict
      point).
    - An agent's CS entry time is the time stamp of the first simulation
      sample with signed_CP_dist <= coll_dist, and its CS exit time is the
      time stamp of the first sample with signed_CP_dist < -coll_dist, i.e.,
      the same definitions as in sc_fitting.metric_agent_entry_time() and
      sc_fitting.metric_agent_exit_time(). An agent starting inside (or past)
      the CS thus gets the start time as entry time.
    - Only samples up to and including sim.actual_end_time are considered
      (the simulation may have been stopped early by a stop criterion, in
      which case the stored arrays are NaN, or contain only initial values,
      beyond that time). Entry/exit events that did not happen within the
      simulated period are returned as math.nan.
    - Time-based metrics derived from the sampled time series have the
      resolution of the simulation time step.
"""

import math
import types
import numpy as np

import sc_scenario_helper


# =============================================================================
# Basic helpers
# =============================================================================

def get_n_simulated_steps(sim):
    """ Return the number of time steps that were actually simulated in sim,
        i.e., up to and including the time step at sim.actual_end_time.

        Parameters
        ----------
        sim: sc_scenario.SCSimulation
            A simulation object, for which .run() has been called.

        Returns
        -------
        int
    """
    if not hasattr(sim, 'actual_end_time'):
        raise Exception('The simulation does not seem to have been run yet.')
    return int(np.argmin(np.abs(sim.time_stamps - sim.actual_end_time))) + 1


def get_simulated_time_stamps(sim):
    """ Return the time stamps of the time steps actually simulated in sim
        (see get_n_simulated_steps()).
    """
    return sim.time_stamps[:get_n_simulated_steps(sim)]


def get_cs_entry_exit_times(sim, i_agent=None):
    """ Return the times of conflict space entry and exit for one agent (or
        all agents if i_agent is None).

        Parameters
        ----------
        sim: sc_scenario.SCSimulation
            A simulation object, for which .run() has been called.
        i_agent: int or None
            Index of the agent in sim.agents, or None to return a tuple of
            (entry_time, exit_time) pairs for all agents in sim.agents.
            Default: None

        Returns
        -------
        tuple of float, or tuple of tuples of float if i_agent is None
            For a single agent, (entry_time, exit_time), where entry_time is
            the time stamp of the first simulated sample with
            signed_CP_dist <= coll_dist, and exit_time is the time stamp of the
            first simulated sample with signed_CP_dist < -coll_dist (same
            semantics as sc_fitting.metric_agent_entry_time() and
            sc_fitting.metric_agent_exit_time()). Each value is math.nan if
            the event did not happen within the simulated period (i.e., up to
            and including sim.actual_end_time).
    """
    if i_agent is None:
        return tuple(get_cs_entry_exit_times(sim, i)
                     for i in range(len(sim.agents)))
    agent = sim.agents[i_agent]
    n_steps = get_n_simulated_steps(sim)
    cp_dists = agent.signed_CP_dists[:n_steps]
    entered_samples = np.nonzero(cp_dists <= agent.coll_dist)[0]
    if len(entered_samples) == 0:
        entry_time = math.nan
    else:
        entry_time = float(sim.time_stamps[entered_samples[0]])
    exited_samples = np.nonzero(cp_dists < -agent.coll_dist)[0]
    if len(exited_samples) == 0:
        exit_time = math.nan
    else:
        exit_time = float(sim.time_stamps[exited_samples[0]])
    return (entry_time, exit_time)


def get_collision_margins(sim):
    """ Return the per-time-step distance margins to collision between the
        two agents, and the indices of the time steps with collision, using
        sc_scenario_helper.get_sc_agent_collision_margins(). Only the
        simulated time steps are included in the returned arrays.

        Returns
        -------
        tuple
            (collision_margins, collision_idxs), where collision_margins is a
            numpy array of length get_n_simulated_steps(sim), with the sum
            over agents of the distance (m) by which each agent is outside of
            the conflict space (zero during collision), and collision_idxs is
            an array of the time step indices where both agents are inside
            the conflict space.
    """
    n_steps = get_n_simulated_steps(sim)
    return sc_scenario_helper.get_sc_agent_collision_margins(
        sim.agents[0].signed_CP_dists[:n_steps],
        sim.agents[1].signed_CP_dists[:n_steps],
        sim.agents[0].coll_dist, sim.agents[1].coll_dist)


def had_collision(sim):
    """ Return True if both agents were inside the conflict space at the
        same time at any simulated time step (same definition as
        sc_fitting.metric_collision()), otherwise False.
    """
    __, collision_idxs = get_collision_margins(sim)
    return len(collision_idxs) > 0


def get_min_distance_margin(sim):
    """ Return the minimum, across the simulated time steps, of the distance
        margin (m) to collision between the agents, as calculated by
        sc_scenario_helper.get_sc_agent_collision_margins(), i.e., the sum
        over agents of the distance by which each agent is outside of the
        conflict space. Zero if there was a collision.
    """
    collision_margins, __ = get_collision_margins(sim)
    return float(np.nanmin(collision_margins))


# =============================================================================
# Post-encroachment time
# =============================================================================

def get_first_agent_in_cs(sim):
    """ Return the index in sim.agents of the agent which entered the
        conflict space first (as per get_cs_entry_exit_times()), or None if
        neither agent entered the conflict space within the simulated period.
        If both agents entered at the same time step, the agent with the
        lower index is returned. (Note that sim.first_passer may differ from
        this in the unusual case of an agent starting past the conflict
        space.)
    """
    entry_times = np.array([get_cs_entry_exit_times(sim, i_agent)[0]
                            for i_agent in range(len(sim.agents))])
    if np.all(np.isnan(entry_times)):
        return None
    return int(np.nanargmin(entry_times))


def get_pet(sim, return_order=False):
    """ Return the post-encroachment time (PET) of the interaction in sim,
        i.e., the time from the first agent exiting the conflict space to
        the second agent entering it, with entry and exit times as defined
        in get_cs_entry_exit_times() (so the PET has the resolution of the
        simulation time step).

        Parameters
        ----------
        sim: sc_scenario.SCSimulation
            A simulation object, for which .run() has been called.
        return_order: bool
            If True, also return the index of the agent that entered the
            conflict space first (see get_first_agent_in_cs()). Default: False

        Returns
        -------
        float, or tuple (float, int or None) if return_order is True
            The PET in seconds:
            - 0 if there was a collision, i.e., if both agents were inside
              the conflict space at the same time (see had_collision()); also
              0 (rather than negative) in the boundary case of the second
              agent's entry sample preceding the first agent's exit sample
              without a strict collision;
            - math.nan if the PET is undefined within the simulated period,
              i.e., if not both agents entered the conflict space, or if the
              first agent did not exit it.
    """
    i_first = get_first_agent_in_cs(sim)
    if i_first is None:
        pet = math.nan
    elif had_collision(sim):
        pet = 0.
    else:
        __, first_exit_time = get_cs_entry_exit_times(sim, i_first)
        second_entry_time, __ = get_cs_entry_exit_times(sim, 1 - i_first)
        if math.isnan(first_exit_time) or math.isnan(second_entry_time):
            pet = math.nan
        else:
            pet = max(0., second_entry_time - first_exit_time)
    if return_order:
        return (pet, i_first)
    else:
        return pet


# =============================================================================
# Time to collision
# =============================================================================

def get_ttc_series(sim, consider_acc=False):
    """ Return the time to collision (TTC) between the two agents at each
        time step of sim, i.e., the time left until both agents are inside
        the conflict space at the same time, if continuing at their current
        speeds (and accelerations, if consider_acc is True). Calculated using
        sc_scenario_helper.add_entry_exit_times_to_state() and
        sc_scenario_helper.get_time_to_sc_agent_collision(), from the agents'
        signed_CP_dists, trajectory.long_speed, and trajectory.long_acc (the
        latter being the acceleration applied from each time step to the
        next; negative speeds, not expected in these simulations, are
        treated as zero).

        Parameters
        ----------
        sim: sc_scenario.SCSimulation
            A simulation object, for which .run() has been called.
        consider_acc: bool
            If True, the agents' current accelerations are considered when
            projecting entry/exit times, otherwise constant speeds are
            assumed. Default: False

        Returns
        -------
        numpy array of float
            Same length as sim.time_stamps, with the TTC in seconds for each
            time step; np.inf when the agents are not on a collision course,
            0 during an ongoing collision, and np.nan for time steps after
            sim.actual_end_time (if the simulation was stopped early).
    """
    n_steps = get_n_simulated_steps(sim)
    ttcs = np.full(len(sim.time_stamps), math.nan)
    for i_time_step in range(n_steps):
        states = []
        for agent in sim.agents:
            state = types.SimpleNamespace(
                signed_CP_dist=agent.signed_CP_dists[i_time_step],
                long_speed=max(0., agent.trajectory.long_speed[i_time_step]),
                long_acc=agent.trajectory.long_acc[i_time_step])
            states.append(sc_scenario_helper.add_entry_exit_times_to_state(
                state, agent.coll_dist))
        ttcs[i_time_step] = sc_scenario_helper.get_time_to_sc_agent_collision(
            states[0], states[1], consider_acc=consider_acc)
    return ttcs


def get_min_ttc(sim, consider_acc=False):
    """ Return the minimum, across the simulated time steps, of the time to
        collision returned by get_ttc_series() (np.inf if the agents were
        never on a collision course, 0 if there was a collision).
    """
    return float(np.nanmin(get_ttc_series(sim, consider_acc=consider_acc)))


# =============================================================================
# Summary and export
# =============================================================================

def summarise_interaction(sim, consider_acc=False):
    """ Return a dict with summary metrics for the interaction in sim.

        Parameters
        ----------
        sim: sc_scenario.SCSimulation
            A simulation object, for which .run() has been called.
        consider_acc: bool
            Passed on to get_min_ttc(). Default: False

        Returns
        -------
        dict
            With the following keys:
            - 'first_passer': name of sim.first_passer, or None if neither
              agent entered the conflict space
            - 'pet': see get_pet()
            - 'min_ttc': see get_min_ttc()
            - 'min_distance_margin': see get_min_distance_margin()
            - 'collision': see had_collision()
            - '<name>_entry_time', '<name>_exit_time' for each agent, with
              <name> the agent's name: see get_cs_entry_exit_times()
            - 'actual_end_time': sim.actual_end_time
    """
    if getattr(sim, 'first_passer', None) is None:
        first_passer = None
    else:
        first_passer = sim.first_passer.name
    summary = {'first_passer': first_passer,
               'pet': get_pet(sim),
               'min_ttc': get_min_ttc(sim, consider_acc=consider_acc),
               'min_distance_margin': get_min_distance_margin(sim),
               'collision': had_collision(sim)}
    for i_agent, agent in enumerate(sim.agents):
        entry_time, exit_time = get_cs_entry_exit_times(sim, i_agent)
        summary[f'{agent.name}_entry_time'] = entry_time
        summary[f'{agent.name}_exit_time'] = exit_time
    summary['actual_end_time'] = float(sim.actual_end_time)
    return summary


def _get_action_label(ctrl_delta):
    return f'a{ctrl_delta:+g}'


def sim_to_dataframe(sim, include_perception=True, include_internal=False,
                     ttc_consider_acc=False):
    """ Return a tidy, wide pandas DataFrame with the time series of sim,
        with one row per simulated time step (time steps after
        sim.actual_end_time are excluded).

        Parameters
        ----------
        sim: sc_scenario.SCSimulation
            A simulation object, for which .run() has been called.
        include_perception: bool
            If True, include each agent's perception of the other agent.
            Default: True
        include_internal: bool
            If True, include also some of the agents' internal model states.
            Default: False
        ttc_consider_acc: bool
            Passed as consider_acc to get_ttc_series(). Default: False

        Returns
        -------
        pandas.DataFrame
            With the columns (where <name> is the name of each agent):
            - 'time': the time stamp (s)
            - '<name>_x', '<name>_y': position (m)
            - '<name>_speed': longitudinal speed (m/s)
            - '<name>_acc': longitudinal acceleration (m/s^2), applied from
              this time step to the next
            - '<name>_cp_dist': signed distance to the conflict point (m),
              positive before the conflict point
            - 'ttc': time to collision (s), see get_ttc_series()
            if include_perception is True:
            - '<name>_perc_oth_cp_dist', '<name>_perc_oth_speed': the agent's
              perceived signed conflict point distance and speed of the other
              agent (NaN for agents with constant acceleration, which do not
              update their perception)
            if include_internal is True, for each action alternative with
            control delta <d> (formatted with sign, e.g., 'a-0.5', 'a+0',
            'a+1') and each behaviour <beh> of the other agent ('const',
            'pass1st', 'pass2nd'):
            - '<name>_est_val_a<d>': estimated value of the action
              (states.est_action_vals)
            - '<name>_beh_prob_<beh>_given_a<d>': estimated probability of
              the other agent's behaviour, given the action
              (states.beh_probs_given_actions)
            (all NaN for agents with constant acceleration).
    """
    import pandas as pd
    n_steps = get_n_simulated_steps(sim)
    columns = {'time': sim.time_stamps[:n_steps]}
    for agent in sim.agents:
        name = agent.name
        columns[f'{name}_x'] = agent.trajectory.pos[0, :n_steps]
        columns[f'{name}_y'] = agent.trajectory.pos[1, :n_steps]
        columns[f'{name}_speed'] = agent.trajectory.long_speed[:n_steps]
        columns[f'{name}_acc'] = agent.trajectory.long_acc[:n_steps]
        columns[f'{name}_cp_dist'] = agent.signed_CP_dists[:n_steps]
        if include_perception:
            x_perceived = agent.perception.states.x_perceived
            columns[f'{name}_perc_oth_cp_dist'] = x_perceived[0, :n_steps]
            columns[f'{name}_perc_oth_speed'] = x_perceived[1, :n_steps]
    columns['ttc'] = get_ttc_series(sim, consider_acc=ttc_consider_acc)[:n_steps]
    if include_internal:
        from sc_scenario import i_CONSTANT, i_PASS1ST, i_PASS2ND
        beh_labels = {i_CONSTANT: 'const', i_PASS1ST: 'pass1st',
                      i_PASS2ND: 'pass2nd'}
        for agent in sim.agents:
            name = agent.name
            action_labels = [_get_action_label(ctrl_delta)
                             for ctrl_delta in agent.params.ctrl_deltas]
            for i_action, action_label in enumerate(action_labels):
                columns[f'{name}_est_val_{action_label}'] = \
                    agent.states.est_action_vals[i_action, :n_steps]
            for i_beh in sorted(beh_labels.keys()):
                for i_action, action_label in enumerate(action_labels):
                    columns[f'{name}_beh_prob_{beh_labels[i_beh]}_given_'
                            f'{action_label}'] = \
                        agent.states.beh_probs_given_actions[
                            i_beh, i_action, :n_steps]
    # copy arrays, so that the DataFrame does not share memory with sim
    return pd.DataFrame({key: np.array(value, dtype=float)
                         for key, value in columns.items()})


def export_csv(sim, path, include_perception=True, include_internal=False,
               ttc_consider_acc=False, **kwargs):
    """ Write the time series of sim to a CSV file, as obtained from
        sim_to_dataframe().

        Parameters
        ----------
        sim: sc_scenario.SCSimulation
            A simulation object, for which .run() has been called.
        path: str or path-like
            Path of the CSV file to write.
        include_perception, include_internal, ttc_consider_acc:
            Passed on to sim_to_dataframe().
        **kwargs:
            Passed on to pandas.DataFrame.to_csv(); index=False is used
            unless specified otherwise.

        Returns
        -------
        pandas.DataFrame
            The DataFrame that was written to file.
    """
    df = sim_to_dataframe(sim, include_perception=include_perception,
                          include_internal=include_internal,
                          ttc_consider_acc=ttc_consider_acc)
    kwargs.setdefault('index', False)
    df.to_csv(path, **kwargs)
    return df
