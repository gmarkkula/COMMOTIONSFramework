import numpy as np
import matplotlib.pyplot as plt
import math


# constants

COLLISION_DISTANCE = 1

START_TIME = 0 # s
END_TIME = 15 # s
TIME_STEP = 0.1 # s
TIME_STAMPS = np.arange(START_TIME, END_TIME, TIME_STEP)

SPEED_ACTIONS = np.array([-1, -0.5, 0, 0.5, 1]) # m/s
HEADING_ACTIONS = np.array([-10, -5, 0, 5, 10]) * math.pi/180 # rad

ACTION_DURATION = 0.3 # s
PREDICTION_TIME = ACTION_DURATION
ACTION_TIME_STEPS = math.ceil(ACTION_DURATION / TIME_STEP)
PREDICTION_TIME_STEPS = math.ceil(PREDICTION_TIME / TIME_STEP)

ACTIONS_VECTOR_LENGTH = TIME_STAMPS.size + PREDICTION_TIME_STEPS

AGENT_COLORS = ('c', 'm')


def get_default_params():
    params = {
        "k_g": 1,
        "k_c": .3,
        "C_v": .3,
        "C_t": .3
    }
    return params


def run_simulation(pos_agents, pos_goals, params = get_default_params(), \
    speed_agents = None, heading_agents = None, pos_obstacles = None):
    """Run the multi-agent simulation.

    Arguments:
    pos_agents -- (N,2) numpy array with initial (x, y) positions in m for N 
        agents 
    pos_goals -- (N,2) numpy array with goal (x, y) positions in m for each 
        agent

    Keyword arguments:
    params -- dictionary with model parameters (default get_default_params())
    speed_agents -- (N,) numpy array with initial speeds in m/s for each agent 
        (default None, yields zero initial speed)
    heading_agents -- (N,) numpy array with initial headings in rad for each 
        agent (default None, yields initial headings toward goals)
    pos_obstacles -- (M,2) numpy array with (x, y) coordinates for M obstacles 
        (default None, for no obstacles)
    """
    # init
    # - get and check no of agents and goals
    N_AGENTS = pos_agents.shape[0]
    assert pos_agents.shape[1] == 2
    assert pos_goals.shape == pos_agents.shape
    # - init matrices and vectors for agent trajectories - agents in rows
    acceleration_trajs = np.zeros((N_AGENTS, ACTIONS_VECTOR_LENGTH))
    yaw_rate_trajs = np.zeros((N_AGENTS, ACTIONS_VECTOR_LENGTH))
    speed_trajs = np.zeros((N_AGENTS, TIME_STAMPS.size)) 
    heading_trajs = np.zeros((N_AGENTS, TIME_STAMPS.size))
    pos_trajs = np.zeros((N_AGENTS, 2, TIME_STAMPS.size))
    # - initial values (when specified)
    pos_trajs[:, :, 0] = pos_agents
    if speed_agents is not None:
        speed_trajs[:, 0] = speed_agents
    else:
        speed_trajs[:, 0] = 0
    if heading_agents is not None:
        heading_trajs[:, 0] = heading_agents
    else:
        agents_to_goals = pos_goals - pos_agents
        heading_trajs[:, 0] = \
            np.arctan2(agents_to_goals[:, 1], agents_to_goals[:, 0])
    # - any obstacles?
    if pos_obstacles is None:
        N_OBSTACLES = 0
    else:
        N_OBSTACLES = pos_obstacles.shape[0]
    # - matrix for holding predicted positions
    pos_predictions = np.zeros((N_AGENTS, 2))


    # helper functions

    def get_future_state(i_agent, local_accelerations, local_yaw_rates, 
        i_curr_time_step, n_time_steps_to_advance):
        
        local_pos = (pos_trajs[i_agent, :, i_curr_time_step])
        local_speed = (speed_trajs[i_agent, i_curr_time_step])
        local_heading = (heading_trajs[i_agent, i_curr_time_step])
        
        for i in range(i_curr_time_step + 1, \
            i_curr_time_step + n_time_steps_to_advance + 1):
            local_speed += local_accelerations[i-1] * TIME_STEP
            local_heading += local_yaw_rates[i-1] * TIME_STEP
            local_pos = local_pos + TIME_STEP * local_speed \
                * np.array([math.cos(local_heading), math.sin(local_heading)])
        return (local_pos, local_speed, local_heading)
    

    def add_action(state_vector, action_magnitude, i_action_time_step):
        state_vector[i_action_time_step:i_action_time_step+ACTION_TIME_STEPS] += \
            action_magnitude / ACTION_DURATION


    def get_real_quadratic_roots(a, b, c):
        d = b ** 2 - 4 * a * c # discriminant
        if d < 0:
            return ()
        elif d == 0:
            x = -b / (2 * a)
            return (x,)
        else:
            x1 = (-b + math.sqrt(d)) / (2 * a)
            x2 = (-b - math.sqrt(d)) / (2 * a)
            return (x1, x2)


    def get_time_to_agent_collision_from_pred_state(\
        pos, speed, heading, i_other_agent, i_base_time_step):
        # get some basics
        other_pos = pos_predictions[i_other_agent, :]
        other_speed = speed_trajs[i_other_agent, i_base_time_step]
        other_heading = heading_trajs[i_other_agent, i_base_time_step]
        delta_x = other_pos[0] - pos[0]
        delta_y = other_pos[1] - pos[1]
        delta_v_x = other_speed * math.cos(other_heading) \
            - speed * math.cos(heading)
        delta_v_y = other_speed * math.sin(other_heading) \
            - speed * math.sin(heading)
        # get coefficients of quadratic equation for squared distance
        # D^2 = at^2 + bt + c 
        a = delta_v_x ** 2 + delta_v_y ** 2
        b = 2 * (delta_x * delta_v_x + delta_y * delta_v_y)
        c = delta_x ** 2 + delta_y ** 2
        # get roots t for D^2 = D_collision^2 <=> D^2 - D_collision^2 = 0
        coll_times = \
            get_real_quadratic_roots(a, b, c - COLLISION_DISTANCE ** 2)
        # interpret roots
        if len(coll_times) == 0:
            # no collision (identical agent headings)
            return math.inf
        elif len(coll_times) == 1:
            # just barely touching
            return coll_times[0]
        else:
            # passing collision threshold twice (typical case) - check when
            # this happens
            if math.copysign(1, coll_times[0]) == math.copysign(1, coll_times[1]):
                # both in future or both in past
                if coll_times[0] > 0:
                    # both in future
                    return min(coll_times)
                else:
                    # both in past
                    return math.inf
            else:
                # one in future one in past - i.e., collision now
                return 0

    
    def get_value_of_predicted_state(i_agent, i_base_time_step, \
        pos, speed, heading, verbose = False):
        
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        
        # reward for progress toward goal and cost for speed
        vector_to_goal = pos_goals[i_agent, :] - pos
        heading_to_goal_vector = vector_to_goal / np.linalg.norm(vector_to_goal)
        heading_toward_goal_component = np.dot(heading_vector, heading_to_goal_vector)
        goal_distance_change_rate = -heading_toward_goal_component * speed
        value = -params["k_g"] * goal_distance_change_rate \
            - params["C_v"] * speed ** 2
        
        # cost for being on collision course with obstacles
        for i_obstacle in range(N_OBSTACLES):
            vector_to_obstacle = x_obstacles[i_obstacle,:] - x
            heading_to_obstacle_vector = \
                vector_to_obstacle / np.linalg.norm(vector_to_obstacle)
            heading_toward_obstacle_component = \
                np.dot(heading_vector, heading_to_obstacle_vector)
            angle_to_obstacle_rel_heading = \
                math.acos(heading_toward_obstacle_component)
            lateral_distance_to_obstacle = \
                np.linalg.norm(vector_to_obstacle) * math.sin(angle_to_obstacle_rel_heading)
            if lateral_distance_to_obstacle < COLLISION_DISTANCE \
                and heading_toward_obstacle_component > 0:
                long_distance_to_obstacle = np.linalg.norm(vector_to_obstacle) \
                    * math.cos(angle_to_obstacle_rel_heading)
                time_to_obstacle_collision = long_distance_to_obstacle / speed
                value += -params["k_c"] / time_to_obstacle_collision   

        # cost for being on collision course with other agents
        for i_other_agent in range(N_AGENTS):
            if i_other_agent == i_agent:
                continue
            # calculate future time at which agents come within collision
            # distance of each other (if at all)
            time_to_agent_collision = \
                get_time_to_agent_collision_from_pred_state(\
                pos, speed, heading, i_other_agent, i_base_time_step)
            if time_to_agent_collision == 0:
                value = -math.inf
            elif time_to_agent_collision < math.inf:
                value += -params["k_c"] / time_to_agent_collision  
        
        if verbose:
            #print(angle_to_obstacle_rel_heading * 180 / math.pi)
            #print(lateral_distance_to_obstacle, time_to_obstacle_collision)
            print(heading_toward_goal_component)
        
        return value


    # simulation loop

    for i_time_step, time_stamp in enumerate(TIME_STAMPS):

        # first loop through agents, for Euler step and constant-speed 
        # predictions
        for i_agent in range(N_AGENTS):

            # do Euler step
            if i_time_step > 0:
                (pos_trajs[i_agent, :, i_time_step], speed_trajs[i_agent, \
                    i_time_step], heading_trajs[i_agent, i_time_step]) = \
                    get_future_state(i_agent, acceleration_trajs[i_agent, :], \
                        yaw_rate_trajs[i_agent, :], i_time_step-1, 1)

            # simulate all agents a prediction interval ahead, with constant 
            # speeds
            curr_pos = pos_trajs[i_agent, :, i_time_step]
            curr_speed = speed_trajs[i_agent, i_time_step]
            curr_heading = heading_trajs[i_agent, i_time_step]
            heading_vector = \
                np.array([math.cos(curr_heading), math.sin(curr_heading)])
            pos_predictions[i_agent, :] = curr_pos \
                + curr_speed * PREDICTION_TIME \
                * heading_vector

        # second loop through agents, to find the best action for each, 
        # assuming constant speeds for all other agents
        for i_agent in range(N_AGENTS):

            # decide on action for this agent and this time step - by looping 
            # through the alternatives and calculating predicted rewards
            best_value = -math.inf
            best_speed_action = 0
            best_heading_action = 0
            for i_speed_action, speed_action in enumerate(SPEED_ACTIONS):
                
                # make acceleration predictions
                if speed_action == 0:
                    predicted_accelerations = acceleration_trajs[i_agent, :]
                else:
                    predicted_accelerations = np.copy(acceleration_trajs[i_agent, :])
                    add_action(predicted_accelerations, speed_action, i_time_step)
                
                for i_heading_action, heading_action in enumerate(HEADING_ACTIONS):

                    # make heading predictions
                    if heading_action == 0:
                        predicted_yaw_rates = yaw_rate_trajs[i_agent, :]
                    else:
                        predicted_yaw_rates = np.copy(yaw_rate_trajs[i_agent, :])
                        add_action(predicted_yaw_rates, heading_action, i_time_step)
                    
                    # get predicted state with this action combination
                    (pred_pos, pred_speed, pred_heading) = \
                        get_future_state(i_agent, predicted_accelerations, \
                            predicted_yaw_rates, i_time_step, PREDICTION_TIME_STEPS)    
                                
                    # get value of the predicted state
                    this_action_value = \
                        get_value_of_predicted_state(i_agent, i_time_step, \
                            pred_pos, pred_speed, pred_heading) \
                        - params["C_t"] * heading_action ** 2
                    #print(pred_pos, pred_speed, pred_heading, this_action_value)
                        
                    if this_action_value > best_value:
                        best_value = this_action_value
                        best_speed_action = speed_action
                        best_heading_action = heading_action
            
            # execute the selected actions
            if best_speed_action != 0:
                # print('%.2f s, agent %i (%.2f, %.2f): speed action %.2f' % \
                #     (time_stamp, i_agent, pos_trajs[i_agent, 0, i_time_step], \
                #     pos_trajs[i_agent, 1, i_time_step], best_speed_action))
                add_action(acceleration_trajs[i_agent, :], best_speed_action, i_time_step)
            if best_heading_action != 0:
                # print('%.2f s, agent %i (%.2f, %.2f): heading action %.2f' % \
                #     (time_stamp, i_agent, pos_trajs[i_agent, 0, i_time_step], \
                #     pos_trajs[i_agent, 1, i_time_step], best_heading_action * 180/math.pi))
                add_action(yaw_rate_trajs[i_agent, :], best_heading_action, i_time_step)




    
    # plotting

    
    delta_x_traj = pos_trajs[1, 0, :] - pos_trajs[0, 0, :]
    delta_y_traj = pos_trajs[1, 1, :] - pos_trajs[0, 1, :]
    distance_traj = np.sqrt(delta_x_traj ** 2 + delta_y_traj ** 2)
    is_colliding_traj = distance_traj < COLLISION_DISTANCE
    
    plt.figure('trajectories')
    for i_agent in range(N_AGENTS):
        plt.plot(pos_trajs[i_agent, 0, :], pos_trajs[i_agent, 1, :], \
            AGENT_COLORS[i_agent] + '.')
        plt.plot(pos_trajs[i_agent, 0, is_colliding_traj], \
            pos_trajs[i_agent, 1, is_colliding_traj], 'r.')
    plt.plot(pos_goals[:, 0], pos_goals[:, 1], 'g+')
    if N_OBSTACLES > 0:
        plt.plot(pos_obstacles[:, 0], pos_obstacles[:, 1], 'r^')
    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    plt.figure('time series')
    plt.subplot(3, 1, 1)
    for i_agent in range(N_AGENTS):
        plt.plot(TIME_STAMPS, speed_trajs[i_agent, :], AGENT_COLORS[i_agent] + '-')
    plt.ylabel('v (m/s)')
    plt.subplot(3, 1, 2)
    for i_agent in range(N_AGENTS):
        plt.plot(TIME_STAMPS, heading_trajs[i_agent, :] * 180 / math.pi, \
            AGENT_COLORS[i_agent] + '-')
    plt.ylabel('$\psi$ (deg)')
    plt.subplot(3, 1, 3)
    plt.plot(TIME_STAMPS, distance_traj, 'k-')
    plt.plot(TIME_STAMPS[is_colliding_traj], distance_traj[is_colliding_traj], 'r.')
    plt.ylabel('d (m)')
    plt.xlabel('t (s)')

    plt.show()


if __name__ == "__main__":
    pos_agents = np.array([[0, -5], [5, 0]])
    pos_goals = np.array([[0, 5], [-5, 0]])
    run_simulation(pos_agents, pos_goals)