import numpy as np
import matplotlib.pyplot as plt
import math



OBSTACLE_COLLISION_DISTANCE = 1

START_TIME = 0 # s
END_TIME = 10 # s
TIME_STEP = 0.1 # s
TIME_STAMPS = np.arange(START_TIME, END_TIME, TIME_STEP)

SPEED_ACTIONS = np.array([-1, -0.5, 0, 0.5, 1]) # m/s^2
HEADING_ACTIONS = np.array([-10, -5, 0, 5, 10]) * math.pi/180 # rad

STEERING_RATIO = 5

ACTION_DURATION = 0.5 # s
PREDICTION_TIME = ACTION_DURATION
ACTION_TIME_STEPS = math.ceil(ACTION_DURATION / TIME_STEP)
PREDICTION_TIME_STEPS = math.ceil(PREDICTION_TIME / TIME_STEP)

ACTIONS_VECTOR_LENGTH = TIME_STAMPS.size + PREDICTION_TIME_STEPS




def get_default_params():
    params = {
        "k_g": 5,
        "k_o": 5,
        "C_g": .1,
        "C_v": .1,
        "C_p": .1,
        "C_t": .1,
        "C_a": .1,
        "C_omega": .1
    }
    return params



def run_simulation(x_goal, x_obstacles, params = get_default_params()):

    
    # init
    accelerations = np.zeros(ACTIONS_VECTOR_LENGTH)
    sw_angles = np.zeros(ACTIONS_VECTOR_LENGTH)
    x_history = np.zeros((TIME_STAMPS.size, 2))
    speed_history = np.zeros(TIME_STAMPS.shape) 
    heading_history = np.zeros(TIME_STAMPS.shape)
    heading_history[0] = math.pi/4.01 # rad
    yaw_rate_history = np.zeros(TIME_STAMPS.shape)

    N_OBSTACLES = x_obstacles.shape[0]
    ttc_history = np.zeros((TIME_STAMPS.size, 2))


    # functions
    def advance_simulation(local_accelerations, local_sw_angles, 
                i_curr_time_step, n_time_steps_to_advance):
        
        local_x = (x_history[i_curr_time_step, :])
        local_speed = (speed_history[i_curr_time_step])
        local_heading = (heading_history[i_curr_time_step])
        
        i_final_time_step = i_curr_time_step + n_time_steps_to_advance

        for i in range(i_curr_time_step + 1, i_final_time_step + 1):
            local_speed += local_accelerations[i-1] * TIME_STEP
            if local_speed < 0:
                local_speed = 0
            local_yaw_rate = STEERING_RATIO * local_speed * local_sw_angles[i-1]
            local_heading += local_yaw_rate * TIME_STEP
            local_x = local_x + TIME_STEP * local_speed \
                * np.array([math.cos(local_heading), math.sin(local_heading)])
        
        return (local_x, local_speed, local_heading, \
            local_accelerations[i_final_time_step], local_yaw_rate)


    def add_action(state_vector, action_magnitude, i_action_time_step):
        state_vector[i_action_time_step:i_action_time_step+ACTION_TIME_STEPS] += \
            np.linspace(0, action_magnitude, ACTION_TIME_STEPS)
        state_vector[i_action_time_step+ACTION_TIME_STEPS:] += action_magnitude

    def get_time_to_collision(x, heading, speed, x_obstacle):
        if speed == 0:
            return math.inf
        vector_to_obstacle = x_obstacle - x
        distance_to_obstacle = np.linalg.norm(vector_to_obstacle)
        if distance_to_obstacle < OBSTACLE_COLLISION_DISTANCE:
            return 0.00001
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        heading_to_obstacle_vector = vector_to_obstacle / distance_to_obstacle
        heading_toward_obstacle_component = np.dot(heading_vector, heading_to_obstacle_vector)
        angle_to_obstacle_rel_heading = math.acos(heading_toward_obstacle_component)
        lateral_distance_to_obstacle = np.linalg.norm(vector_to_obstacle) * math.sin(angle_to_obstacle_rel_heading)
        if lateral_distance_to_obstacle < OBSTACLE_COLLISION_DISTANCE \
            and heading_toward_obstacle_component > 0:
            long_distance_to_obstacle = np.linalg.norm(vector_to_obstacle) \
                * math.cos(angle_to_obstacle_rel_heading) - OBSTACLE_COLLISION_DISTANCE
            return long_distance_to_obstacle / speed
        else:
            return math.inf
        
    def get_value_of_state(x, speed, heading, acc, yaw_rate, verbose = False):
        
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        
        vector_to_goal = x_goal - x


        heading_to_goal_vector = vector_to_goal / np.linalg.norm(vector_to_goal)
        
        heading_toward_goal_component = np.dot(heading_vector, heading_to_goal_vector)
        goal_distance_change_rate = -heading_toward_goal_component * speed

        dist_to_goal = np.linalg.norm(vector_to_goal)
        dist_to_passing_goal = heading_toward_goal_component * dist_to_goal # heading_toward_goal_component is also the cosine of the angle between heading_vector and heading_to_goal_vector

        #required_acc_to_stop_at_goal = -(speed ** 2 / (2 * dist_to_goal)) # accurate if heading toward goal
        required_acc_to_stop_at_goal = -(speed ** 2 / (2 * dist_to_passing_goal))
        
        value = -params["k_g"] * goal_distance_change_rate \
            - params["C_g"] * required_acc_to_stop_at_goal ** 2 \
            - params["C_v"] * speed ** 2 \
            - params["C_a"] * acc ** 2 \
            - params["C_omega"] * (yaw_rate * speed) ** 2
        
        for i_obstacle in range(N_OBSTACLES):
            time_to_obstacle_collision = \
                get_time_to_collision(x, heading, speed, x_obstacles[i_obstacle,:])
            if not math.isinf(time_to_obstacle_collision):
                value += -params["k_o"] / time_to_obstacle_collision
                #value += -params["k_o"] * (speed / time_to_obstacle_collision) ** 2
            
        
        if verbose:
            #print(angle_to_obstacle_rel_heading * 180 / math.pi)
            #print(lateral_distance_to_obstacle, time_to_obstacle_collision)
            print(heading_toward_goal_component)
        
        return value




    # run simulation
    for i_time_step, time_stamp in enumerate(TIME_STAMPS):
        
        #get_value_of_state(x, speed, heading, verbose = True)
        
        # do Euler step
        if i_time_step > 0:
            (x_history[i_time_step, :], speed_history[i_time_step], \
                heading_history[i_time_step], _, yaw_rate_history[i_time_step]) = \
                advance_simulation(accelerations, sw_angles, i_time_step-1, 1)


        # store time to collision info
        for i_obstacle in range(N_OBSTACLES):
            ttc_history[i_time_step, i_obstacle] = get_time_to_collision(\
                x_history[i_time_step, :], heading_history[i_time_step], \
                speed_history[i_time_step], x_obstacles[i_obstacle,:])
            
        
        # decide on action for this time step - by looping through the alternatives and getting predicted rewards
        best_value = -math.inf
        for i_speed_action, speed_action in enumerate(SPEED_ACTIONS):
            
            # make acceleration predictions
            if speed_action == 0:
                predicted_accelerations = accelerations
            else:
                predicted_accelerations = np.copy(accelerations)
                add_action(predicted_accelerations, speed_action, i_time_step)
            
            for i_heading_action, heading_action in enumerate(HEADING_ACTIONS):

                # make heading predictions
                if heading_action == 0:
                    predicted_sw_angles = sw_angles
                else:
                    predicted_sw_angles = np.copy(sw_angles)
                    add_action(predicted_sw_angles, heading_action, i_time_step)
                
                # get predicted state with this action combination
                (pred_x, pred_speed, pred_heading, pred_acc, pred_yaw_rate) = \
                    advance_simulation(predicted_accelerations, predicted_sw_angles, \
                            i_time_step, PREDICTION_TIME_STEPS)    
                            
                # get value of the predicted state
                this_action_value = get_value_of_state(pred_x, pred_speed, \
                    pred_heading, pred_acc, pred_yaw_rate) \
                    - params["C_p"] * speed_action ** 2 \
                    - params["C_t"] * heading_action ** 2
                #print(pred_x, pred_speed, pred_heading, this_action_value)
                    
                if this_action_value > best_value:
                    best_value = this_action_value
                    best_speed_action = speed_action
                    best_heading_action = heading_action
        
        # execute the selected actions
        if best_speed_action != 0:
            #print('%.2f s (%.2f, %.2f): speed action %.2f' % \
            #      (time_stamp, x_history[i_time_step, 0], x_history[i_time_step, 1], best_speed_action))
            add_action(accelerations, best_speed_action, i_time_step)
        if best_heading_action != 0:
            #print('%.2f s (%.2f, %.2f): heading action %.2f' % \
            #      (time_stamp, x_history[i_time_step, 0], x_history[i_time_step, 1], best_heading_action * 180/math.pi))
            add_action(sw_angles, best_heading_action, i_time_step)
        
        
                
    # traj_ax = plt.subplots()[1]
    # for i_obstacle in range(N_OBSTACLES):
    #     traj_ax.plot(x_obstacles[i_obstacle,0], x_obstacles[i_obstacle,1], 'r^')
    # traj_ax.plot(x_history[:, 0], x_history[:, 1], 'k.')
    # traj_ax.plot(x_goal[0], x_goal[1], 'g+')
    plt.figure()
    for i_obstacle in range(N_OBSTACLES):
        plt.plot(x_obstacles[i_obstacle,0], x_obstacles[i_obstacle,1], 'r^')
    plt.plot(x_history[:, 0], x_history[:, 1], 'k.')
    plt.plot(x_goal[0], x_goal[1], 'g+')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(TIME_STAMPS, accelerations[:len(TIME_STAMPS)], 'k-')
    plt.ylabel('a (m/s^2)')
    plt.subplot(5, 1, 2)
    plt.plot(TIME_STAMPS, speed_history, 'k-')
    plt.ylabel('v (m/s)')
    plt.subplot(5, 1, 3)
    plt.plot(TIME_STAMPS, sw_angles[:len(TIME_STAMPS)] * 180 / math.pi, 'k-')
    plt.ylabel('$\delta$ (deg)')
    plt.subplot(5, 1, 4)
    plt.plot(TIME_STAMPS, yaw_rate_history * 180 / math.pi, 'k-')
    plt.ylabel('$\omega$ (deg/s)')
    plt.subplot(5, 1, 5)
    plt.plot(TIME_STAMPS, heading_history * 180 / math.pi, 'k-')
    plt.ylabel('$\psi$ (deg)')
    plt.xlabel('t (s)')

    plt.figure()
    for i_obstacle in range(N_OBSTACLES):
        plt.subplot(N_OBSTACLES, 1, i_obstacle+1)
        plt.plot(TIME_STAMPS, ttc_history[:, i_obstacle], 'k-')
        plt.ylabel('TTC (s)')
        plt.ylim((0, 10))
        plt.xlim((TIME_STAMPS[0], TIME_STAMPS[-1]))
    plt.xlabel('t (s)')

    plt.show()


if __name__ == "__main__":

    x_goal = np.array([10, 10])
    x_obstacles = np.array([[5, 4.5], [7, 9]])
    run_simulation(x_goal, x_obstacles)