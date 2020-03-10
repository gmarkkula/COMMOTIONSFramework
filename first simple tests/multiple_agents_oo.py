import math
import numpy as np
import matplotlib.pyplot as plt
import commotions


class SimpleAgent(commotions.AgentWithGoal):

    def get_default_parameters(self):
        """Return a Parameters object with default parameter values."""
        params = commotions.Parameters()
        params.k_g = 1
        params.k_c = 0.3
        params.C_v = 0.3
        params.C_t = 0.3
        params.DeltaT = 0.3
        params.deltavs = np.array([-1, -0.5, 0, 0.5, 1]) # m/s
        params.deltapsis = np.array([-10, -5, 0, 5, 10]) * math.pi/180 # rad/s
        return params
    
    def get_default_shared_parameters(self):
        params = commotions.Parameters()
        params.T_P = 0.3 # prediction time
        params.d_C = 1 # collision distance
        return params

    def prepare_for_action_update(self):
        """Make any preparations needed by this agent and/or other agents in 
        the simulation, before making the action update. 
        """
        # simulate the agent a prediction interval ahead, with constant 
        # speeds
        state = self.get_current_state()
        heading_vector = \
            np.array([math.cos(state.yaw_angle), math.sin(state.yaw_angle)])
        self.predicted_state = state
        self.predicted_state.pos = self.predicted_state.pos \
            + state.long_speed * self.simulation.shared_params.T_P \
            * heading_vector
        
    def do_action_update(self):
        """Do the action update for the agent. 
        """
        # decide on an action for this agent at current time step - by looping 
        # through the alternatives and calculating predicted rewards
        i_time_step = self.simulation.state.i_time_step
        best_value = -math.inf
        best_speed_action = 0
        best_heading_action = 0
        for speed_action in self.params.deltavs:
            
            # make acceleration predictions
            if speed_action == 0:
                pred_long_accs = self.action_long_accs
            else:
                pred_long_accs = np.copy(self.action_long_accs)
                self.add_action(pred_long_accs, speed_action, i_time_step)
            
            for heading_action in self.params.deltapsis:

                # make heading predictions
                if heading_action == 0:
                    pred_yaw_rates = self.action_yaw_rates
                else:
                    pred_yaw_rates = np.copy(self.action_yaw_rates)
                    self.add_action(pred_yaw_rates, heading_action, i_time_step)
                
                # get predicted state with this action combination
                (pred_pos, pred_speed, pred_yaw_angle) = \
                    self.get_future_state(pred_long_accs, \
                        pred_yaw_rates, i_time_step, \
                        self.simulation.prediction_time_steps)   

                # get value of the predicted state
                this_action_value = self.get_value_of_predicted_state( \
                    pred_pos, pred_speed, pred_yaw_angle) \
                    - self.params.C_t * heading_action ** 2
                #print(pred_pos, pred_speed, pred_heading, this_action_value)
                    
                if this_action_value > best_value:
                    best_value = this_action_value
                    best_speed_action = speed_action
                    best_heading_action = heading_action
        
        # execute the selected actions (add future longaccs/yawrates)
        if best_speed_action != 0:
            # print('%.2f s, agent %i (%.2f, %.2f): speed action %.2f' % \
            #     (time_stamp, i_agent, pos_trajs[i_agent, 0, i_time_step], \
            #     pos_trajs[i_agent, 1, i_time_step], best_speed_action))
            self.add_action(self.action_long_accs, best_speed_action, i_time_step)
        if best_heading_action != 0:
            # print('%.2f s, agent %i (%.2f, %.2f): heading action %.2f' % \
            #     (time_stamp, i_agent, pos_trajs[i_agent, 0, i_time_step], \
            #     pos_trajs[i_agent, 1, i_time_step], best_heading_action * 180/math.pi))
            self.add_action(self.action_yaw_rates, best_heading_action, i_time_step)
        
        # copy over action long. accelerations and yaw rates to actual agent 
        # trajectory
        self.trajectory.long_acc[i_time_step] = \
            self.action_long_accs[i_time_step]
        self.trajectory.yaw_rate[i_time_step] = \
            self.action_yaw_rates[i_time_step]
    

    def add_action(self, state_vector, action_magnitude, i_action_time_step):
        state_vector[i_action_time_step : i_action_time_step + self.action_time_steps] += \
            action_magnitude / self.params.DeltaT


    def get_real_quadratic_roots(self, a, b, c):
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
        self, pos, speed, heading, agent):
        # get some basics
        other_pos = agent.predicted_state.pos
        other_speed = agent.predicted_state.long_speed
        other_heading = agent.predicted_state.yaw_angle
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
            self.get_real_quadratic_roots(a, b, \
                c - self.simulation.shared_params.d_C ** 2)
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

    
    def get_value_of_predicted_state(self, pos, speed, heading, verbose = False):
        
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        
        # reward for progress toward goal and cost for speed
        vector_to_goal = self.goal - pos
        heading_to_goal_vector = vector_to_goal / np.linalg.norm(vector_to_goal)
        heading_toward_goal_component = np.dot(heading_vector, heading_to_goal_vector)
        goal_distance_change_rate = -heading_toward_goal_component * speed
        value = -self.params.k_g * goal_distance_change_rate \
            - self.params.C_v * speed ** 2
        
        # # cost for being on collision course with obstacles
        # for i_obstacle in range(N_OBSTACLES):
        #     vector_to_obstacle = x_obstacles[i_obstacle,:] - x
        #     heading_to_obstacle_vector = \
        #         vector_to_obstacle / np.linalg.norm(vector_to_obstacle)
        #     heading_toward_obstacle_component = \
        #         np.dot(heading_vector, heading_to_obstacle_vector)
        #     angle_to_obstacle_rel_heading = \
        #         math.acos(heading_toward_obstacle_component)
        #     lateral_distance_to_obstacle = \
        #         np.linalg.norm(vector_to_obstacle) * math.sin(angle_to_obstacle_rel_heading)
        #     if lateral_distance_to_obstacle < COLLISION_DISTANCE \
        #         and heading_toward_obstacle_component > 0:
        #         long_distance_to_obstacle = np.linalg.norm(vector_to_obstacle) \
        #             * math.cos(angle_to_obstacle_rel_heading)
        #         time_to_obstacle_collision = long_distance_to_obstacle / speed
        #         value += -params["k_c"] / time_to_obstacle_collision   

        # cost for being on collision course with other agents
        for agent in self.simulation.agents:
            if agent is self:
                continue
            # calculate future time at which agents come within collision
            # distance of each other (if at all)
            time_to_agent_collision = \
                self.get_time_to_agent_collision_from_pred_state(\
                pos, speed, heading, agent)
            if time_to_agent_collision == 0:
                value = -math.inf
            elif time_to_agent_collision < math.inf:
                value += -self.params.k_c / time_to_agent_collision  
        
        if verbose:
            #print(angle_to_obstacle_rel_heading * 180 / math.pi)
            #print(lateral_distance_to_obstacle, time_to_obstacle_collision)
            print(heading_toward_goal_component)
        
        return value




    def __init__(self, name, simulation, initial_state, goal, \
        params = None, shared_params = None, plot_color = 'k'): # rad
        # call ancestor class init
        super().__init__(name, simulation, initial_state, goal, plot_color)
        # set default parameters if not supplied by caller
        if params is None:
            params = self.get_default_parameters()
        if shared_params is None:
            shared_params = self.get_default_shared_parameters()
        # store parameters
        self.params = params
        # store parameters shared across all agents of this type - in 
        # the simulation object
        self.simulation.shared_params = shared_params
        self.simulation.prediction_time_steps = \
            math.ceil(shared_params.T_P / self.simulation.settings.time_step)
        # store some derived constants
        self.action_time_steps = math.ceil(
            self.params.DeltaT / self.simulation.settings.time_step)
        self.actions_vector_length = \
            self.simulation.settings.n_time_steps + \
            self.simulation.prediction_time_steps
        # prepare vectors for storing long acc and yaw rates, incl lookahead
        # with added actions
        self.action_long_accs = initial_state.long_acc \
            * np.ones(self.actions_vector_length)
        self.action_yaw_rates = initial_state.yaw_rate \
            * np.ones(self.actions_vector_length)


if __name__ == "__main__":
    test_simulation = commotions.Simulation(0, 10, 0.1)
    agent_A = SimpleAgent( 'A', test_simulation, initial_state = \
        commotions.KinematicState( pos = np.array((0, -5.5)), yaw_angle = None ), \
        goal = np.array((0, 5)), \
        plot_color = 'c' )
    agent_B = SimpleAgent( 'B', test_simulation, initial_state = \
        commotions.KinematicState( pos = np.array((5, 0)) ), \
        goal = np.array((-5, 0)), \
        plot_color = 'm' )
    test_simulation.run()
    print(agent_A.trajectory.pos)
    print(agent_B.trajectory.pos)    
    test_simulation.plot_trajectories()
    plt.show()
