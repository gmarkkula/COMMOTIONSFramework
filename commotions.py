# POSSIBLE TODOs:
# - add class AgentWithIntermittentActions(BaseAgent) that is "cooperative" in 
#   its __init__ method with AgentWithGoal
#   see https://rhettinger.wordpress.com/2011/05/26/super-considered-super/



import numpy as np
import matplotlib.pyplot as plt
import math
import copy


def add_uniform_action_to_array(\
    action_array, action_magnitude, n_action_time_steps, i_start_time_step):
    """ Add the value action_magnitude to action_array over n_action_time_steps
        starting at i_start_time_step.
    """
    action_array[i_start_time_step : i_start_time_step + n_action_time_steps] += \
        action_magnitude


def add_linear_ramp_action_to_array(action_array, action_magnitude, \
    n_action_time_steps, i_start_time_step):
    """ Add a ramp from zero to action_magnitude to action_array over
        n_action_time_steps starting at i_start_time_step, and then adds
        action_magnitude to the rest of action_array.
    """
    action_array[i_start_time_step : i_start_time_step + n_action_time_steps] += \
        np.linspace(0, action_magnitude, n_action_time_steps)
    action_array[i_start_time_step + n_action_time_steps : ] += action_magnitude


def get_future_kinematic_state(initial_state, long_acc, yaw_rate, \
    time_step, n_time_steps_to_advance, i_start_time_step = 0, \
    can_reverse = True, as_tuple = False):
    
    # handle vector vs scalar input for the action variables
    if type(long_acc) is np.ndarray:
        get_long_acc = lambda i : long_acc[i]
    else:
        get_long_acc = lambda i : long_acc
    if type(yaw_rate) is np.ndarray:
        get_yaw_rate = lambda i : yaw_rate[i]
    else:
        get_yaw_rate = lambda i : yaw_rate

    # copy initial state into a new KinematicState object
    local_state = copy.copy(initial_state)

    # do the Euler time-stepping
    for i in range(i_start_time_step + 1, 
                    i_start_time_step + n_time_steps_to_advance + 1):
        if can_reverse:
            min_speed = -math.inf
        else:
            min_speed = 0
        # a more exact version of the position update below should also take into
        # account the effect of non-zero yaw rate
        local_state.pos = (
            local_state.pos
            + max(min_speed, time_step * local_state.long_speed 
                + get_long_acc(i-1) * (time_step ** 2) / 2)
            * np.array([math.cos(local_state.yaw_angle), 
                        math.sin(local_state.yaw_angle)]))
        local_state.long_speed += get_long_acc(i-1) * time_step
        local_state.long_speed = max(min_speed, local_state.long_speed)
        local_state.yaw_angle += get_yaw_rate(i-1) * time_step

    # return the final state, as tuple or as KinematicState object
    if as_tuple:
        return (local_state.pos, local_state.long_speed, local_state.yaw_angle)
    else:
        return local_state


def get_intersection_of_lines(line1_pointA, line1_pointB, line2_pointA, line2_pointB):
    # get vectors from points A to B
    line1_vector = line1_pointB - line1_pointA
    line2_vector = line2_pointB - line2_pointA
    # get intermediate variables (just for ease of reading the expressions further below)
    x_1 = line1_pointA[0]
    y_1 = line1_pointA[1]
    x_2 = line2_pointA[0]
    y_2 = line2_pointA[1]
    Deltax_1 = line1_vector[0]
    Deltay_1 = line1_vector[1]
    Deltax_2 = line2_vector[0]
    Deltay_2 = line2_vector[1]
    # calculate how many of line1_vector is needed to reach the intersection
    denominator = Deltax_2 * Deltay_1 - Deltax_1 * Deltay_2
    if denominator == 0:
        # the lines don't intersect
        return None
    else:
        numerator = Deltax_2 * (y_2 - y_1) - Deltay_2 * (x_2 - x_1)
        t = numerator / denominator
    # get and return the intersection point
    return line1_pointA + t * line1_vector



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


def get_time_to_agent_collision(state1, state2, collision_distance):
    # collision already happening?
    if np.linalg.norm(state2.pos - state1.pos) <= collision_distance:
        return 0
    # get some basics
    delta_x = state2.pos[0] - state1.pos[0]
    delta_y = state2.pos[1] - state1.pos[1]
    delta_v_x = state2.long_speed * math.cos(state2.yaw_angle) \
        - state1.long_speed * math.cos(state1.yaw_angle)
    delta_v_y = state2.long_speed * math.sin(state2.yaw_angle) \
        - state1.long_speed * math.sin(state1.yaw_angle)
    # get coefficients of quadratic equation for squared distance
    # D^2 = at^2 + bt + c 
    a = delta_v_x ** 2 + delta_v_y ** 2
    b = 2 * (delta_x * delta_v_x + delta_y * delta_v_y)
    c = delta_x ** 2 + delta_y ** 2
    # get roots t for D^2 = D_collision^2 <=> D^2 - D_collision^2 = 0
    coll_times = \
        get_real_quadratic_roots(a, b, c - collision_distance ** 2)
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
            # one in future, one in past - i.e., collision ongoing now
            return 0


class Parameters:
    pass

class ActionState:
    def __init__(self, long_acc = 0, yaw_rate = 0):
        self.long_acc = long_acc
        self.yaw_rate = yaw_rate

class KinematicState:
    def __init__(self, pos = np.zeros(2), \
        long_speed = 0, yaw_angle = 0):
        self.pos = pos
        self.long_speed = long_speed
        self.yaw_angle = yaw_angle

class Trajectory:
    def __init__(self, simulation, initial_state = KinematicState(), \
        initial_action_state = ActionState()):
        # allocate and initialise numpy arrays for action variables
        self.long_acc = initial_action_state.long_acc \
            * np.ones(simulation.settings.n_time_steps)
        self.yaw_rate = initial_action_state.yaw_rate \
            * np.ones(simulation.settings.n_time_steps)
        # allocate numpy arrays for variables to be simulated
        self.pos = math.nan * np.ones((2, simulation.settings.n_time_steps))
        self.long_speed = math.nan * np.ones(simulation.settings.n_time_steps)
        self.yaw_angle = math.nan * np.ones(simulation.settings.n_time_steps)
        # set initial state for variables to be simulated
        self.pos[:, 0] = initial_state.pos
        self.long_speed[0] = initial_state.long_speed
        self.yaw_angle[0] = initial_state.yaw_angle


class BaseAgent:

    # these can all to be overridden and implemented by descendant classes
    def prepare_for_simulation(self):
        """Make any preparations needed before the actual simulation is started.
        """
        pass
    def prepare_for_action_update(self):
        """Make any preparations needed by this agent and/or other agents in 
        the simulation, before making the action update for the current 
        simulation time step. 
        """
        pass
    def do_action_update(self):
        """Do the action update for the agent for the current simulation time step. 
        """
        pass


    def get_kinematic_state(self, i_time_step):
        state = KinematicState(pos = self.trajectory.pos[:, i_time_step], \
            long_speed = self.trajectory.long_speed[i_time_step], \
            yaw_angle = self.trajectory.yaw_angle[i_time_step])
        return state


    def get_current_kinematic_state(self):
        state = self.get_kinematic_state(self.simulation.state.i_time_step)
        return state


    def get_future_kinematic_state(self, long_acc, yaw_rate, 
        n_time_steps_to_advance, i_start_time_step = None, as_tuple = False):
        """Use the supplied action vectors, simulate the specified number of
        time steps ahead from the agent's state at the specified starting time 
        step, and return the resulting agent state. Setting i_start_time_step
        None simulates from the current simulation time step.
        """

        if i_start_time_step is None:
            i_start_time_step = self.simulation.state.i_time_step
        
        initial_state = self.get_kinematic_state(i_start_time_step)

        # call the commotions.get_future_kinematic_state helper function
        return get_future_kinematic_state(\
            initial_state, long_acc, yaw_rate, \
            self.simulation.settings.time_step, n_time_steps_to_advance, \
            i_start_time_step, self.can_reverse, as_tuple)


    def do_kinematics_update(self):
        i_time_step = self.simulation.state.i_time_step
        (self.trajectory.pos[:, i_time_step], \
            self.trajectory.long_speed[i_time_step], \
            self.trajectory.yaw_angle[i_time_step]) = \
            self.get_future_kinematic_state( \
                self.trajectory.long_acc, self.trajectory.yaw_rate, \
                1, i_time_step-1, as_tuple = True)


    def plot_trajectory(self):
        return plt.plot(self.trajectory.pos[0, :], self.trajectory.pos[1, :], \
            self.plot_color + '.', label = self.name)
                

    def __init__(self, name, simulation, initial_state, \
        initial_action_state = ActionState(), can_reverse = True, plot_color = 'k'):
        # store some basic info about the agent
        self.name = name
        self.can_reverse = can_reverse
        self.plot_color = plot_color
        # assign agent to simulation
        self.i_agent = simulation.add_agent(self)
        self.simulation = simulation
        # parse the provided initial state, if any
        if initial_state is None:
            initial_state = KinematicState() # all zeros
        else:
            if initial_state.pos is None:
                initial_state.pos = np.zeros(2)
            if initial_state.long_speed is None:
                initial_state.long_speed = 0
            if initial_state.yaw_angle is None:
                initial_state.yaw_angle = 0
        # create a Trajectory object for the agent
        self.trajectory = \
            Trajectory(simulation, initial_state, initial_action_state)


class AgentWithGoal(BaseAgent):

    def plot_trajectory(self):
        super().plot_trajectory()
        plt.plot(self.goal[0], self.goal[1], 'g+')


    def __init__(self, name, simulation, goal, initial_kinematic_state, \
        initial_action_state = ActionState(), can_reverse = True, plot_color = 'k'):
        # check if the caller has provided start and goal positions, but no
        # starting yaw angle - if so set it to point at goal
        if (initial_kinematic_state is not None) \
            and (initial_kinematic_state.pos is not None) \
            and (goal is not None) \
            and (initial_kinematic_state.yaw_angle is None):
            agent_to_goal_vector = goal - initial_kinematic_state.pos
            initial_kinematic_state.yaw_angle = \
                np.arctan2(agent_to_goal_vector[1], agent_to_goal_vector[0])
        # parse and store the agent's goal position
        if goal is None:
            goal = np.zeros(2)
        self.goal = goal
        # run ancestor initialisation
        super().__init__(name, simulation, initial_kinematic_state, \
            initial_action_state, can_reverse, plot_color)


class SimulationSettings:
    def __init__(self, time_stamps, time_step):
        self.start_time = time_stamps[0]
        self.end_time = time_stamps[-1]
        self.time_step = time_step
        self.n_time_steps = len(time_stamps)

class SimulationState:
    def __init__(self, simulation):
        self.simulation = simulation
        self.i_time_step = 0
        self.time = simulation.time_stamps[0]
    def set_time_step(self, i_time_step):
        self.i_time_step = i_time_step
        self.time = self.simulation.time_stamps[i_time_step]


class Simulation:
    
    def after_time_step(self):
        """
        Called at the end of each time step in the simulation loop in the 
        run() method. Can be overridden by descendant classes to do any 
        post-time-step processing, for example to set self.stop_now if
        simulation should be terminated.

        """
        pass
    
    def after_simulation(self):
        """
        Called at the end of the run() method. Can be overridden by descendant 
        classes to do any post-simulation processing.

        """
        pass

    def add_agent(self, agent):
        self.agents.append(agent)
        return len(self.agents)-1
    
    def plot_trajectories(self):
        for agent in self.agents:
            agent.plot_trajectory()
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

    def run(self):

        for agent in self.agents:
            agent.prepare_for_simulation()

        self.stop_now = False
        for i_time_step in range(self.settings.n_time_steps):
            self.state.set_time_step(i_time_step)
            if i_time_step > 0:
                for agent in self.agents:
                    agent.do_kinematics_update()
            for agent in self.agents:
                agent.prepare_for_action_update()
            for agent in self.agents:
                agent.do_action_update()
            self.after_time_step()
            if self.stop_now:
                break
            
        self.actual_end_time = self.state.time
            
        self.after_simulation()

    
    def __init__(self, start_time, end_time, time_step):
        self.time_stamps = np.arange(start_time, end_time, time_step)
        self.settings = SimulationSettings(self.time_stamps, time_step) 
        self.state = SimulationState(self)
        self.agents = []

    

if __name__ == "__main__":
    test_simulation = Simulation(0, 10, 0.1)
    agent_A = BaseAgent( 'A', test_simulation, \
        initial_state = KinematicState( pos = np.array((-5, -5)), long_speed = 1 ), \
        initial_action_state = ActionState( yaw_rate = 0.1 ) )
    agent_B = BaseAgent( 'B', test_simulation, \
        initial_state = KinematicState( pos = np.array((5, 5)) ), \
        initial_action_state = ActionState( long_acc = 1 ) )
    test_simulation.run()
    print(agent_A.trajectory.pos)
    print(agent_B.trajectory.pos)
    test_simulation.plot_trajectories()


    # test line intersection function
    plt.figure()
    # example 1
    line1_ptA = np.array((0, 0))
    line1_ptB = np.array((1, 0))
    line2_ptA = np.array((0.5, -1))
    line2_ptB = np.array((0.5, 1))
    pos = get_intersection_of_lines(line1_ptA, line1_ptB, line2_ptA, line2_ptB)
    plt.plot((line1_ptA[0], line1_ptB[0]), (line1_ptA[1], line1_ptB[1]), 'k')
    plt.plot((line2_ptA[0], line2_ptB[0]), (line2_ptA[1], line2_ptB[1]), 'k')
    plt.plot(pos[0], pos[1], 'r*')
    # example 2
    line1_ptA = np.array((1, 2))
    line1_ptB = np.array((2, 3))
    line2_ptA = np.array((1.5, 0))
    line2_ptB = np.array((1.7, 4))
    pos = get_intersection_of_lines(line1_ptA, line1_ptB, line2_ptA, line2_ptB)
    plt.plot((line1_ptA[0], line1_ptB[0]), (line1_ptA[1], line1_ptB[1]), 'k')
    plt.plot((line2_ptA[0], line2_ptB[0]), (line2_ptA[1], line2_ptB[1]), 'k')
    plt.plot(pos[0], pos[1], 'r*')
    # non-intersecting
    line1_ptA = np.array((-1, 0))
    line1_ptB = np.array((-1, 1))
    line2_ptA = np.array((-2, 0))
    line2_ptB = np.array((-2, 1))
    pos = get_intersection_of_lines(line1_ptA, line1_ptB, line2_ptA, line2_ptB)
    plt.plot((line1_ptA[0], line1_ptB[0]), (line1_ptA[1], line1_ptB[1]), 'k')
    plt.plot((line2_ptA[0], line2_ptB[0]), (line2_ptA[1], line2_ptB[1]), 'k')
    if pos is None:
        print('No intersection')
    else:
        plt.plot(pos[0], pos[1], 'r*')

    plt.show()

    
