import numpy as np
import matplotlib.pyplot as plt
import math


class Parameters:
    pass

class KinematicState:
    def __init__(self, pos = np.zeros(2), \
        long_speed = 0, long_acc = 0, yaw_angle = 0, yaw_rate = 0):
        self.pos = pos
        self.long_speed = long_speed
        self.long_acc = long_acc
        self.yaw_angle = yaw_angle
        self.yaw_rate = yaw_rate

class Trajectory:
    def __init__(self, simulation, initial_state = KinematicState()):
        # allocate and initialise numpy arrays for action variables
        self.long_acc = initial_state.long_acc \
            * np.ones(simulation.settings.n_time_steps)
        self.yaw_rate = initial_state.yaw_rate \
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

    # these are all to be overridden and implemented by descendant classes
    def get_default_parameters(self):
        """Return a Parameters object with default parameter values."""
        empty_parameters = Parameters()
        return empty_parameters
    def prepare_for_action_update(self):
        """Make any preparations needed by this agent and/or other agents in 
        the simulation, before making the action update. 
        """
        pass
    def do_action_update(self):
        """Do the action update for the agent. 
        """
        #print('Updating agent %s to time step %i' % (self.name, self.simulation.state.i_time_step))
        pass


    # def prepare_for_simulation(self):
    #     self.other_agents = self.simulation.get_other_agents(self)


    def get_state(self, i_time_step):
        state = KinematicState(pos = self.trajectory.pos[:, i_time_step], \
            long_speed = self.trajectory.long_speed[i_time_step], \
            long_acc = self.trajectory.long_acc[i_time_step], \
            yaw_angle = self.trajectory.yaw_angle[i_time_step], \
            yaw_rate = self.trajectory.yaw_rate[i_time_step])
        return state

    def get_current_state(self):
        state = self.get_state(self.simulation.state.i_time_step)
        return state


    def get_future_state(self, local_long_accs, local_yaw_rates, 
        i_start_time_step, n_time_steps_to_advance):
        """Use the supplied action vectors, simulate the specified number of
        time steps ahead from the agent's state at the specified starting time 
        step, and return the resulting agent state.
        """
        
        local_pos = self.trajectory.pos[:, i_start_time_step]
        local_long_speed = self.trajectory.long_speed[i_start_time_step]
        local_yaw_angle = self.trajectory.yaw_angle[i_start_time_step]
        
        i_end_time_step = i_start_time_step + n_time_steps_to_advance
        for i in range(i_start_time_step + 1, i_end_time_step + 1):
            local_long_speed += \
                local_long_accs[i-1] * self.simulation.settings.time_step
            if local_long_speed < 0:
                local_long_speed = 0
            local_yaw_angle += \
                local_yaw_rates[i-1] * self.simulation.settings.time_step
            local_pos = local_pos \
                + self.simulation.settings.time_step * local_long_speed \
                * np.array([math.cos(local_yaw_angle), math.sin(local_yaw_angle)])

        return (local_pos, local_long_speed, local_yaw_angle, \
            local_long_accs[i_end_time_step])


    def do_kinematics_update(self):
        i_time_step = self.simulation.state.i_time_step
        (self.trajectory.pos[:, i_time_step], \
            self.trajectory.long_speed[i_time_step], \
            self.trajectory.yaw_angle[i_time_step], _) = \
            self.get_future_state( \
                self.trajectory.long_acc, self.trajectory.yaw_rate, \
                i_time_step-1, 1)

    def plot_trajectory(self):
        plt.plot(self.trajectory.pos[0, ::3], self.trajectory.pos[1, ::3], \
            self.plot_color + '.')
                
    def plot_time_series(self, speed_ax, yaw_ax, long_acc_ax):
        speed_ax.plot(self.simulation.time_stamps, self.trajectory.long_speed, \
            self.plot_color + '-')
        yaw_ax.plot(self.simulation.time_stamps, self.trajectory.yaw_angle * 180/math.pi, \
            self.plot_color + '-')
        long_acc_ax.plot(self.simulation.time_stamps, self.trajectory.long_acc, \
            self.plot_color + '-')

    def __init__(self, name, simulation, initial_state, plot_color = 'k'):
        # store agent name and plot color
        self.name = name
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
            if initial_state.long_acc is None:
                initial_state.long_acc = 0
            if initial_state.yaw_angle is None:
                initial_state.long_acc = 0
            if initial_state.yaw_rate is None:
                initial_state.yaw_rate = 0
        # create a Trajectory object for the agent
        self.trajectory = Trajectory(simulation, initial_state)


class AgentWithGoal(BaseAgent):

    def plot_trajectory(self):
        super().plot_trajectory()
        plt.plot(self.goal[0], self.goal[1], 'g+')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')


    def __init__(self, name, simulation, initial_state, goal, plot_color = 'k'):
        # check if the caller has provided start and goal positions, but no
        # starting yaw angle - if so set it to point at goal
        if (initial_state is not None) and (initial_state.pos is not None) \
            and (goal is not None) and (initial_state.yaw_angle) is None:
            agent_to_goal_vector = goal - initial_state.pos
            initial_state.yaw_angle = \
                np.arctan2(agent_to_goal_vector[1], agent_to_goal_vector[0])
        # parse and store the agent's goal position
        if goal is None:
            goal = np.zeros(2)
        self.goal = goal
        # run ancestor initialisation
        super().__init__(name, simulation, initial_state, plot_color)


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

    def add_agent(self, agent):
        self.agents.append(agent)
        return len(self.agents)-1

    # def get_other_agents(self, calling_agent):
    #     other_agents = []
    #     for agent in self.agents:
    #         if agent is not calling_agent:
    #             other_agents.append(agent)
    
    def plot_trajectories(self):
        plt.figure(figsize = [4, 3])
        for agent in self.agents:
            agent.plot_trajectory()
        plt.axis('equal')
        plt.tight_layout()

    def plot_time_series(self):

        fig, (speed_ax, long_acc_ax, pos_ax, dist_ax) = \
            plt.subplots(4, 1, sharex = True, figsize = [4, 4])

        for agent in self.agents:
            agent.plot_time_series(speed_ax, pos_ax, long_acc_ax)
        speed_ax.set_ylabel('v (m/s)')
        long_acc_ax.set_ylabel('a (m/s^2)')

        pos_ax.cla()
        pos_ax.plot(self.time_stamps[[0, -1]], np.array((0, 0)), 'k--')
        pos_ax.plot(self.time_stamps, -self.agents[0].trajectory.pos[1,:], \
            self.agents[0].plot_color + '-')
        pos_ax.plot(self.time_stamps, self.agents[1].trajectory.pos[0,:], \
            self.agents[1].plot_color + '-')        
        pos_ax.set_ylim(-10, 10)
        pos_ax.set_ylabel('$d_{CP}$ (m)')
        
        agent_dist_vectors = self.agents[1].trajectory.pos - self.agents[0].trajectory.pos
        agent_distances = np.linalg.norm(agent_dist_vectors, axis = 0)
        dist_ax.plot(self.time_stamps, agent_distances, 'k-')
        colliding_time_stamps = agent_distances < self.shared_params.d_C
        dist_ax.plot(self.time_stamps[colliding_time_stamps], agent_distances[colliding_time_stamps], 'r-')
        dist_ax.plot(self.time_stamps[[0, -1]], self.shared_params.d_C * np.array((1, 1)), 'r--')
        dist_ax.set_ylim(-1, 10)
        dist_ax.set_ylabel('$d$ (m)')
        dist_ax.set_xlabel('t (s)')

        plt.tight_layout()

    def run(self):

        # for agent in self.agents:
        #     agent.prepare_for_simulation()

        for i_time_step in range(self.settings.n_time_steps):
            self.state.set_time_step(i_time_step)
            if i_time_step > 0:
                for agent in self.agents:
                    agent.do_kinematics_update()
            for agent in self.agents:
                agent.prepare_for_action_update()
            for agent in self.agents:
                agent.do_action_update()

    
    def __init__(self, start_time, end_time, time_step):
        self.time_stamps = np.arange(start_time, end_time, time_step)
        self.settings = SimulationSettings(self.time_stamps, time_step) 
        self.state = SimulationState(self)
        self.agents = []

    

if __name__ == "__main__":
    test_simulation = Simulation(0, 10, 0.1)
    agent_A = BaseAgent( 'A', test_simulation, initial_state = \
        KinematicState( pos = np.array((-5, -5)), long_speed = 1, yaw_rate = 0.1 ) )
    agent_B = BaseAgent( 'B', test_simulation, initial_state = \
        KinematicState( pos = np.array((5, 5)), long_acc = 1 ) )
    test_simulation.run()
    print(agent_A.trajectory.pos)
    print(agent_B.trajectory.pos)
    test_simulation.plot_trajectories()
    plt.show()

    
