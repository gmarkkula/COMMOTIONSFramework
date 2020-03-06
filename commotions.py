import numpy as np
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
        # allocate numpy arrays
        self.pos = math.nan * np.ones((2, simulation.settings.n_time_steps))
        self.long_speed = math.nan * np.ones(simulation.settings.n_time_steps)
        self.long_acc = math.nan * np.ones(simulation.settings.n_time_steps)
        self.yaw_angle = math.nan * np.ones(simulation.settings.n_time_steps)
        self.yaw_rate = math.nan * np.ones(simulation.settings.n_time_steps)
        # set initial state
        self.pos[:, 0] = initial_state.pos
        self.long_speed[0] = initial_state.long_speed
        self.long_acc[0] = initial_state.long_acc
        self.yaw_angle[0] = initial_state.yaw_angle
        self.yaw_rate[0] = initial_state.yaw_rate


class BaseAgent:

    # these are all to be overridden and implemented by descendant classes
    def get_default_parameters(self):
        """Return a Parameters object with default parameter values."""
        empty_parameters = Parameters()
        return empty_parameters
    def determine_update(self):
        """Prepare for taking a simulation time step, without actually making 
        the update. Overriding methods should end by a call to this method, using
        BaseAgent.determine_update(self)
        """
        #print('Preparing agent %s for time step %i' % (self.name, self.simulation.state.i_time_step))
        self.i_determined_update_time_step = self.simulation.state.i_time_step
    def apply_update(self):
        """Apply a previously prepared update to the agent. Overriding methods
        should start by a call to this method, using 
        BaseAgent.apply_update(self).
        """
        #print('Updating agent %s to time step %i' % (self.name, self.simulation.state.i_time_step))
        if self.i_determined_update_time_step != \
            self.simulation.state.i_time_step:
            raise Exception('Agent object not prepared for simulation update.')
    

    def __init__(self, name, simulation, initial_state = KinematicState()):
        self.name = name
        simulation.add_agent(self)
        self.simulation = simulation
        self.parameters = self.get_default_parameters()
        self.trajectory = Trajectory(simulation, initial_state)
        self.i_determined_update_time_step = None



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
    def advance(self):
        self.i_time_step += 1
        self.time = self.simulation.time_stamps[self.i_time_step]

class Simulation:

    def add_agent(self, agent):
        self.agents.append(agent)
    
    def run(self):

        for i_time_step in range(self.settings.n_time_steps-1):
            self.state.advance()
            for agent in self.agents:
                agent.determine_update()
            for agent in self.agents:
                agent.apply_update()
    
    def __init__(self, start_time, end_time, time_step):
        self.time_stamps = np.arange(start_time, end_time, time_step)
        self.settings = SimulationSettings(self.time_stamps, time_step) 
        self.state = SimulationState(self)
        self.agents = []

    

if __name__ == "__main__":
    test_simulation = Simulation(0, 10, 0.1)
    agent_A = BaseAgent( 'A', test_simulation, initial_state = KinematicState( pos = np.array((-5, -5)) ) )
    agent_B = BaseAgent( 'B', test_simulation, initial_state = KinematicState( pos = np.array((5, 5)) ) )
    test_simulation.run()
    print(agent_A.trajectory.pos)
    print(agent_B.trajectory.pos)
    
