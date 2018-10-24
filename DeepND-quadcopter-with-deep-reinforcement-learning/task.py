import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., goal={'name':'default', 'target_pos': None, 'reward':0}):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.target_pos = goal['target_pos'] if goal['target_pos'] is not None else np.array([0., 0., 10.])
        self.name = goal['name']
        self.reward = goal['reward']
        if self.name == 'takeoff':
            # Random start for the takeoff task
            init_pose = [np.random.normal(0, 2), np.random.normal(0, 2), np.max(np.random.normal(0.5, 0.1),0), 0., 0., 0.] 
        
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        

        # Goal
         
        
        print('Creation of '+self.name+' task with target position '+str(self.target_pos))

    def get_reward(self,done):
        """ Return the reward given the current position and velocity."""
        if self.name is 'default':
            # Uses current pose of sim to return reward (by default)
            reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        elif self.name == 'takeoff':
            distance_to_goal =  1-.3*abs(self.sim.pose[2]- self.target_pos[2]) # distance along z axis
            #reward = -distance_to_goal
            x_velocity, y_velocity, z_velocity  = self.sim.v[:3].flatten() # Velocities along the different axis
            #reward = z_velocity-np.sqrt(x_velocity**2+y_velocity**2)/10  
            if self.reward == 0: # Only z-veloctiy
                reward = z_velocity
            elif self.reward == 1: # Add time
                reward = 1+z_velocity
            elif self.reward == 2: # Add position
                reward = 1+z_velocity -np.linalg.norm(self.sim.pose[:2]-self.sim.init_pose[:2])/1000+0.3*self.sim.pose[2]
            elif self.reward == 3:
                reward = 1+z_velocity -np.linalg.norm(self.sim.pose[:2]-self.sim.init_pose[:2])/1000+0.3*self.sim.pose[2]
                if done and self.sim.time < self.sim.runtime: 
                    reward -= 5 
            else:
                raise NotImplementedError('Not implemented reward')
            #reward = np.clip(reward,-1,1)
            reward = np.tanh(reward)
            #if done and self.sim.time < self.sim.runtime: 
            #    reward -= 1
            #if self.sim.pose[2] > self.target_pos[2]:
            #    reward += 1
            #reward = 1
            #if done and self.sim.time < self.sim.runtime: 
            #    reward -= 100
        else:
            raise NotImplementedError('Not implemented task')
           
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        if self.name == 'takeoff':
            # Random start for the takeoff task
            self.sim.init_pose = [np.random.normal(0, 2), np.random.normal(0, 2), np.max(np.random.normal(0.5, 0.1),0), 0., 0., 0.] 
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state