import sys
from controller import Supervisor

try:
    import gym
    import random
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from sklearn.preprocessing import KBinsDiscretizer
    import time, math, random
    from typing import Tuple
    
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"'
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # Open AI Gym generic
        self.theta_threshold_radians = 0.2
        self.x_threshold = 0.3
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max
            ],
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__pendulum_sensor = None

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Motors
        self.__wheels = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        # Sensors
        self.__pendulum_sensor = self.getDevice('position sensor')
        self.__pendulum_sensor.enable(self.__timestep)

        # Internals
        super().step(self.__timestep)

        # Open AI Gym generic
        return np.array([0, 0, 0, 0]).astype(np.float32)

    def step(self, action):
        # Execute the action
        for wheel in self.__wheels:
            wheel.setVelocity(1.3 if action == 1 else -1.3)
        super().step(self.__timestep)

        # Observation
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")
        self.state = np.array([robot.getPosition()[2], robot.getVelocity()[2],
                               self.__pendulum_sensor.getValue(), endpoint.getVelocity()[3]])

        # Done
        done = bool(
            self.state[0] < -self.x_threshold or
            self.state[0] > self.x_threshold or
            self.state[2] < -self.theta_threshold_radians or
            self.state[2] > self.theta_threshold_radians
        )

        # Reward
        reward = 0 if done else 1

        return self.state.astype(np.float32), reward, done, {}

n_bins = (6 , 12 )
env = OpenAIGymEnvironment()
check_env(env)

lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]   

Q_table = np.zeros(n_bins + (env.action_space.n,))
Q_table.shape  

def discretizer( pos , velocity , angle, pole_velocity ) -> Tuple[int,...]:
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value
    
    
def policy( state : tuple ):
    return np.argmax(Q_table[state])

def learning_rate(n : int , min_rate=0.01 ) -> float  :
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))
    
   
def main():
    
    n_episodes = 10000 
    for e in range(n_episodes):
        print("===================")    
        print("episodie:", e)
        current_state, done = discretizer(*env.reset()), False
        sun_reward = 0
        while done==False:
           
            action = policy(current_state) # exploit
            
            if np.random.random() < exploration_rate(e) : 
                action = env.action_space.sample() # explore 
             
            obs, reward, done, _ = env.step(action)
            new_state = discretizer(*obs)
            sun_reward = sun_reward + reward 
            
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward , new_state )
            old_value = Q_table[current_state][action]
            Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value
            current_state = new_state
            
            
        print("reward: ",sun_reward)  
        fileq = open("reward.txt", "a")
        for s in range(1):
            fileq.write("%d %d\n" % (e, sun_reward))   
        fileq.close()             

if __name__ == '__main__':
    main()
    env.reset()
