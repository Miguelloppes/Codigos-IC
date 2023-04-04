# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#aaaaa
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
    from openpyxl.reader.excel import load_workbook
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

def create_Htable(H_table):
    for j in range(12):
        H_table[(0,j)][0] = 5
        H_table[(1,j)][0] = 5
        H_table[(2,j)][0] = 5
        H_table[(3,j)][1] = 5
        H_table[(4,j)][1] = 5
        H_table[(5,j)][1] = 5
    return H_table
        
def discretizer( pos , velocity , angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
    """Temperal diffrence for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value
    
def policy( state : tuple, e ):
    """Choosing action based on epsilon-greedy policy"""
    global Q_table
    Q_table = Q_table + H_table
    return np.argmax(Q_table[state])

def learning_rate(n : int , min_rate=0.1 ) -> float  :
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    """Decaying exploration rate"""
    min_rate = max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
    return min_rate
def python2excel(list, times):
    workbook = load_workbook(filename = "H_table.xlsx") 
    sheet = workbook.active   
    for i in range(1, len(list)+1,1):
        sheet[chr(int(ord('A')) + times) + str(i)] = list[i-1]
    workbook.save(filename="H_table.xlsx") 
    print("Excel salvo")
   
n_bins = (6 , 12 )
env = OpenAIGymEnvironment()
check_env(env)
Q_table = np.zeros(n_bins + (env.action_space.n,))
H_table = np.zeros(n_bins + (env.action_space.n,))
H_table = create_Htable(H_table)
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]   
Q_table = Q_table + H_table
list_rewards = []
def main():
    list_rewards.clear()
    n_episodes = 5000
    for e in range(n_episodes):
        print("===================")    
        print("episodie:", e)
        # Siscretize state into buckets
        current_state, done = discretizer(*env.reset()), False
        sun_reward = 0
        while done==False:
            # policy action 
            action = policy(current_state,e) # exploit
            # insert random action
            if np.random.random() < exploration_rate(e,0.1): 
                action = env.action_space.sample() # explore 
            # increment enviroment
            obs, reward, done, _ = env.step(action)
            new_state = discretizer(*obs)
            sun_reward = sun_reward + reward 
            # Update Q-Table
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward , new_state )
            old_value = Q_table[current_state][action]
            Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value
            current_state = new_state
            # Render the cartpole environment
        print("reward: ",sun_reward)  
        list_rewards.append(sun_reward)
    return list_rewards
if __name__ == '__main__':
    for episodes in range(1,30,1):
        list = main()
        python2excel(list,episodes)
        env.reset()
