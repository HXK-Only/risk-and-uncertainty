import numpy as np
import gym



class SparseRewards(gym.Wrapper):

    def __init__(self, env, skip=4):

        self.env = env

        gym.Wrapper.__init__(self, env)


    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if done and self.env._elapsed_steps != self.env._max_episode_steps: 
            reward = 10
        else:
            reward = 0

        return obs, reward, done, info

