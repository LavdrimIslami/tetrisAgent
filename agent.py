import cv2
import sys
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import tetris_gymnasium
from gym.spaces import Box
from gym.wrappers import FrameStack
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        "return every skipp'th frame"
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        "reapeat action, sum reward"
        total_reward = 0.0
        for i in range(self.skip):
            #accumulate reward and repeat same action
            observation, reward, done, trunk, info = self.env.step(action)
            total_reward += reward

            if done:
                break
        return observation, total_reward, done, trunk, info
    










if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    terminated = False
    while not terminated:
        env.render()
        action = env.action_space.sample() #swap with our own policy
        observation, reward, terminated, truncated, info = env.step(action)
        key = cv2.waitKey(100) # timeout to see the movement
    print("Game Over!")

env = SkipFrame(env, skip=4)
env = FrameStack(env, num_stack=4)