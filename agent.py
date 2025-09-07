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
        for i in range(self._skip):
            #accumulate reward and repeat same action
            observation, reward, done, trunk, info = self.env.step(action)
            total_reward += reward

            if done:
                break
        return observation, total_reward, done, trunk, info

class Player:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #agent dnn to predict the most optimal action we implement in learn
        self.net = PlayerNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device = self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5 #experiences between saving playernet

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync


    def act(self, state):
        "given a state, choose epsilon greedy action"
        #inputs:
        #state(lazyframe) single observation of the current state, dimension is state_dim
        #outputs:
        #action_idx int: an int representing which action player will perform

        #explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        #exploit
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device = self.device).unsqueeze(0)
            action_values = self.net(state, model = "online")
            action_idx = torch.argmax(action_values, axis = 1).item()

        #decrease rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        #increment step
        self.curr_step += 1
        return action_idx

        

    def cache(self, experience):
        "add experience to memory"
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))



    def recall(self):
        "sample experiences from memory"
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
        

    def learn(self):
        "update online action value Q function with batch of experiences"
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin == 0:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None
        
        #sample from memory
        state, next_state, action, reward, done = self.recall()

        #get td estimate
        td_est = self.td_estimate(state, action)

        #get target
        td_tgt = self.td_target(reward, next_state, done)

        #backprop
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)



if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env = SkipFrame(env, skip=4)
    env = FrameStack(env, num_stack=4)
    use_cuda = torch.cuda.is_available()
    print(f"USING CUDA: {use_cuda}")
    print()

    save_path = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    player = Player(state_dim=(4,84,84), action_dim=env.action_space.n, save_dir=save_dir)

    episodes = 40

    for e in range(episodes):
        state = env.reset()

        #play
        while True:

            #run agent on the state
            action = player.act(state)

            #agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            #remember
            player.cache(state, next_state, action, reward, done)

            #learn
            q, loss = player.learn()

            #logg later not now

            #update
            state = next_state

            #check if done
            if done or info["flag_get"]:
                break

    

    # terminated = False
    # while not terminated:
    #     env.render()
    #     action = env.action_space.sample() #swap with our own policy
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     key = cv2.waitKey(100) # timeout to see the movement
    # print("Game Over!")


