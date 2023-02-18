import gym
from gym import spaces
import numpy as np
from scipy.linalg import hadamard
import torch
from torch.distributions import Normal
from tqdm import tqdm


class CombLock(gym.Env):
    def __init__(
        self,
        H=10,
        latent_dim=2,
        action_dim=2,
        noise_std=0.1,
        reward_std1=0.1,
        reward_std2=0.05,
        obs_dim=30,
        reward_dim=2,
    ):
        super().__init__()

        assert obs_dim >= H + latent_dim
        assert action_dim >= latent_dim - 1

        self.H = H
        self.noise_std = noise_std
        self.reward_std1 = reward_std1
        self.reward_std2 = reward_std2

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.reward_dim = reward_dim

        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Discrete(self.obs_dim)

        self.reset()

    def step(self, action):
        if self.latent < self.latent_dim - 1:
            if action != self.latent:  # optimal action
                self.latent += 1
        self.h += 1
        done = self.h == self.H
        reward = self.sample_reward(self.latent) if done else np.zeros(self.reward_dim)
        return self.sample_obs(self.latent, self.h), reward, done, {}

    def reset(self, h=0):
        self.latent = 0
        self.h = h
        self.obs = self.sample_obs(self.latent, self.h)
        return self.obs

    def sample_obs(self, latent, h):
        obs = np.zeros(self.obs_dim)
        obs[latent] += 1
        obs[self.latent_dim + h] += 1
        obs[self.latent_dim + self.H :] = np.random.normal(
            0, self.noise_std, self.obs_dim - self.latent_dim - self.H
        )
        return obs

    def sample_reward(self, latent):

        if self.reward_dim == 1:
            return np.random.randn(1) * self.reward_std1 + (1 - 2 * latent)
        else:
            latent = 1 - latent
            radius = 2 * latent / (self.latent_dim - 1)
            r = np.random.multivariate_normal(
                np.zeros(self.reward_dim),
                np.eye(self.reward_dim) * self.reward_std2,
            )
            return r + radius * r / np.linalg.norm(r, 2)

    def generate_uniform_data(self, n_entry):
        entry_list = [[] for _ in range(self.H)]
        print("Generating data...")
        for _ in tqdm(range(0, n_entry)):
            for latent in range(self.latent_dim):
                for h in range(self.H):
                    obs = self.sample_obs(latent, h)
                    action = np.random.randint(0, self.action_dim)
                    if latent < self.latent_dim - 1:
                        next_latent = latent + 1 if action != latent else latent
                    else:
                        next_latent = latent
                    next_obs = self.sample_obs(next_latent, h + 1)
                    done = 1 if h == self.H - 1 else 0
                    reward = (
                        self.sample_reward(next_latent)
                        if done
                        else np.zeros(self.reward_dim)
                    )

                    entry = dict()
                    entry["observation"] = obs
                    entry["action"] = [action]
                    entry["reward"] = reward
                    entry["next_observation"] = next_obs
                    entry["done"] = [done]
                    entry_list[h].append(entry)
        return entry_list
