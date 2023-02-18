import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import Dataset
from env import CombLock


class DiffusionLearner:
    def __init__(
        self,
        obs_dim,
        num_action,
        h_to_learn,
        num_step,
        beta_start,
        beta_end,
        lr,
        max_iter,
        batch_size,
        data_dim,
        device="cuda",
    ) -> None:
        self.obs_dim = obs_dim
        self.num_step = num_step
        self.h_to_learn = h_to_learn
        self.num_action = num_action
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(obs_dim + 1 + data_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
        ).to(self.device)

        self.init_paras()

    def init_paras(self):
        self.beta = torch.linspace(
            start=self.beta_start, end=self.beta_end, steps=self.num_step
        ).to(self.device)
        self.sigma = torch.sqrt(self.beta)
        self.alpha = 1 - self.beta
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha = 1 - self.alpha
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.multiplier2 = self.one_minus_alpha / self.sqrt_one_minus_alpha_bar
        self.multiplier1 = 1 / self.sqrt_alpha

    def forward(self, obs, action, x, t):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float)

        ret = self.network(
            torch.cat([obs, action, x, t.reshape(-1, 1) / self.num_step], dim=1)
        )
        return ret

    def reverse_sample(self, obs, action, x_t, t):
        mul2_t = self.multiplier2.gather(0, t).reshape(-1, 1)
        mul1_t = self.multiplier1.gather(0, t).reshape(-1, 1)

        eps_theta = self.forward(obs, action, x_t, t)
        mean = mul1_t * (x_t - mul2_t * eps_theta)
        sigma_z = torch.gather(self.sigma, 0, t).reshape(-1, 1) * torch.randn_like(
            x_t, device=self.device
        )
        return mean + sigma_z

    def sample(self, obs, action, during_training=False):
        if during_training is False:
            if not isinstance(obs, torch.Tensor):
                if isinstance(obs, list):
                    obs = np.array(obs)
                obs = torch.tensor(obs, dtype=torch.float)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.int64)
            obs = obs.to(self.device)
            action = action.to(self.device)

        batch_size = obs.shape[0]
        x = torch.randn([batch_size, self.data_dim], device=self.device)
        for t in reversed(range(self.num_step)):
            x = self.reverse_sample(
                obs,
                action,
                x,
                torch.tensor(t).repeat(batch_size).to(self.device),
            ).detach()
        return x if during_training else x.cpu()

    def compute_loss(self, obs, action, x_0):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)

        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_step, size=[batch_size], device=self.device)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar.gather(0, t).reshape(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar.gather(0, t).reshape(
            -1, 1
        )
        eps = torch.randn_like(x_0, device=self.device)
        eps_theta = self.forward(
            obs,
            action,
            sqrt_alpha_bar_t * x_0 + eps * sqrt_one_minus_alpha_bar_t,
            t,
        )
        return torch.square(eps - eps_theta).mean()

    def fit(self, dataset):
        loss_list = []
        opt = Adam(self.network.parameters(), lr=self.lr)

        tmp = []

        for _ in tqdm(range(self.max_iter)):

            obs, action, reward, next_obs, next_action, done, next_val = dataset.sample(
                self.batch_size, self.h_to_learn
            )

            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
            next_action = torch.tensor(next_action, dtype=torch.long).to(self.device)
            done = torch.tensor(done, dtype=torch.float).to(self.device)
            next_val = torch.tensor(next_val, dtype=torch.float).to(self.device)

            ret = reward + (1 - done) * next_val
            ret = ret.detach().float()

            loss = self.compute_loss(
                obs,
                action,
                x_0=ret,
            )
            loss_list.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        return self, loss_list, tmp
