import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import random
from tqdm import tqdm


class QRLearner:
    def __init__(
        self,
        obs_dim,
        num_action,
        h_to_learn,
        num_quan,
        lr,
        max_iter,
        batch_size,
        target_model,
        device="cuda",
    ) -> None:
        self.obs_dim = obs_dim
        self.num_quan = num_quan
        self.num_action = num_action
        self.h_to_learn = h_to_learn
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.target_model = target_model
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(obs_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, num_quan),
        ).to(self.device)

        self.tau = (
            torch.Tensor((2 * np.arange(self.num_quan) + 1) / (2.0 * self.num_quan))
            .reshape(1, -1)
            .to(self.device)
        )

    def forward(self, obs, action):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)

        ret = self.network(torch.cat([obs, action], dim=1))
        return ret

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

        quan = self.forward(obs, action).detach()
        idx = torch.randint(0, self.num_quan, [quan.shape[0], 1], device=self.device)
        x = quan.gather(1, idx)
        return x.cpu()

    def huber(self, x, k=5e-2):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

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

            quan = self.forward(obs, action)
            next_quan = (
                self.target_model.forward(next_obs, next_action)
                if self.target_model is not None
                else quan
            )
            target_quan = reward + (1 - done) * next_quan
            target_quan = target_quan.detach()

            diff = target_quan.t().unsqueeze(-1) - quan

            loss = self.huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()
            loss = loss.mean()

            loss_list.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        return self, loss_list, tmp
