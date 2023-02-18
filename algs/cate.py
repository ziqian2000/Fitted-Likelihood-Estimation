import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import random
from tqdm import tqdm


class CateLearner:
    def __init__(
        self,
        obs_dim,
        num_action,
        h_to_learn,
        num_atom,
        lr,
        max_iter,
        batch_size,
        data_dim,
        v_min,
        v_max,
        target_model,
        device="cuda",
    ) -> None:
        self.obs_dim = obs_dim
        self.h_to_learn = h_to_learn
        self.num_atom = num_atom
        self.num_action = num_action
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.v_min = v_min
        self.v_max = v_max
        self.target_model = target_model
        self.device = device
        self.v_gap = (v_max - v_min) / (num_atom - 1)
        self.v_range = torch.linspace(v_min, v_max, num_atom).to(self.device)

        self.network = nn.Sequential(
            nn.Linear(obs_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, num_atom**data_dim),
        ).to(self.device)

    def forward(self, obs, action):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)

        raw_output = self.network(torch.cat([obs, action], dim=1))
        output = F.softmax(raw_output, dim=1)
        return output

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

        val = self.forward(obs, action).detach()
        samples = Categorical(probs=val).sample()
        if self.data_dim == 1:
            return self.v_range[samples].reshape(-1, 1).cpu()
        else:
            d1 = torch.floor(samples / self.num_atom).long()
            d2 = (samples % self.num_atom).long()
            x = torch.cat(
                [self.v_range[d1].reshape(-1, 1), self.v_range[d2].reshape(-1, 1)],
                dim=1,
            )
            return x.cpu()

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

            val = self.forward(obs, action)
            next_val = (
                self.target_model.forward(next_obs, next_action)
                if self.target_model is not None
                else val
            )

            if self.data_dim == 1:

                target_pos = reward + (1 - done) * self.v_range.reshape(1, -1)
                target_pos = target_pos.clamp(self.v_min, self.v_max)
                target_rel_pos = (target_pos - self.v_min) / self.v_gap

                l_pos = torch.floor(target_rel_pos).long()
                u_pos = torch.ceil(target_rel_pos).long()
                prop = target_rel_pos - l_pos

                target_val = torch.zeros_like(val)
                for j in range(self.num_atom):
                    target_val[range(self.batch_size), l_pos[:, j]] += next_val[
                        :, j
                    ] * (1 - prop[:, j])
                    target_val[range(self.batch_size), u_pos[:, j]] += (
                        next_val[:, j] * prop[:, j]
                    )

            else:
                target_pos = reward.reshape(-1, self.data_dim, 1) + (
                    (1 - done) * self.v_range.reshape(1, -1)
                ).reshape(-1, 1, self.num_atom)
                target_rel_pos = (target_pos - self.v_min) / self.v_gap
                target_rel_pos = target_rel_pos.clamp(0, self.num_atom - 1)

                l_pos = torch.floor(target_rel_pos).long()
                u_pos = torch.ceil(target_rel_pos).long()
                prop = target_rel_pos - l_pos

                val = val.reshape(-1, self.num_atom, self.num_atom)
                next_val = next_val.reshape(-1, self.num_atom, self.num_atom)
                target_val = torch.zeros_like(val)
                for i in range(self.num_atom):
                    for j in range(self.num_atom):

                        target_val[
                            range(self.batch_size), l_pos[:, 0, i], l_pos[:, 1, j]
                        ] += (
                            next_val[:, i, j]
                            * (1 - prop[:, 0, i])
                            * (1 - prop[:, 1, j])
                        )

                        target_val[
                            range(self.batch_size), l_pos[:, 0, i], u_pos[:, 1, j]
                        ] += (next_val[:, i, j] * (1 - prop[:, 0, i]) * prop[:, 1, j])

                        target_val[
                            range(self.batch_size), u_pos[:, 0, i], l_pos[:, 1, j]
                        ] += (next_val[:, i, j] * prop[:, 0, i] * (1 - prop[:, 1, j]))

                        target_val[
                            range(self.batch_size), u_pos[:, 0, i], u_pos[:, 1, j]
                        ] += next_val[:, i, j] * (prop[:, 0, i] * prop[:, 1, j])

            loss = -target_val.detach() * torch.log(val + 1e-7)
            loss = loss.mean()

            loss_list.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        return self, loss_list, tmp
