import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, Categorical, MultivariateNormal
import torch.nn.functional as F
import random
from tqdm import tqdm


class GMMLearner:
    def __init__(
        self,
        obs_dim,
        num_action,
        h_to_learn,
        num_dist,
        lr,
        max_iter,
        batch_size,
        data_dim,
        device="cuda",
    ) -> None:
        self.obs_dim = obs_dim
        self.num_dist = num_dist
        self.num_action = num_action
        self.h_to_learn = h_to_learn
        self.lr = lr
        self.max_iter = max_iter
        self.data_dim = data_dim
        self.batch_size = batch_size
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(obs_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, num_dist * 3 * data_dim),
        ).to(self.device)

    def forward(self, obs, action):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)

        ret = self.network(torch.cat([obs, action], dim=1))
        return ret

    def prob(self, obs, action, ret, cdf=False):
        if not isinstance(obs, torch.Tensor):
            if isinstance(obs, list):
                obs = np.array(obs)
            obs = torch.tensor(obs, dtype=torch.float)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64)

        sum_weighted_prob = 0
        sum_weight = 0
        output = self.forward(obs, action)
        for i in range(self.num_dist):

            if self.data_dim == 1:
                mean = output[:, [i * 3]]
                log_std = output[:, [i * 3 + 1]]
                log_weight = output[:, [i * 3 + 2]]

                dist = Normal(mean, torch.exp(log_std))
                if not cdf:
                    log_prob = dist.log_prob(ret)
                    prob = torch.exp(log_prob)
                else:
                    prob = dist.cdf(ret)

            else:
                mean = output[:, [i * 6, i * 6 + 1]]
                var1 = torch.exp(output[:, [i * 6 + 2]])
                var2 = torch.exp(output[:, [i * 6 + 3]])
                raw_cov = output[:, [i * 6 + 4]]
                cov = raw_cov.clamp(
                    -torch.sqrt(var1 * var2) + 1e-4, torch.sqrt(var1 * var2) - 1e-4
                )
                log_weight = output[:, i * 6 + 5]

                cov_mat = torch.cat(
                    [
                        var1,
                        cov,
                        cov,
                        var2,
                    ],
                    dim=-1,
                ).reshape(output.shape[0], 2, 2)

                dist = MultivariateNormal(mean, cov_mat)
                if not cdf:
                    log_prob = dist.log_prob(ret)
                    prob = torch.exp(log_prob)
                else:
                    prob = dist.cdf(ret)

            weighted_prob = prob * torch.exp(log_weight)
            sum_weight += torch.exp(log_weight)
            sum_weighted_prob += weighted_prob

        return sum_weighted_prob / sum_weight

    def sample(self, obs, action, during_training):
        if during_training is False:
            if not isinstance(obs, torch.Tensor):
                if isinstance(obs, list):
                    obs = np.array(obs)
                obs = torch.tensor(obs, dtype=torch.float)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.int64)
            obs = obs.to(self.device)
            action = action.to(self.device)

        output = self.forward(obs, action).detach()

        if self.data_dim == 1:
            picker = Categorical(logits=output[:, 2::3])
            i = picker.sample()
            mean = output.gather(1, (i * 3).reshape(-1, 1))
            log_std = output.gather(1, (i * 3 + 1).reshape(-1, 1))
            dist = Normal(mean, torch.exp(log_std))

        else:
            picker = Categorical(logits=output[:, 5::6])
            i = picker.sample()
            mean1 = output.gather(1, (i * 6).reshape(-1, 1))
            mean2 = output.gather(1, (i * 6 + 1).reshape(-1, 1))
            mean = torch.cat([mean1, mean2], dim=-1)
            var1 = torch.exp(output.gather(1, (i * 6 + 2).reshape(-1, 1)))
            var2 = torch.exp(output.gather(1, (i * 6 + 3).reshape(-1, 1)))
            raw_cov = output.gather(1, (i * 6 + 4).reshape(-1, 1))
            cov = raw_cov.clamp(
                -torch.sqrt(var1 * var2) + 1e-5, torch.sqrt(var1 * var2) - 1e-5
            )
            cov_mat = torch.cat(
                [
                    var1,
                    cov,
                    cov,
                    var2,
                ],
                dim=-1,
            ).reshape(output.shape[0], 2, 2)
            dist = MultivariateNormal(mean, cov_mat)

        return dist.sample().cpu()

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

            ll = torch.log(self.prob(obs, action, ret).clamp(1e-7, 1e9))
            loss = -ll.mean()
            loss_list.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        return self, loss_list, tmp
