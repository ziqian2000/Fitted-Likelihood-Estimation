import numpy as np
from env import CombLock
from scipy.special import comb


class Policy:
    def __init__(self, env) -> None:
        pass

    def sample_action(self, ob):
        raise NotImplementedError


class CombLockPolicy(Policy):
    def __init__(self, env: CombLock) -> None:
        super().__init__(env)
        self.noise = 7
        self.action_dim = env.action_dim
        self.latent_dim = env.latent_dim
        self.H = env.H

    def sample_action(self, ob):
        if np.random.randint(0, self.noise) == 0:
            # random
            action = np.random.randint(0, self.action_dim)
        else:
            # optimal
            latent = np.argmax(ob[: self.latent_dim])
            action = (
                latent
                if latent < self.latent_dim - 1
                else np.random.randint(0, self.action_dim)
            )
        return action

    def sample_batch_action(self, obs):
        # optimal
        latent = np.argmax(obs[:, : self.latent_dim], axis=1)
        optimal_action = np.where(
            latent < self.latent_dim - 1,
            latent,
            np.random.randint(0, self.action_dim, obs.shape[0]),
        )

        rand = np.random.randint(0, self.noise, obs.shape[0])
        return np.where(
            rand == 0,
            np.random.randint(0, self.action_dim, obs.shape[0]),
            optimal_action,
        )
