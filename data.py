from typing import Dict, List
import numpy as np
import random


class Dataset:
    def __init__(
        self,
        entry_list: List[List[Dict]],
        policy,
        env,
    ) -> None:

        self.H = env.H
        self.sample_pool = [[] for _ in range(self.H)]
        self.pool_idx = [0 for _ in range(self.H)]

        for h in range(self.H):
            action = policy.sample_batch_action(
                np.vstack([e["next_observation"] for e in entry_list[h]])
            )
            for e, a in zip(entry_list[h], action):
                e["next_action"] = [a]
            self.sample_pool[h] = entry_list[h]

    def pool_reset(self, h):
        random.shuffle(self.sample_pool[h])
        self.pool_idx[h] = 0

    def generate_next_val(self, h, model):
        if model is None:
            for sample in self.sample_pool[h]:
                sample["next_val"] = [0]
        else:
            agg_next_obs = np.array(
                [e["next_observation"] for e in self.sample_pool[h]]
            )
            agg_next_action = np.array([e["next_action"] for e in self.sample_pool[h]])
            agg_next_val = model.sample(
                agg_next_obs, agg_next_action, during_training=False
            )
            if not isinstance(agg_next_val, np.ndarray):
                agg_next_val = agg_next_val.numpy()
            for i, sample in enumerate(self.sample_pool[h]):
                sample["next_val"] = agg_next_val[i]

    def sample(self, batch_size, h):
        assert len(self.sample_pool[h]) % batch_size == 0
        if self.pool_idx[h] >= len(self.sample_pool[h]):
            self.pool_reset(h)
        batch = self.sample_pool[h][self.pool_idx[h] : self.pool_idx[h] + batch_size]
        self.pool_idx[h] += batch_size
        ret = (
            np.array([e["observation"] for e in batch]),
            np.array([e["action"] for e in batch]),
            np.array([e["reward"] for e in batch]),
            np.array([e["next_observation"] for e in batch]),
            np.array([e["next_action"] for e in batch]),
            np.array([e["done"] for e in batch]),
            np.array([e["next_val"] for e in batch]),
        )
        return ret
