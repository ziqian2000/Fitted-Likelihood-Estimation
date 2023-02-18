from typing import Dict, List
import numpy as np
from tqdm import tqdm
from data import Dataset


def rollout(init_action, env, n_traj, policy, h=0) -> np.ndarray:
    rets = []
    for _ in range(n_traj):
        s = env.reset(h)
        d = False
        ret = 0
        is_first_action = True
        while not d:
            if is_first_action:
                a = init_action
                is_first_action = False
            else:
                a = policy.sample_action(s)
            s, r, d, _ = env.step(a)
            ret += r
        rets.append(ret)
    return np.array(rets)


def compute_distance_via_samples(s1, s2, v_min, v_max, n_bins, tv=True):
    s1 = np.array(s1)
    s2 = np.array(s2)
    n_samples, data_dim = s1.shape
    delta = (v_max - v_min) / n_bins

    assert v_min <= s1.min() and s1.max() <= v_max
    assert v_min <= s2.min() and s2.max() <= v_max

    if data_dim == 1:
        if tv:
            bin1 = np.zeros(n_bins)
            bin2 = np.zeros(n_bins)
            for i in range(n_samples):
                bin1[((s1[i][0] - v_min) // delta).astype(int)] += 1
                bin2[((s2[i][0] - v_min) // delta).astype(int)] += 1
            pdf1 = bin1 / n_samples / delta
            pdf2 = bin2 / n_samples / delta
            return np.sum(np.abs(pdf1 - pdf2) * delta) / 2
        else:  ## wasserstein
            return np.sum(np.abs(np.sort(s1, 0) - np.sort(s2, 0))) / n_samples

    else:
        bin1 = np.zeros([n_bins, n_bins])
        bin2 = np.zeros([n_bins, n_bins])
        for i in range(n_samples):
            bin1[((s1[i][0] - v_min) // delta).astype(int)][
                ((s1[i][1] - v_min) // delta).astype(int)
            ] += 1
            bin2[((s2[i][0] - v_min) // delta).astype(int)][
                ((s2[i][1] - v_min) // delta).astype(int)
            ] += 1
        assert tv is True
        pdf1 = bin1 / n_samples / (delta**2)
        pdf2 = bin2 / n_samples / (delta**2)
        return np.sum(np.abs(pdf1 - pdf2) * (delta**2)) / 2
