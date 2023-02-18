from algs.diffusion import DiffusionLearner
from algs.cate import CateLearner
from algs.gaussian import GMMLearner
from algs.quantile import QRLearner
import data
import utils
from policy import CombLockPolicy
from env import CombLock
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import torch
import os
from data import Dataset
from typing import List
from time import time

DATA_DIM = 1
DEVICE = "cpu"
N_SAMPLE_EVAL = 10000
N_DATA = 10000
HIST_BINS = 70
BATCH_SIZE = 500

DIFFUSION_ON = True
GAUSSIAN_ON = True
CATEGORICAL_ON = True
QUANTILE_ON = True


def save_model(models_to_save: List, alg_name: str, data_dim: int, n_data: int):
    name = f"{alg_name}_dim{data_dim}_data{n_data}_{int(time()*1000)}"
    path = f"./models/{name}.pt"
    torch.save(models_to_save, path)
    print(f"{name} saved.")


if __name__ == "__main__":

    env = CombLock(reward_dim=DATA_DIM)
    eval_policy = CombLockPolicy(env)
    dataset = Dataset(
        entry_list=env.generate_uniform_data(n_entry=N_DATA),
        policy=eval_policy,
        env=env,
    )

    algs_list = []
    para_list = []
    name_list = []

    if GAUSSIAN_ON:
        algs_list.append(GMMLearner)
        para_list.append(
            {
                "obs_dim": env.observation_space.n,
                "num_action": env.action_space.n,
                "num_dist": 10 if DATA_DIM == 2 else 10,
                "lr": 2e-4 if DATA_DIM == 2 else 1e-4,
                "max_iter": 10000 if DATA_DIM == 2 else 20000,
                "batch_size": BATCH_SIZE,
                "data_dim": DATA_DIM,
                "device": DEVICE,
            }
        )
        name_list.append("GMM")

    if DIFFUSION_ON:
        algs_list.append(DiffusionLearner)
        para_list.append(
            {
                "obs_dim": env.observation_space.n,
                "num_action": env.action_space.n,
                "num_step": 200,
                "beta_start": 1e-3,
                "beta_end": 1e-1,
                "lr": 1e-3,
                "max_iter": 15000 if DATA_DIM == 2 else 5000,
                "batch_size": BATCH_SIZE,
                "data_dim": DATA_DIM,
                "device": DEVICE,
            }
        )
        name_list.append("Diffusion")

    if CATEGORICAL_ON:
        algs_list.append(CateLearner)
        para_list.append(
            {
                "obs_dim": env.observation_space.n,
                "num_action": env.action_space.n,
                "num_atom": 30 if DATA_DIM == 2 else 100,
                "lr": 3e-2 if DATA_DIM == 2 else 1e-2,
                "max_iter": 100 if DATA_DIM == 2 else 200,
                "v_min": -4.0 if DATA_DIM == 2 else -1.5,
                "v_max": 4.0 if DATA_DIM == 2 else 1.5,
                "batch_size": BATCH_SIZE,
                "data_dim": DATA_DIM,
                "device": DEVICE,
            }
        )
        name_list.append("Categorical")

    if QUANTILE_ON and DATA_DIM == 1:
        algs_list.append(QRLearner)
        para_list.append(
            {
                "obs_dim": env.observation_space.n,
                "num_action": env.action_space.n,
                "num_quan": 100,
                "lr": 1e-3,
                "max_iter": 1000,
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
            }
        )
        name_list.append("Quantile")

    figure_width = env.H + 1
    figure_length = len(algs_list) * 2
    plt.figure(figsize=(figure_width * 5, figure_length * 5))

    for i, (alg, para, name) in enumerate(zip(algs_list, para_list, name_list)):
        model_list = [None]
        loss_list = []
        for h in reversed(range(env.H)):
            dataset.generate_next_val(h, model_list[-1])
            para["h_to_learn"] = h
            if alg is CateLearner or alg is QRLearner:
                para["target_model"] = model_list[-1]
            model = alg(**para)
            model, loss, _ = model.fit(dataset=dataset)
            model_list.append(model)
            loss_list.append(loss)

        # plot
        for h in range(env.H):
            obs_to_show = np.stack([env.sample_obs(0, h) for _ in range(N_SAMPLE_EVAL)])
            action_to_show = torch.zeros(obs_to_show.shape[0], 1)

            samples = model_list[-h - 1].sample(
                obs_to_show, action_to_show, during_training=False
            )
            if isinstance(samples, torch.Tensor):
                samples = samples.numpy()

            ax = plt.subplot(figure_length, figure_width, i * figure_width * 2 + h + 1)
            if DATA_DIM == 1:
                plt.hist(
                    samples[:, 0],
                    bins=HIST_BINS,
                    range=(-1.5, 1.5),
                    density=True,
                )
            else:
                plt.hist2d(
                    samples[:, 0],
                    samples[:, 1],
                    bins=HIST_BINS,
                    range=((-4, 4), (-4, 4)),
                )
            plt.title(f"{name} h={h}")

            ax = plt.subplot(
                figure_length,
                figure_width,
                i * figure_width * 2 + figure_width + h + 1,
            )
            plt.yscale("log")
            plt.plot(range(len(loss_list[-h - 1])), loss_list[-h - 1])
            plt.title("Loss")

        save_model(model_list, alg_name=name, data_dim=DATA_DIM, n_data=N_DATA)

    rets = utils.rollout(0, env, N_SAMPLE_EVAL, eval_policy)
    ax = plt.subplot(figure_length, figure_width, env.H + 1)
    if DATA_DIM == 1:
        plt.hist(
            rets[:, 0],
            bins=HIST_BINS,
            range=(-1.5, 1.5),
            density=True,
        )
    else:
        plt.hist2d(rets[:, 0], rets[:, 1], bins=HIST_BINS, range=((-4, 4), (-4, 4)))
    plt.title("Truth")

    plt.show()
