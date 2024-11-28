import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# sbi
import sbi.utils as utils
from sbi.inference import SNPE, simulate_for_sbi
from sbi.analysis import pairplot
import pickle
from simulator import run_md

from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)
from multiprocessing import Pool, cpu_count

if __name__ == "__main__":
    # theta = (T, sigma, epsilon)
    prior_min = [1e-3, 0.005, 1.6]
    prior_max = [1000.0, 0.05, 8.0]

    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

    # load observation
    with open("data/observation_free.pkl", "rb") as pf:
        t, observed_pos = pickle.load(pf)

    # get initial positions
    x0 = observed_pos[0, :]
    dt = t[1] - t[0]


    def create_x(theta):
        if theta.ndim == 1:
            theta = theta[np.newaxis, :]
        theta = np.array(theta)
        return run_md(dt, len(t), x0, theta[:, 0], theta[:, 1], theta[:, 2])


    def get_summary(theta):
        x = create_x(theta)
        variance_pos = x.std(axis=0)
        min_distances = np.array(
            [min(x[:, 0] - x[:, 1]), min(x[:, 0] - x[:, 2]), min(x[:, 1] - x[:, 2])]
        )
        final_pos = x[-1, :]
        return np.stack([variance_pos, min_distances, final_pos]).flatten()

    x_truth = observed_pos
    for i in range(0, 3):
        plt.plot(t, x_truth[:, i], "k", label=f"atom {i}")

    n_samples = 10000
    num_workers = 48
    print(f"Found {num_workers} workers.")
    # theta = prior.sample((n_samples,))
    # x = create_x(theta.numpy())
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(create_x, prior, prior_returns_numpy)
    print("Starting to simulate for sbi")
    start_time = time.time()
    theta, x = simulate_for_sbi(simulator, prior, num_simulations=n_samples, num_workers=num_workers)
    elapsed_time = time.time() - start_time
    print(f"Finished simulating for sbi in {elapsed_time:.2f} seconds")
    
    color = ["blue", "orange", "green"]
    for i in range(0, 3):
        for j in range(20):
            plt.plot(t, x[np.random.choice(n_samples), :, i], color=color[i], alpha=0.2)
    plt.ylim(-5, 15)
    plt.savefig("figures/uninformative_prior.png")
    plt.close()

    # train with summary statistics
    print("Starting to create summary statistics")
    start_time = time.time()
    with Pool(num_workers) as pool:
        x = pool.map(get_summary, theta)
    x = torch.tensor(np.array(x), dtype=torch.float32)
    elapsed_time = time.time() - start_time
    print(f"Finished summary statistics calculations in {elapsed_time:.2f} seconds")

    inference = SNPE(prior)
    print("Starting to train nn")
    start_time = time.time()
    _ = inference.append_simulations(theta, x).train()
    elapsed_time = time.time() - start_time
    print(f"Finished training nn in {elapsed_time:.2f} seconds")
    posterior = inference.build_posterior()

    # generate observed point summary statistics
    variance_pos_truth = x_truth.std(axis=0)
    min_distances_truth = np.array(
        [
            min(x_truth[:, 0] - x_truth[:, 1]),
            min(x_truth[:, 0] - x_truth[:, 2]),
            min(x_truth[:, 1] - x_truth[:, 2]),
        ]
    )
    final_pos_truth = x_truth[-1, :]
    x_o = np.stack([variance_pos_truth, min_distances_truth, final_pos_truth]).flatten()

    theta_p = posterior.sample((10000,), x=x_o)

    # THIS IS UNKOWN
    theta_o = (600, 0.0103, 3.4)


    fig, axes = pairplot(
        theta_p,
        limits=list(zip(prior_min, prior_max)),
        ticks=list(zip(prior_min, prior_max)),
        fig_kwargs={'figsize':(7, 7)},
        labels=["T", "epsilon", "sigma"],
        points_offdiag={"markersize": 6},
        points_colors="r",
        points=theta_o,
    )
    plt.savefig("figures/result.png", dpi=300)
    plt.close()
