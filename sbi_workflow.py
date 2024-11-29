import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from scipy.signal import find_peaks

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

# from multiprocessing import Pool, cpu_count

if __name__ == "__main__":
    # theta = (T, sigma, epsilon)
    prior_min = [1e-3, 0.005, 1.6]
    prior_max = [2000.0, 0.05, 5.0]

    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

    # load observation
    with open("data/observation_free.pkl", "rb") as pf:
        t, observed_pos = pickle.load(pf)

    # get initial positions
    x0 = observed_pos[0, :]
    dt = t[1] - t[0]
    box_length = 20

    def create_x(theta):
        T, epsilon, sigma = theta.numpy()
        return run_md(dt, len(t), x0, T, epsilon, sigma, box_length=box_length)

    def pairwise_distances(x):
        # Create a meshgrid of indices
        n = x.shape[1]
        i, j = np.triu_indices(n, k=1)
        # Compute distances for upper triangular part (unique pairs)
        distances = np.abs(x[:, i] - x[:, j])
        return distances

    def create_summary(theta):
        T, epsilon, sigma = theta.numpy()
        x = run_md(dt, len(t), x0, T, epsilon, sigma)
        final_pos = x[-1, :]
        avg_pairwise_distance = pairwise_distances(x).mean(axis=0)
        num_cp = np.array(
            [len(find_peaks(i)[0]) for i in ((x - x.mean(axis=0)) ** 2).T]
        )
        return np.concatenate([avg_pairwise_distance, final_pos, num_cp], axis=0)

    x_truth = observed_pos
    for i in range(0, x_truth.shape[1]):
        plt.plot(t, x_truth[:, i], "k", label=f"atom {i}")

    n_samples = 10000
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulate_x = process_simulator(create_x, prior, prior_returns_numpy)
    simulate_summary = process_simulator(create_summary, prior, prior_returns_numpy)
    print("Starting to simulate for sbi")
    start_time = time.time()
    theta, x_traj = simulate_for_sbi(
        simulate_x, prior, num_simulations=n_samples, num_workers=-1
    )
    elapsed_time = time.time() - start_time
    print(f"Finished simulating for sbi in {elapsed_time:.2f} seconds")

    for i, ci in zip(range(0, x_traj.shape[2]), mcolors.TABLEAU_COLORS):
        for j in range(20):
            plt.plot(t, x_traj[np.random.choice(n_samples), :, i], color=ci, alpha=0.2)
    plt.ylim(-box_length / 2.0, box_length / 2.0)
    plt.savefig("figures/uninformative_prior.png")
    plt.close()

    # train with summary statistics
    print("Starting to create summary statistics")
    start_time = time.time()
    theta, x = simulate_for_sbi(
        simulate_summary, prior, num_simulations=n_samples, num_workers=-1
    )
    elapsed_time = time.time() - start_time
    print(f"Finished summary statistics calculations in {elapsed_time:.2f} seconds")

    inference = SNPE(prior)
    print("Starting to train nn")
    start_time = time.time()
    density_estimator = inference.append_simulations(
        theta, x, proposal=prior, exclude_invalid_x=True
    ).train(training_batch_size=30, learning_rate=0.0001, show_train_summary=True)
    elapsed_time = time.time() - start_time
    print(f"Finished training nn in {elapsed_time:.2f} seconds")
    posterior = inference.build_posterior()

    # generate observed point summary statistics
    final_pos = x_truth[-1, :]
    avg_pairwise_distance = pairwise_distances(x_truth).mean(axis=0)
    num_cp = np.array(
        [len(find_peaks(i)[0]) for i in ((x_truth - x_truth.mean(axis=0)) ** 2).T]
    )
    x_o = np.concatenate([avg_pairwise_distance, final_pos, num_cp], axis=0)

    theta_p = posterior.sample((10000,), x=x_o)

    # THIS IS UNKOWN
    theta_o = (600, 0.0103, 3.4)

    fig, axes = pairplot(
        theta_p,
        limits=list(zip(prior_min, prior_max)),
        ticks=list(zip(prior_min, prior_max)),
        fig_kwargs={
            "figsize": (7, 7),
            "points_offdiag": {"markersize": 6},
            "points_colors": "r",
        },
        labels=["T", "epsilon", "sigma"],
        upper="kde",
        diag="kde",
        points=theta_o,
    )
    plt.savefig("figures/result.png", dpi=300)
    plt.close()
