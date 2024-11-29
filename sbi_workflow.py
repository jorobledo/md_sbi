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
from sbi.neural_nets import posterior_nn
import pickle
from simulator import run_md

from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)

class FlexibleBoxUniform(utils.BoxUniform):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.device = None
        
    def to(self, device):
        self.device = device
        super().__init__(
            low=self.low,
            high=self.high,
            device=self.device
        )
        
if __name__ == "__main__":
    # theta = (T, sigma, epsilon)
    prior_min = torch.as_tensor([1e-3, 0.005, 1.6])
    prior_max = torch.as_tensor([2000.0, 0.05, 5.0])

    prior = FlexibleBoxUniform(
        low=prior_min, high=prior_max
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
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

    # simulate in the cpu, multiple cores
    n_samples = 2000
    prior.to("cpu")
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

    # train NN
    # density_estimator_build_fun = torch.nn.DataParallel(posterior_nn(
    # model="nsf", hidden_features=30, num_transforms=4
    # ), device_ids=[i for i in range(4)])
    prior.to(device)
    # inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun, device=device)
    inference = SNPE(prior=prior, device=device)
    print("Starting to train nn")
    start_time = time.time()
    density_estimator = inference.append_simulations(
        theta.to(device), x.to(device),
        proposal=prior, exclude_invalid_x=True
    ).train(training_batch_size=30,
            learning_rate=0.0001,
            show_train_summary=True)
    elapsed_time = time.time() - start_time
    print(f"Finished training nn in {elapsed_time:.2f} seconds")
    
    # bring posterior back again to cpu just in case we are in gpu
    inference._neural_net.to("cpu")
    posterior = inference.build_posterior(prior=prior.to("cpu"))
 
    # generate observed point summary statistics
    final_pos = x_truth[-1, :]
    avg_pairwise_distance = pairwise_distances(x_truth).mean(axis=0)
    num_cp = np.array(
        [len(find_peaks(i)[0]) for i in ((x_truth - x_truth.mean(axis=0)) ** 2).T]
    )
    x_o = torch.as_tensor(np.concatenate([avg_pairwise_distance, final_pos, num_cp], axis=0))

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
