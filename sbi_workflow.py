import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# sbi
import sbi.utils as utils
from sbi.inference import infer
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi.analysis import pairplot
import pickle
from simulator import run_md

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

prior_min = [1e-3, 0.005, 1.6]
prior_max = [2000.0, 0.05, 8.0]

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min),
    high=torch.as_tensor(prior_max)
)

# load observation
with open("observation.pkl", "rb") as pf:
        t, observed_pos = pickle.load(pf)

# get initial positions
x0 = observed_pos[0,:]
dt = t[1]-t[0]
    
def create_x(theta):
    if theta.ndim == 1:
        theta = theta[np.newaxis, :]
    theta = np.array(theta)
    return run_md(dt, len(t), x0, theta[:,0], theta[:,1], theta[:,2])
    
theta_o = np.array([600, 0.01, 3.4])

def get_summary(theta):
    x = create_x(theta)
    variance_pos = x.std(axis=0)
    min_distances = np.array([min(x[:,0]-x[:,1]), min(x[:,0] - x[:,2]), min(x[:,1] - x[:,2])])
    final_pos = x[-1,:]
    return np.stack([variance_pos, min_distances, final_pos]).flatten()

x_truth = create_x(theta_o)
for i in range(0,3):
    plt.plot(t, x_truth[:,i], "k", label=f"atom {i}")

n_samples = 1000
# theta = prior.sample((n_samples,))
# x = create_x(theta.numpy())
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(create_x, prior, prior_returns_numpy)
theta, x = simulate_for_sbi(simulator, prior, num_simulations=n_samples)

color = ['blue', 'orange', 'green']
for i in range(0,3):
    for j in range(n_samples):
        plt.plot(t, x[j,:,i], color=color[i], alpha=0.2)
plt.ylim(-5,15)
plt.savefig("prueba.png")

x = get_summary(theta.numpy())
inference = SNPE(prior)
_ = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

x_o = get_summary(theta_o)
theta_p = posterior.sample((1000,), x=x_o)

fig, axes = pairplot(
    theta_p,
    limits=list(zip(prior_min, prior_max)),
    ticks=list(zip(prior_min, prior_max)),
    figsize=(7, 7),
    labels=["T", "epsilon", "sigma"],
    points_offdiag={"markersize": 6},
    points_colors="r",
    points=theta_o,
)
plt.savefig("result_2.png",dpi=300)
plt.close()

