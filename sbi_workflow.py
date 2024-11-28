import numpy as np
import scipy.stats as stats
import torch
import pickle
import matplotlib.pyplot as plt


from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.utils.get_nn_models import (
    posterior_nn,
) 
from sbi.utils import BoxUniform
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import simulate_for_sbi, SNPE
from sbi.analysis import pairplot


from simulator import run_md

def plot_sim(t, x, name=False):
    for i in range(x.shape[1]):
            plt.plot(t, x[:, i], '.', label='atom {}'.format(i))
    plt.xlabel(r'time (s)')
    plt.ylabel(r'$x$-Position/Å')
    plt.legend(frameon=False)
    if name:
        plt.savefig(f"{name}.png")

if __name__ == "__main__":
    
    # load observation
    with open("observation.pkl", "rb") as pf:
        t, observed_pos = pickle.load(pf)
    
    # get initial positions
    x0 = observed_pos[0,:]
    dt = t[1]-t[0]
    
    # plot observation
    plot_sim(t, observed_pos, "observed_data")
    
    # define simulator
    def simulator(theta):
        ''' Define a simulator based on the initial positions observed '''
        temp, epsilon, sigma = theta.numpy()
        x = run_md(dt, len(t), x0, temp, epsilon, sigma)
        variance_pos = x.std(axis=0)
        min_distances = np.array([min(x[:,0]-x[:,1]), min(x[:,0] - x[:,2]), min(x[:,1] - x[:,2])])
        final_pos = x[-1,:]
        return np.stack([variance_pos, min_distances, final_pos]).flatten()
    
    # # define what will be compared as observation
    # def summary_statistics(theta):
    #     ''' calculate summary statistics based on the measured times '''
    #     x = simulator(theta)
    #     # delta_pos = x[-1,:] - x[0,:]
    #     # delta_t = t[-1] - t[0]
    #     # avg_velocity = delta_pos / delta_t
    #     # avg_pos = x.mean(axis=0)
        
        

    x_o = np.stack([observed_pos.std(axis=0), 
                    np.array([min(observed_pos[:,0] - observed_pos[:,1]),
                              min(observed_pos[:,0] - observed_pos[:,2]), 
                              min(observed_pos[:,1] - observed_pos[:,2])]),
                    observed_pos[-1,:]]).flatten()
    
    print("summary statistics observed:", x_o)
    
    # define prior (temperature, epsilon, sigma)
    prior = BoxUniform(low = torch.tensor([1e-3, 0.005, 1.6]), high=torch.tensor([2000.0, 0.05, 8.0]))
    
    # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    # Consistency check after making ready for sbi.
    check_sbi_inputs(simulator, prior)
    
    torch.manual_seed(123)
    np.random.seed(100)
    # theta = prior.sample((10000,))
    theta, x = simulate_for_sbi(simulator, prior, num_simulations=10000, num_workers=-1)
    # x = summary_statistics(theta.numpy())
    # define method for inference, defaults in training a maf
    density_estimator_build_fun = posterior_nn(
    model="nsf", hidden_features=60, num_transforms=3
    )
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    print(posterior) # prints how the posterior was trained
    
    samples = posterior.sample((10000,), x=x_o)
    mean_vals = samples.mean(axis=0).numpy() 
    print("result", mean_vals)
    
    
    _ = pairplot(samples,
                figsize=(6, 6),
                limits=[(0, 1200),(0.001, 0.05),(1.6, 8.0)],
                upper='kde', diag='kde',
                labels=[r"Temperature (K)", r"$\epsilon (eV)$", r"$\sigma (\AA)$"])
    plt.savefig("result_sbi.png", dpi=300)
    plt.close()
    
    plot_sim(t, observed_pos)
    x_est = simulator(*mean_vals)
    print("summary statistics estimated", x_est)

    for i in range(x_est.shape[1]):
        plt.plot(t, x_est[:, i], '.', label=f'estimated {i}')
    plt.savefig("estimated.png", dpi=300)
    