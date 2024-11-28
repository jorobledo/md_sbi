import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
import pickle
mass_of_argon = 39.948 # amu

"""Simulation from https://pythoninchemistry.org/sim_and_scat/molecular_dynamics/build_an_md"""

def update_pos(x, v, a, dt, box_length):
    """
    Update the particle positions accounting for the 
    periodic boundary condition.
    
    Parameters
    ----------
    x: ndarray of floats
        The positions of the particles in a single dimension
    v: ndarray of floats
        The velocities of the particles in a single dimension
    a: ndarray of floats
        The accelerations of the particles in a single dimension
    dt: float
        The timestep length
    box_length: float 
        The size of the periodic cell
    
    Returns
    -------
    ndarray of floats:
        New positions of the particles in a single dimension
    """
    new_pos = x + v * dt + 0.5 * a * dt * dt
    #print(new_pos)
    new_pos = new_pos % box_length
    #print(new_pos)
    return new_pos

def lj_force(r, epsilon, sigma):
    """
    Implementation of the Lennard-Jones potential 
    to calculate the force of the interaction.
    
    Parameters
    ----------
    r: float
        Distance between two particles (Å)
    epsilon: float 
        Potential energy at the equilibrium bond 
        length (eV)
    sigma: float 
        Distance at which the potential energy is 
        zero (Å)
    
    Returns
    -------
    float
        Force of the van der Waals interaction (eV/Å)
    """
    return 48 * epsilon * np.power(
        sigma / r, 13) - 24 * epsilon * np.power(
        sigma / r, 7)

def get_accelerations(positions, box_length, cutoff, lj_params):
    """
    Calculate the acceleration on each particle as a 
    result of each other particle. 
    
    Parameters
    ----------
    positions: ndarray of floats
        The positions, in a single dimension, for all
        of the particles
    box_length: float 
        The size of the periodic cell
    cutoff: float
        The distance after which the interaction 
        is ignored
        
    Returns
    -------
    ndarray of floats
        The acceleration on each particle
    """
    accel_x = np.zeros((positions.size, positions.size))
    for i in range(0, positions.size - 1):
        for j in range(i + 1, positions.size):
            r_x = positions[j] - positions[i]
            r_x = r_x % box_length
            rmag = np.sqrt(r_x * r_x)
            force_scalar = lj_force(rmag, *lj_params)
            force_x = force_scalar * r_x / rmag
            accel_x[i, j] = force_x / mass_of_argon
            accel_x[j, i] = - force_x / mass_of_argon
    return np.sum(accel_x, axis=0)


def update_velo(v, a, a1, dt):
    """
    Update the particle velocities.
    
    Parameters
    ----------
    v: ndarray of floats
        The velocities of the particles in a single dimension
    a: ndarray of floats
        The accelerations of the particles in a single dimension 
        at the previous timestep
    a1: ndarray of floats
        The accelerations of the particles in a single dimension
        at the current timestep
    dt: float
        The timestep length
    
    Returns
    -------
    ndarray of floats:
        New velocities of the particles in a single dimension
    """
    return v + 0.5 * (a + a1) * dt

def init_velocity(T, number_of_particles):
    """
    Initialise the velocities for a series of particles.
    
    Parameters
    ----------
    T: float
        Temperature of the system at initialisation
    number_of_particles: int
        Number of particles in the system
    
    Returns
    -------
    ndarray of floats
        Initial velocities for a series of particles
    """
    R = np.random.rand(number_of_particles) - 0.5
    # R = np.array([0.1,0.1,-0.1])
    return R * np.sqrt((Boltzmann / 1.602e-19) * T / mass_of_argon)

def run_md(dt, number_of_steps, x, initial_temp, epsilon, sigma, seed=0):
    """
    Run a MD simulation.
    
    Parameters
    ----------
    dt: float
        The timestep length
    number_of_steps: int
        Number of iterations in the simulation
    initial_temp: float
        Temperature of the system at initialisation
    x: ndarray of floats
        The initial positions of the particles in a single dimension
        
    Returns
    -------
    ndarray of floats
        The positions for all of the particles throughout the simulation
    """
    box_length = 13.0
    cutoff = box_length / 2.
    positions = np.zeros((number_of_steps, x.size))
    v = init_velocity(initial_temp, x.size)
    lj_params = (epsilon, sigma)
    a = get_accelerations(x, box_length, cutoff, lj_params)
    for i in range(number_of_steps):
        x = update_pos(x, v, a, dt, box_length)
        a1 = get_accelerations(x, box_length, cutoff, lj_params)
        v = update_velo(v, a, a1, dt)
        a = np.array(a1)
        positions[i, :] = x
    return positions

if __name__ == "__main__":
    x = np.array([1.0, 5.0, 10.0])
    lj_params = (0.0103, 3.4)
    temperature = 600
    t0 = 0.0
    dt = 0.1
    t_steps = 4000
    statement = f"""
{x.shape[0]} atom simulator
---------------------------
Initial positions = {x}
Lennard-Jones potential parameters sigma={lj_params[0]}, epsilon={lj_params[1]}.
System temperature = {temperature}K
time step = {dt}
time steps = {t_steps}
Result stored in data/observation_pbc.pkl - (t,positions)
Trajectory plot in figures/example_pbc_trajectories.pnh     
"""
    print(statement)
    t = np.linspace(t0,t0+(dt*t_steps), t_steps, endpoint=False)
    sim_pos = run_md(dt, t_steps, x, temperature, *lj_params)
        
    for i in range(sim_pos.shape[1]):
        plt.plot(t, sim_pos[:, i], '.', label='atom {}'.format(i))
    plt.xlabel(r'time (s)')
    plt.ylabel(r'$x$-Position/Å')
    plt.legend(frameon=False)
    plt.savefig("figures/example_pbc_trajectories.png")
    
    with open("data/observation_pbc.pkl", "wb") as pf:
        pickle.dump((t,sim_pos), pf)