import numpy as np
import matplotlib.pyplot as plt
import random
from isingpy.ising import IsingModel

def simulate(magnet, temp=1.0, n_runs=3, n_cycles=10000, sampling_freq=100, write_data=False):
    """Simulate an Ising magnet at a given tempeature.
    
    Parameters
    ----------
    magnet : IsingLattice
        An IsingLattice object
    temp : float
        Value of temperature at which simulation is performed
    n_runs : int
        Number of repetitions of the Monte Carlo simulation
    n_cycles : int
        Number of cycles in each MC run
    sampling_freq : int
        Number of samples to use for calculating observables
    write_data : bool, optional
        Write various data to disk (functionality to be added later)
    
    Returns
    -------
    E_per_spin : numpy.ndarray, dims=(n_runs, n_cycles/sampling_freq)
        Energy per spin recorded for each sample
    M_per_spin : numpy.ndarray
        Magnetization per spin recorded for each sample
    """
    M, N = magnet.dims[0],  magnet.dims[1]
    n_spins = M * N

    kb = 1.0
    beta = 1.0/(kb * temp)
    # initialize arrays to store data during simulation
    E_per_spin = np.zeros((n_runs, n_cycles/sampling_freq))
    M_per_spin = np.zeros((n_runs, n_cycles/sampling_freq))
    print "Current simulation running at Temperature %.1f." % temp
    for rep in xrange(n_runs):
        n_samples = 0
        for cycle in xrange(n_cycles):
            for attempt in xrange(n_spins):
                i, j = random.randint(0, M - 1), random.randint(0, N - 1)
                update_interactions = 0.0
                nearest_nbrs = magnet.get_nearest_nbrs([i, j])
                for nbr in nearest_nbrs:
                    update_interactions += magnet.lattice[nbr[0], nbr[1]]
                dE = -2.0 * magnet.lattice[i, j] * update_interactions
                boltzmann_factor = np.exp(-beta*dE)
                x = random.random()
                if dE < 0:
                    magnet.lattice[i, j] *= -1.0
                else:
                    magnet.lattice[i, j] *= -1.0
            if not cycle % sampling_freq:
                current_avg_e, current_avg_s = magnet.sample_observables()
                #print "%i %f %f" % (trial, current_avg_s, current_avg_e)
                E_per_spin[rep, n_samples] = current_avg_e
                M_per_spin[rep, n_samples] = current_avg_s
                n_samples += 1

    return E_per_spin, M_per_spin


ising = IsingModel([8, 8])
n_t_sample = 2**8
Tc = 2.269
#temperatures = np.random.normal(Tc, 0.64, n_t_sample)
#temperatures = temperatures[(temperatures > 1.2) & (temperatures < 3.8)]
temperatures = np.arange(1.2, 3.9, 0.1)
E = np.zeros((temperatures.shape[0]))
M = np.zeros((temperatures.shape[0]))
errors_E = np.zeros((E.shape[0]))
errors_M = np.zeros((M.shape[0]))
for index, T in enumerate(temperatures):
    runs = 1
    data_e, data_m = simulate(ising, temp=T, n_runs=runs, n_cycles=10000, sampling_freq=10)
    E[index] = np.mean(data_e)
    M[index] = np.mean(data_m)
    errors_E[index] = np.std(data_e)
    errors_M[index] = np.std(data_m)

fig, ax = plt.subplots(2, sharex='row')
ax[0].plot(temperatures, E, 'or')
ax[1].plot(temperatures, M, 'or')
ax[0].fill_between(temperatures, E - errors_E, E + errors_E,
                 color='gray', alpha=0.2)
ax[1].plot(temperatures, M, 'or')
ax[1].fill_between(temperatures, M - errors_M, M + errors_M,
                 color='gray', alpha=0.2)

fig.savefig("scripts/plots.pdf", dpi=300)

