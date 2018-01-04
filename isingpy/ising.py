import numpy as np
import random

class IsingModel(object):
    """Class for modeling 2-D ising lattices.
    """
    def __init__(self, dims=None, interaction_strength=1.0, magnetic_field=0.0):
        """Initialize a 2-D Ising lattice.
        
        Parameters
        ----------
        dims : list, optional
            Number of spins in each dimension where only two dimensions are allowed
        interaction_strength : float, optional
            Pair-wise interaction strength, Default=1.0
        magnetic_field : float, optional
            Value of external magnetic field, Default=0.0

        Examples
        --------
        >>> ising_64 = IsingLattice(8, 8)
        >>> ising_16 = IsingLattice()

        Notes
        -----
        No upper-bound set up for the size of lattice, but unusually large lattices
        would be problematic for analytical solution of partition function.
        """

        if dims is None:
            dims = [4, 4]
        else:
            assert len(dims) == 2, "Only 2-D lattice can be modelled with this class."
            assert dims[0] != 0 or dims[1] != 0, "Only non-zero size allowed."
        self.dims = dims
        self.interaction_strength = interaction_strength
        self.magnetic_field = magnetic_field
        # Initialize lattice with all spins set to +1
        self.lattice = np.ones(self.dims, dtype="float")
        # randomly flip spins to generate initial an configuration
        for i in xrange(self.dims[0]):
            for j in xrange(self.dims[1]):
                if random.random() < 0.5:
                    self.lattice[i, j] *= -1.0
        
    def get_nearest_nbrs(self, pos):
        """Obtain nearest neighbor spins for a spin at a given position in lattice.
        
        Parameters
        ----------
        pos : list
            Position indices of the spin.
        """
        indices = []
        assert len(pos) == self.lattice.ndim, "Mismatch in dimensions."
        if pos[0] >= self.dims[0] or pos[1] >= self.dims[1]:
            raise ValueError("Index must be within the lattice of size %i x %i" % 
                             (self.dims[0], self.dims[1]))
        else:
            i, j = pos
            nbrs = [[(i - 1)%self.dims[0], j], [(i + 1)%self.dims[0], j],
                    [i, (j - 1)%self.dims[0]], [i, (j + 1)%self.dims[1]]]
            return nbrs
    
    def hamiltonian(self):
        """Returns hamiltonian of the system based on current configuration.
        
        Notes
        -----
            This should not be used when we are interested in differences in energies
            resulting from flipping a single spin. In those cases, no need to loop over
            all spins, just calculate new interactions with nearest neighbors of flipped spin.
        """
        interaction_energy = 0.0
        sum_spins = 0.0
        for i in xrange(self.dims[0]):
            for j in xrange(self.dims[1]):
                s_ij = self.lattice[i, j]
                sum_spins += s_ij
                interaction_energy -= s_ij * (self.lattice[i, (j+1)%self.dims[0]] + self.lattice[(i+1)%self.dims[0], j])
        hamiltonian = (self.interaction_strength * interaction_energy) - (self.magnetic_field * sum_spins) 
        return hamiltonian
    
    
    def sample_observables(self):
        """Sample current state of the system by obtaining average energy and magnetization
        
        Returns
        -------
        (avg_energy, avg_spin) : tuple
            A tuple with values of average energy and average spin in the current config
        """
        avg_energy = self.hamiltonian() / (self.dims[0]*self.dims[1]) 
        avg_spin = np.sum(self.lattice)/(self.dims[0]*self.dims[1])
        return avg_energy, avg_spin