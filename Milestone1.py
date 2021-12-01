import numpy as np
from matplotlib import pyplot as plt

# only do longitudual friction
# print("x = {0}".format(x))

def grad_h(func): # Gradiant h
    """ Modified trapezoidal integration"""
    # Pads a 0 at the end of an array
    temp = np.pad(func, (0,1),
                  'constant',
                   constant_values=(0,0)) # Using roll calculate the diff (ghost node of 0)
    return (temp - np.roll(temp, 1))

class crossrod:
    def __init__(self, dt, total_length, elements, density, radius, G = 20, E = 40, dim = 3, **kwargs):
        # Element Info
        self.e = elements
        self.n = self.e + 1 # nodes
        self.n_i = self.e - 1 # internal nodes

        # Initializing node mass
        area = np.pi * (radius**2) # Update?
        total_volume = area * total_length
        total_mass = density * total_volume
        self.m = np.zeros((1,self.n))
        element_mass = total_mass / self.e
        self.m[0][0] = element_mass/2
        self.m[0][1:self.n-1] = element_mass
        self.m[0][self.n-1] = element_mass/2

        # Initializing node radii
        self.r = np.full((1,self.n),radius) # Update?

        # Initializing node position
        self.pos = np.zeros((dim,self.n))
        for col in range(self.n):
            self.pos[0,col] = (total_length/self.e) * col

        # Length Info
        # UPDATE THIS AT EVERY TIME STEP
        self.l = self.pos[:,1:] - self.pos[:,:-1] # length vector
        self.l_mag = np.linalg.norm(self.l, axis = 0) # magnitude of length
        # DO NOT UPDATE THIS AT EVERY TIME STEP
        self.l_ref = self.pos[:,1:] - self.pos[:,:-1] # reference length (unstrecthed length of the rod)
        self.l_ref_mag = np.linalg.norm(self.l_ref, axis = 0) # magnitude of reference length as a scalar

        # Parameters determined by Length Info
        self.dil_fac = self.l_mag / self.l_ref_mag # dilatation factor
        self.tangents = self.l / self.l_mag # tangent vectors

        # Directors
        self.directors = np.zeros((3, 3, self.e))
        for idx in range(self.e):
            self.directors[:, :, idx] = np.eye(3) # maps from lab to material frame

        self.forces = np.zeros((dim,self.n)) # forces INITIALIZE
        self.vel = np.zeros((dim,self.n)) # velocities

        # Shear/stretch diagonal matrix INITIALIZE INPUT FROM MATERIAL PROPERTIES
        self.S_hat = np.zeros((3,3,self.e))
        alpha_c = 4./3. # shape factor
        self.S_hat[0,0,:] = alpha_c * G * area
        self.S_hat[1,1,:] = alpha_c * G * area
        self.S_hat[2,2,:] = E * area

        # shear/stress strain
        self.s = self.dil_fac * self.tangents - self.directors[2,:,:]

        # Governing Equations
        # pos += vel * dt # Equation 1
        # dv_dt = (grad_h(S_hat @ s / dil_fac) + f) / m # Equation 3

        self.pos, self.vel = self.position_verlet(dt, self.pos, self.vel)
        update(self)

    def position_verlet(self, dt, x, v):
        """Does one iteration/timestep using the Position verlet scheme
        Parameters
        ----------
        dt : float
            Simulation timestep in seconds
        x : float/array-like
            Quantity of interest / position of COM
        v : float/array-like
            Quantity of interest / velocity of COM
        force_rule : ufunc
            A function, f, that takes one argument and
            returns the instantaneous forcing
        Returns
        -------
        x_n : float/array-like
            The quantity of interest at the Next time step
        v_n : float/array-like
            The quantity of interest at the Next time step
        """
        temp_x = x + 0.5*dt*v
        v_n = v + dt * self.force_rule(temp_x)
        x_n = temp_x + 0.5 * dt * v_n
        return x_n, v_n

    def force_rule(self, temp_pos):
        # First update
        self.update(temp_pos)

        print("S_hat = {0}".format(self.S_hat))
        print("s = {0}".format(self.s))

        # Governing Equation 3
        dv_dt = (grad_h(self.S_hat @ self.s / self.dil_fac) + self.forces) / self.m
        return dv_dt

    def update(self, temp_pos):
        # Update Length
        self.l = temp_pos[:,1:] - temp_pos[:,:-1]
        self.l_mag = np.linalg.norm(self.l, axis = 0)

        # Update dilatation factor
        self.dil_fac = self.l_mag / self.l_ref_mag

        # Update tangents
        self.tangents = self.l / self.l_mag

        # Update shear/stress strain
        self.s = self.dil_fac * self.tangents - self.directors[2,:,:]
        pass

test = crossrod(dt = 3E-4, total_length = 3, elements = 20, density = 5E3, radius = 0.25)
