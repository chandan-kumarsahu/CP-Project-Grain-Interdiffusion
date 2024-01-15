####################################################################################################
# Solving heat diffusion for a metal rod using finite difference method
# when it is heated initially to a uniform temperature of 1000 C.
# One of its end is then isolated and the other end is cooled by keeping in contact with 
# reservoir at 100 C, so that heat loss happens through one end only.
####################################################################################################

import matplotlib.pyplot as plt
import numpy as np

# Constants and parameters
L = 1.0  # length of the rod
T_initial = 1000.0  # initial temperature
T_reservoir = 100.0  # temperature at the end in contact with the reservoir
alpha = 1e-4  # thermal diffusivity
dt = 0.1  # time step
dx = 0.01  # spatial step
duration = 5000  # total simulation time

# Discretization
Nx = int(L / dx) + 1
Nt = int(duration / dt) + 1
x_values = np.linspace(0, L, Nx)
t_values = np.linspace(0, duration, Nt)

# Initialize temperature array
T = np.zeros((Nt, Nx))

# Initial condition
T[0, :] = T_initial

# Boundary conditions
T[:, 0] = T_reservoir  # One end in contact with the reservoir
T[:, -1] = T[:, -2]  # Insulated boundary condition

# Finite difference method
for n in range(0, Nt - 1):
    for i in range(1, Nx - 1):
        T[n + 1, i] = T[n, i] + alpha * dt / dx**2 * (T[n, i + 1] - 2 * T[n, i] + T[n, i - 1])

    # Maintain the temperature at the insulated end
    T[n + 1, -1] = T[n, -2]

# Create a 3D surface plot
X, T_values = np.meshgrid(x_values, t_values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T_values, X, T, cmap='viridis')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Temperature (C)')
ax.set_title('Heat Diffusion in a Metal Rod (One End in Contact with Reservoir and Insulated at the other End)')
plt.show()
