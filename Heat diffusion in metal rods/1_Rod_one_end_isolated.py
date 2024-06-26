####################################################################################################
# Solving heat diffusion for a metal rod using finite difference method
# when it is heated initially to a custom initial condition.
# Here an initial condition is specified as a quadratic function as
# T(x, 0) = -800 * (x - 0.6)**2 + 1000
# One of its end is then isolated and the other end is cooled by keeping in contact with 
# reservoir at 100 C, so that heat loss happens through one end only.
####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Constants and parameters
L = 1.0  # length of the rod
T_reservoir = 400
alpha = 1e-4  # thermal diffusivity
dt = 0.1  # time step
dx = 0.01  # spatial step
duration = 5000  # total simulation time

# Discretization
Nx = int(L / dx) + 1
Nt = int(duration / dt) + 1
x_values = np.linspace(0, L, Nx)
t_values = np.linspace(0, duration, Nt)

# Initialize temperature array with the specified quadratic initial condition
T = np.zeros((Nt, Nx))
T[0, :] = 1000

# Boundary conditions
T[:, -1] = T_reservoir  # Right end in contact with the reservoir
T[:, 0] = T[:, 1]  # Insulated left end

# Finite difference method
for n in tqdm(range(0, Nt - 1)):
    for i in range(1, Nx - 1):
        T[n + 1, i] = T[n, i] + alpha * dt / dx**2 * (T[n, i + 1] - 2 * T[n, i] + T[n, i - 1])

    # Maintain the temperature at the insulated left end
    T[n + 1, 0] = T[n, 1]

# Create a 3D surface plot
# X, T_values = np.meshgrid(x_values, t_values)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(T_values, X, T, cmap='viridis')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Distance (m)')
# ax.set_zlabel('Temperature (C)')
# ax.set_title('Heat Diffusion in a Metal Rod (Left End Insulated, Right End in Contact with Reservoir)')
# plt.savefig('Heat_diffusion_in_metal_rods/Plots/HeatDiff_rod_one_end_isolated_3D.png', dpi=300)

def plot_contour(x_values, t_values, T):
    T_values, X = np.meshgrid(t_values, x_values)
    plt.figure(figsize=(6, 4))
    plt.contourf(X, T_values, T.T, 100, cmap='Spectral_r')
    plt.colorbar(label='Temperature (C)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (s)')
    plt.title('Heat Diffusion in a Metal Rod \n(Left End Insulated, Right End in Contact with Reservoir)')
    plt.tight_layout()
    plt.savefig('Heat_diffusion_in_metal_rods/Plots/HeatDiff_rod_one_end_isolated_contour.png', dpi=300)

plot_contour(x_values, t_values, T)

# plt.show()
