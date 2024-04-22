import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Constants and parameters
L = 1.0  # length of the rod
thickness = 0.05  # thickness of the rod
T_reservoir = 400
alpha = 1e-4  # thermal diffusivity
dt = 0.8  # time step
dx = 0.02  # spatial step
duration = 5000  # total simulation time

# Discretization
Nx = int(L / dx) + 1
Ny = int(thickness / dx) + 1
Nt = int(duration / dt) + 1
x_values = np.linspace(0, L, Nx)
y_values = np.linspace(0, thickness, Ny)
t_values = np.linspace(0, duration, Nt)

# Initialize temperature array with the specified quadratic initial condition
T = np.zeros((Nt, Ny, Nx))
T[0, :, :] = -800 * (x_values - 0.3)**2 + 1000

# Boundary conditions
T[:, :, -1] = T_reservoir  # Right end in contact with the reservoir
T[:, :, 0] = T[:, :, 1]  # Insulated left end

# Finite difference method
for n in tqdm(range(0, Nt - 1)):
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            T[n + 1, j, i] = T[n, j, i] + alpha * dt / dx**2 * (
                T[n, j, i + 1] - 2 * T[n, j, i] + T[n, j, i - 1]
            )

    # Maintain the temperature at the insulated left end
    T[n + 1, :, 0] = T[n, :, 1]

print("Calculations done!")

# Function to update the animation
def update(frame):
    ax.clear()
    ax.imshow(T[frame, :, :], extent=[0, L, 0, thickness], aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Thickness (m)')
    ax.set_title(f'Time: {frame * dt:.1f} s')


# Create the animation
fig, ax = plt.subplots()
animation = FuncAnimation(fig, update, frames=Nt, interval=1, repeat=False)

plt.show()
