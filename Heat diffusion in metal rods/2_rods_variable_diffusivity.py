####################################################################################################
# Solving heat diffusion for system with two metal rods of different material using 
# finite difference method initially heated to two different arbitrary temperature profiles 
# T(x, 0) = -800 * (x - 0.6)**2 + 1000  and 
# T(x, 0) = -600 * (x - 1.5)**2 + 600
# and then joined together while keeping their extreme ends isolated. 
# The heat flow balances in both rods to equilibriate. 
####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm


def my_code():
    # Constants and parameters
    L_rod1 = 1.0  # length of the first rod
    L_rod2 = 1.0  # length of the second rod
    dt = 0.1  # time step
    dx = 0.05  # spatial step
    duration = 10000  # total simulation time

    # Discretization
    Nx_rod1 = int(L_rod1 / dx) + 1
    Nx_rod2 = int(L_rod2 / dx) + 1
    Nt = int(duration / dt) + 1

    alpha_rod1 = 1e-3*np.linspace(0.1, 1, Nx_rod1)  # thermal diffusivity for rod 1
    alpha_rod2 = 1e-4*np.linspace(1, 0.1, Nx_rod2)  # thermal diffusivity for rod 2

    k_rod1 = 398.0*np.linspace(0.1, 1, Nx_rod1)
    k_rod2 = 100.0*np.linspace(0.1, 1, Nx_rod2)

    # Define spatial coordinates for each rod
    x_values_rod1 = np.linspace(0, L_rod1, Nx_rod1)
    x_values_rod2 = np.linspace(L_rod1, L_rod1 + L_rod2, Nx_rod2)
    x_values_total = np.concatenate((x_values_rod1, x_values_rod2))

    # Initialize temperature arrays for each rod
    T_rod1 = np.zeros((Nt, Nx_rod1))
    T_rod2 = np.zeros((Nt, Nx_rod2))

    # Initial conditions for each rod
    T_rod1[0, :] = -800 * (x_values_rod1 - 0.5)**2 + 1000
    T_rod2[0, :] = -600 * (x_values_rod2 - 1.5)**2 + 600

    # Boundary conditions
    T_rod1[:, 0] = T_rod1[:, 1]  # Isolated boundary for rod 1
    T_rod2[:, -1] = T_rod2[:, -2]  # Isolated boundary for rod 2

    # Finite difference method for each rod
    for n in tqdm(range(0, Nt - 1)):
        for i in range(1, Nx_rod1 - 1):
            T_rod1[n + 1, i] = T_rod1[n, i] + alpha_rod1[i] * dt / dx**2 * (T_rod1[n, i + 1] - 2 * T_rod1[n, i] + T_rod1[n, i - 1]) + (alpha_rod1[i+1] - alpha_rod1[i]) * dt / (dx**2) * (T_rod1[n, i + 1] - T_rod1[n, i])

        for i in range(1, Nx_rod2 - 1):
            T_rod2[n + 1, i] = T_rod2[n, i] + alpha_rod2[i] * dt / dx**2 * (T_rod2[n, i + 1] - 2 * T_rod2[n, i] + T_rod2[n, i - 1]) + (alpha_rod2[i+1] - alpha_rod2[i]) * dt / (dx**2) * (T_rod2[n, i + 1] - T_rod2[n, i])

        # Isolated boundary conditions for both rods
        T_rod1[n + 1, 0] = T_rod1[n, 1]
        T_rod2[n + 1, -1] = T_rod2[n, -2]

        # Boundary condition at the junction of the two rods
        T_rod1[n + 1, -1] = (k_rod1[-1]*T_rod1[n+1, -2] + k_rod2[0]*T_rod2[n+1, 1]) / (k_rod1[-1] + k_rod2[0])
        T_rod2[n + 1, 0] = T_rod1[n + 1, -1]

    # Combine the temperatures of the two rods
    T_total = np.concatenate((T_rod1, T_rod2), axis=1)

    return T_total, x_values_total, duration, Nt

T_total, x_values_total, duration, Nt = my_code()

def plot_3D():
    # Create a 3D surface plot for the total temperature
    X_total, T_values_total = np.meshgrid(x_values_total, np.linspace(0, duration, Nt))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_total, T_values_total, T_total, cmap='viridis')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Temperature (C)')
    ax.set_title('Heat Diffusion in Several Connected Rods of Different Materials')

    plt.savefig('Heat_diffusion_in_metal_rods/Plots/HeatDiff_two_connected_rods_mdiff_3D.png', dpi=300)

def plot_2D():
    # Create a 2D plot for Temperature vs Length for every (Nt/10)th time step
    plt.figure(figsize=(12, 8))
    for i in range(0, Nt, int(Nt/10)):
        plt.plot(x_values_total, T_total[i, :], label='t = ' + str(i*duration/(Nt-1)) + ' s')
    plt.xlabel('Distance (m)')
    plt.ylabel('Temperature (C)')
    plt.title('Heat Diffusion in Several Connected Rods of Different Materials')
    plt.legend()
    plt.savefig('Heat_diffusion_in_metal_rods/Plots/HeatDiff_two_connected_rods_mdiff.png', dpi=300)

def plot_contour(T_total, x_values_total, duration, Nt):
    # Create meshgrid for contour plot
    X_total, T_values_total = np.meshgrid(x_values_total, np.linspace(0, duration, Nt))

    # Plot contour
    plt.figure(figsize=(6, 4))
    plt.contourf(X_total, T_values_total, T_total, 100, cmap='Spectral_r')
    plt.colorbar(label='Temperature (C)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (s)')
    plt.title('Heat Diffusion in Several Connected Rods \nof Different Materials with Variable Diffusivity')
    plt.savefig('Heat_diffusion_in_metal_rods/Plots/HeatDiff_two_connected_rods_mdiff_contour.png', dpi=300)

plot_contour(T_total, x_values_total, duration, Nt)

# plt.show()

