####################################################################################################
# Solving the diffusion for two grains in contact to get the concentration of elements
# during element partitioning using finite difference method, initially with two arbitrary 
# grain concentration profiles 
# C(x, 0) = -0.02 * (x - 50)**2 + 200
# C(x, 0) = -0.02 * (x - 150)**2 + 150
# and modified junction conditions, are then joined together to attain equilibrium grain concentration.
# An arbitrary diffusion coefficient profile has also been considered w.r.t. the grain length.
####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def my_code():
    L_grain1 = 100.0  # length of the first grain
    L_grain2 = 100.0  # length of the second grain
    dt = 0.1  # time step
    dx = 1  # spatial step
    duration = 1000  # total simulation time
    gamma = 0.5

    # Discretization
    Nx_grain1 = int(L_grain1 / dx) + 1
    Nx_grain2 = int(L_grain2 / dx) + 1
    Nt = int(duration / dt) + 1

    # Define spatial coordinates for each grain
    x_values_grain1 = np.linspace(0, L_grain1, Nx_grain1)
    x_values_grain2 = np.linspace(L_grain1, L_grain1 + L_grain2, Nx_grain2)

    alpha_grain1 = 1*np.linspace(0.1, 1, Nx_grain1)  # thermal diffusivity for rod 1
    alpha_grain2 = 0.1*np.linspace(1, 0.1, Nx_grain2)  # thermal diffusivity for rod 2

    # Initialize concentration arrays for each grain
    Conc_grain1 = np.zeros((Nt, Nx_grain1))
    Conc_grain2 = np.zeros((Nt, Nx_grain2))

    # Initial conditions for each grain
    Conc_grain1[0, :] = 1
    Conc_grain2[0, :] = 0

    # Boundary conditions
    Conc_grain1[:, 0] = Conc_grain1[:, 1]  # Isolated boundary for grain 1
    Conc_grain2[:, -1] = Conc_grain2[:, -2]  # Isolated boundary for grain 2

    # Finite difference method for each grain
    for n in range(0, Nt - 1):
        for i in range(1, Nx_grain1 - 1):
            Conc_grain1[n + 1, i] = Conc_grain1[n, i] + alpha_grain1[i] * dt / dx**2 * (Conc_grain1[n, i + 1] - 2 * Conc_grain1[n, i] + Conc_grain1[n, i - 1])

        for i in range(1, Nx_grain2 - 1):
            Conc_grain2[n + 1, i] = Conc_grain2[n, i] + alpha_grain2[i] * dt / dx**2 * (Conc_grain2[n, i + 1] - 2 * Conc_grain2[n, i] + Conc_grain2[n, i - 1])

        # Isolated boundary conditions for both grains
        Conc_grain1[n + 1, 0] = Conc_grain1[n, 1]
        Conc_grain2[n + 1, -1] = Conc_grain2[n, -2]

        # Boundary condition at the junction of the two grains
        Conc_grain1[n + 1, -1] = (alpha_grain1[-1]*Conc_grain1[n+1, -2] + alpha_grain2[0]*Conc_grain2[n+1, 1]) / (alpha_grain1[-1] + gamma*alpha_grain2[0])
        Conc_grain2[n + 1, 0] = (alpha_grain1[-1]*Conc_grain1[n+1, -2] + alpha_grain2[0]*Conc_grain2[n+1, 1]) / (alpha_grain1[-1]/gamma + alpha_grain2[0])

    return Conc_grain1, Conc_grain2, x_values_grain1, x_values_grain2, duration, Nt

def plot_3D_twocolor(Conc_grain1, Conc_grain2, x_values_grain1, x_values_grain2, duration, Nt):
    # Create a 3D surface plot for the total concentration
    X_mesh1, t_mesh = np.meshgrid(x_values_grain1, np.linspace(0, duration, Nt))
    X_mesh2, t_mesh = np.meshgrid(x_values_grain2, np.linspace(0, duration, Nt))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_mesh1, t_mesh, Conc_grain1, cmap='viridis')
    ax.plot_surface(X_mesh2, t_mesh, Conc_grain2, cmap='plasma')
    ax.set_xlabel(r'Distance ($\mu$m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Concentration (ppm)')
    ax.set_title('Diffusion and element partitioning in two different grains')
    plt.savefig('Element_Partitioning_and_Diffusion/Miscellaneous/Plots/ElemPart_two_grains_var_diff_mBC_3D_twocolor.png', dpi=300)


def plot_3D(Conc_grain1, Conc_grain2, x_values_grain1, x_values_grain2, duration, Nt):
    x_values_total = np.concatenate((x_values_grain1, x_values_grain2))
    Conc_total = np.concatenate((Conc_grain1, Conc_grain2), axis=1)
    # Create a 3D surface plot for the total concentration
    X_mesh, t_mesh = np.meshgrid(x_values_total, np.linspace(0, duration, Nt))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_mesh, t_mesh, Conc_total, cmap='viridis')
    ax.set_xlabel(r'Distance ($\mu$m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Concentration (ppm)')
    ax.set_title('Heat Diffusion in Several Connected Rods of Different Materials')
    plt.savefig('Element_Partitioning_and_Diffusion/Miscellaneous/Plots/ElemPart_two_grains_var_diff_mBC_3D.png', dpi=300)


def plot_2D(Conc_grain1, Conc_grain2, x_values_grain1, x_values_grain2, duration, Nt):
    x_values_total = np.concatenate((x_values_grain1, x_values_grain2))
    Conc_total = np.concatenate((Conc_grain1, Conc_grain2), axis=1)
    # Create a 2D plot for Concentration vs Length for every (Nt/10)th time step
    plt.figure(figsize=(12, 8))
    for i in range(0, Nt, int(Nt/10)):
        plt.plot(x_values_total, Conc_total[i, :], label='t = ' + str(i*duration/(Nt-1)) + ' s')
    plt.xlabel(r'Distance ($\mu$m)')
    plt.ylabel('Concentration of mineral 1 (ppm)')
    plt.title('Heat Diffusion in Several Connected Rods of Different Materials')
    plt.legend()
    plt.savefig('Element_Partitioning_and_Diffusion/Miscellaneous/Plots/ElemPart_two_grains_var_diff_mBC.png', dpi=300)


Conc_grain1, Conc_grain2, x_values_grain1, x_values_grain2, duration, Nt = my_code()

plot_2D(Conc_grain1, Conc_grain2, x_values_grain1, x_values_grain2, duration, Nt)
plt.show()

