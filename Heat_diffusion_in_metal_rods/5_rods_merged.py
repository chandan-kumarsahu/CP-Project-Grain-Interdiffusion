####################################################################################################
# Solving heat diffusion for system with several metal rods of different material using 
# finite difference method initially heated to different arbitrary temperature profiles 
# and then joined together while keeping their extreme ends isolated. 
# The heat flow balances in both rods to equilibriate. 

# Here we have 5 rods of different materials joined together. The first, third and fifth
# rods are made of metal with thermal diffusivity of 1e-4 m^2/s and thermal conductivity
# of 398 W/mK. The second and fourth rods are made of metal with thermal diffusivity of
# 1e-5 m^2/s and thermal conductivity of 100 W/mK. 
####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def my_code():
    # Constants and parameters
    L_rod1 = 1.0  # length of the first rod
    L_rod2 = 1.0  # length of the second rod
    L_rod3 = 1.0  # length of the third rod
    L_rod4 = 1.0  # length of the fourth rod
    L_rod5 = 1.0  # length of the fifth rod

    alpha_rod1 = 1e-4  # thermal diffusivity for rod 1
    alpha_rod2 = 1e-5  # thermal diffusivity for rod 2
    alpha_rod3 = 1e-4  # thermal diffusivity for rod 3
    alpha_rod4 = 1e-5  # thermal diffusivity for rod 4
    alpha_rod5 = 1e-4  # thermal diffusivity for rod 5

    dt = 0.1  # time step
    dx = 0.01  # spatial step
    duration = 20000  # total simulation time

    k_rod1 = 398.0
    k_rod2 = 100.0
    k_rod3 = 398.0
    k_rod4 = 100.0
    k_rod5 = 398.0

    # Discretization
    Nx_rod1 = int(L_rod1 / dx) + 1
    Nx_rod2 = int(L_rod2 / dx) + 1
    Nx_rod3 = int(L_rod3 / dx) + 1
    Nx_rod4 = int(L_rod4 / dx) + 1
    Nx_rod5 = int(L_rod5 / dx) + 1
    Nt = int(duration / dt) + 1

    # Define spatial coordinates for each rod
    x_values_rod1 = np.linspace(0, L_rod1, Nx_rod1)
    x_values_rod2 = np.linspace(L_rod1, L_rod1 + L_rod2, Nx_rod2)
    x_values_rod3 = np.linspace(L_rod1 + L_rod2, L_rod1 + L_rod2 + L_rod3, Nx_rod3)
    x_values_rod4 = np.linspace(L_rod1 + L_rod2 + L_rod3, L_rod1 + L_rod2 + L_rod3 + L_rod4, Nx_rod4)
    x_values_rod5 = np.linspace(L_rod1 + L_rod2 + L_rod3 + L_rod4, L_rod1 + L_rod2 + L_rod3 + L_rod4 + L_rod5, Nx_rod5)
    x_values_total = np.concatenate((x_values_rod1, x_values_rod2, x_values_rod3, x_values_rod4, x_values_rod5))

    # Initialize temperature arrays for each rod
    T_rod1 = np.zeros((Nt, Nx_rod1))
    T_rod2 = np.zeros((Nt, Nx_rod2))
    T_rod3 = np.zeros((Nt, Nx_rod3))
    T_rod4 = np.zeros((Nt, Nx_rod4))
    T_rod5 = np.zeros((Nt, Nx_rod5))

    # Initial conditions for each rod
    T_rod1[0, :] = -600 * (x_values_rod2 - 1.5)**2 + 600
    T_rod2[0, :] = -800 * (x_values_rod1 - 0.5)**2 + 1000
    T_rod3[0, :] = -600 * (x_values_rod2 - 1.5)**2 + 600
    T_rod4[0, :] = -800 * (x_values_rod1 - 0.5)**2 + 1000
    T_rod5[0, :] = -600 * (x_values_rod2 - 1.5)**2 + 600

    # Boundary conditions
    T_rod1[:, 0] = T_rod1[:, 1]  # Isolated boundary for rod 1
    T_rod2[:, -1] = T_rod2[:, -2]  # Isolated boundary for rod 2
    T_rod3[:, -1] = T_rod3[:, -2]  # Isolated boundary for rod 3
    T_rod4[:, -1] = T_rod4[:, -2]  # Isolated boundary for rod 4
    T_rod5[:, -1] = T_rod5[:, -2]  # Isolated boundary for rod 5

    # Finite difference method for each rod
    for n in (range(0, Nt - 1)):
        for i in range(1, Nx_rod1 - 1):
            T_rod1[n + 1, i] = T_rod1[n, i] + alpha_rod1 * dt / dx**2 * (T_rod1[n, i + 1] - 2 * T_rod1[n, i] + T_rod1[n, i - 1])

        for i in range(1, Nx_rod2 - 1):
            T_rod2[n + 1, i] = T_rod2[n, i] + alpha_rod2 * dt / dx**2 * (T_rod2[n, i + 1] - 2 * T_rod2[n, i] + T_rod2[n, i - 1])
        
        for i in range(1, Nx_rod3 - 1):
            T_rod3[n + 1, i] = T_rod3[n, i] + alpha_rod3 * dt / dx**2 * (T_rod3[n, i + 1] - 2 * T_rod3[n, i] + T_rod3[n, i - 1])

        for i in range(1, Nx_rod4 - 1):
            T_rod4[n + 1, i] = T_rod4[n, i] + alpha_rod4 * dt / dx**2 * (T_rod4[n, i + 1] - 2 * T_rod4[n, i] + T_rod4[n, i - 1])

        for i in range(1, Nx_rod5 - 1):
            T_rod5[n + 1, i] = T_rod5[n, i] + alpha_rod5 * dt / dx**2 * (T_rod5[n, i + 1] - 2 * T_rod5[n, i] + T_rod5[n, i - 1])

        # Isolated boundary conditions for both rods
        T_rod1[n + 1, 0] = T_rod1[n, 1]
        T_rod2[n + 1, -1] = T_rod2[n, -2]
        T_rod3[n + 1, -1] = T_rod3[n, -2]
        T_rod4[n + 1, -1] = T_rod4[n, -2]
        T_rod5[n + 1, -1] = T_rod5[n, -2]

        # Boundary condition at the junction of the two rods
        T_rod1[n + 1, -1] = (k_rod1*T_rod1[n+1, -2] + k_rod2*T_rod2[n+1, 1]) / (k_rod1 + k_rod2)
        T_rod2[n + 1, 0] = T_rod1[n + 1, -1]
        T_rod2[n + 1, -1] = (k_rod2*T_rod2[n+1, -2] + k_rod3*T_rod3[n+1, 1]) / (k_rod2 + k_rod3)
        T_rod3[n + 1, 0] = T_rod2[n + 1, -1]
        T_rod3[n + 1, -1] = (k_rod3*T_rod3[n+1, -2] + k_rod4*T_rod4[n+1, 1]) / (k_rod3 + k_rod4)
        T_rod4[n + 1, 0] = T_rod3[n + 1, -1]
        T_rod4[n + 1, -1] = (k_rod4*T_rod4[n+1, -2] + k_rod5*T_rod5[n+1, 1]) / (k_rod4 + k_rod5)
        T_rod5[n + 1, 0] = T_rod4[n + 1, -1]
        
    # Combine the temperatures of the two rods
    T_total = np.concatenate((T_rod1, T_rod2, T_rod3, T_rod4, T_rod5), axis=1)

    return T_total, x_values_total, duration, Nt

T_total, x_values_total, duration, Nt = my_code()

# Create a 3D surface plot for the total temperature
X_total, T_values_total = np.meshgrid(x_values_total, np.linspace(0, duration, Nt))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_total, T_values_total, T_total, cmap='viridis')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Temperature (C)')
ax.set_title('Heat Diffusion in Several Connected Rods of Different Materials')
plt.savefig('Heat_diffusion_in_metal_rods/Plots/HeatDiff_several_connected_rods_3D.png', dpi=300)

# # Create a 2D plot for Temperature vs Length for every (Nt/10)th time step
# plt.figure(figsize=(12, 8))
# for i in range(0, Nt, int(Nt/10)):
#     plt.plot(x_values_total, T_total[i, :], label='t = ' + str(i*duration/(Nt-1)) + ' s')
# plt.xlabel('Distance (m)')
# plt.ylabel('Temperature (C)')
# plt.title('Heat Diffusion in Several Connected Rods of Different Materials')
# plt.legend()
# plt.savefig('Heat_diffusion_in_metal_rods/Plots/HeatDiff_several_connected_rods.png', dpi=300)

plt.show()

