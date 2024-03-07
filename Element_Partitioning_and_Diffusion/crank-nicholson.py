import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_matrix_heat_diff(N, sigma):
    A = [[0 for j in range(N)] for k in range(N)]
    B = [[0 for j in range(N)] for k in range(N)]

    for i in range(0, N):
        A[i][i] = 1 + 2*sigma
        B[i][i] = 1 - 2*sigma
        if i > 0:
            A[i][i-1] = -sigma
            B[i][i-1] = sigma
        if i < N-1:
            A[i][i+1] = -sigma
            B[i][i+1] = sigma

    return A, B

def crank_nicolson_heat_diffusion(L, T, dx, dt, Diff):
    """
    Solve 1D heat diffusion equation using Crank-Nicolson method.

    Parameters:
    - L: Length of the rod
    - T: Total time
    - dx: Spatial step size
    - dt: Time step size
    - Diff: Thermal diffusivity

    Returns:
    - u: Temperature distribution over space and time
    - x: Spatial grid
    - t: Time grid
    """

    alpha = Diff * dt / (2 * dx**2)

    # Spatial grid
    x = [i*dx for i in range(int(L/dx)+1)]
    t = [i*dt for i in range(int(T/dt)+1)]

    # Initialize temperature array
    Temp = [[0 for j in range(len(x))] for k in range(int(T/dt)+1)]

    # Initial condition
    for i in range(len(x)):
        Temp[0][i] = 4*x[i] - x[i]**2/2

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = get_matrix_heat_diff(len(x), alpha)

    Temp = np.array(Temp)
    A = np.array(A)
    B = np.array(B)

    for i in range(1, int(T/dt)+1):
        Temp[i, :] = np.linalg.solve(A, np.dot(B, Temp[i - 1, :]))

    return Temp, x, t

# Parameters
L = 8.0         # Length of the rod
T = 5.0         # Total time
dx = 0.1        # Spatial step size
dt = 0.01       # Time step size
alpha = 4       # Thermal diffusivity

# Solve the heat diffusion equation
solution, spatial_grid, time_grid = crank_nicolson_heat_diffusion(L, T, dx, dt, alpha)

# Plot the results
plt.figure(figsize=(8, 6))
plt.imshow(solution, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='hot')
plt.colorbar(label='Temperature')
plt.title('Heat Diffusion (Crank-Nicolson Method)')
plt.xlabel('Spatial Position')
plt.ylabel('Time')
plt.show()
