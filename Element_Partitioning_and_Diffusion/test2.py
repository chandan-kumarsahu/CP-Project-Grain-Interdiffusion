import matplotlib.pyplot as plt
import numpy as np


def diff_matrix_exchange_boundary(N, sigma1, sigma2):
    # Initialize matrices A and B with zeros
    A = [[0] * N for _ in range(N)]
    B = [[0] * N for _ in range(N)]

    # Fill diagonal and off-diagonal values for matrices A and B
    for i in range(N):
        A[i][i] = 2 + 2 * sigma1  # Diagonal element of A
        B[i][i] = 2 - 2 * sigma1  # Diagonal element of B

        # Connect to the left neighbor (if not on the left edge)
        if i > 0:
            A[i][i - 1] = -sigma1
            B[i][i - 1] = sigma1

        # Connect to the right neighbor (if not on the right edge)
        if i < N - 1:
            A[i][i + 1] = -sigma1
            B[i][i + 1] = sigma1

    # Boundary conditions
    A[0][0] = 2 + sigma1 + sigma2
    B[0][0] = 2 - sigma1 - sigma2
    A[-1][-1] = 2 + sigma1 + sigma2
    B[-1][-1] = 2 - sigma1 - sigma2

    return A, B

def crank_nicolson_diffusion_two_grains(x_max, t_max, dx, dt, Diff1, Diff2, z_max1, z_max2, init_cond1, init_cond2, source_term, boundary):
    alpha1 = Diff1 * dt / (dx**2)
    alpha2 = Diff2 * dt / (dx**2) 

    # Spatial grid
    x1 = [i*dx for i in range(int(z_max1/dx)+1)]
    x2 = [z_max1 + i*dx for i in range(int(z_max2/dx)+1)]
    x = x1 + x2

    t = [j*dt for j in range(int(t_max/dt)+1)]

    # Initialize temperature array
    Temp = np.zeros((len(x), len(t)))

    # Initial conditions
    for i in range(len(x1)):
        Temp[i][0] = init_cond1(x1[i])
    for i in range(len(x2)):
        Temp[len(x1)+i][0] = init_cond2(x2[i])

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = boundary(len(x), alpha1, alpha2)

    A = np.array(A)
    B = np.array(B)

    for j in range(1, len(t)):
        source_vector = np.array([source_term(xi, t[j]) for xi in x])
        Temp[:, j] = np.linalg.solve(A, np.dot(B, Temp[:, j - 1]) + dt * source_vector)

    return Temp, np.array(x), np.array(t)

def init_cond1(x):
    return 1000*np.exp(-0.005*(x-50)**2)+500

def init_cond2(x):
    return 2000*np.exp(-0.005*(x-300)**2)

def plot_diff(time_grid, spatial_grid, solution):

    # Create 2D plots
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    for i in (range(0, len(time_grid), int(len(time_grid)/5))):
        plt.plot(spatial_grid, solution[:, i], label=f'time = {time_grid[i]:.1f}')
    plt.xlabel(r'Grain length ($\mu m$)')
    plt.ylabel(r'Concentration (ppm)')
    plt.title('Diffusion and element partitioning in one grain')
    plt.ylim(np.min(solution)-10, np.max(solution)+10)
    plt.grid()
    plt.legend()

    # Create imshow plot
    plt.subplot(1, 2, 2)
    plt.contourf(*np.meshgrid(spatial_grid, time_grid), solution.T, 40, cmap='Spectral_r')
    plt.colorbar(label=r'Concentration (ppm)')
    plt.title('Diffusion and element partitioning in one grain')
    plt.xlabel('Grain length ($\mu m$)')
    plt.ylabel('Time (s)')
    plt.grid()

def source_term(x, t):
    return 0 # 10 * np.sin(np.pi * x) * np.exp(-0.1 * t)

# Constants and parameters
alpha1 = 10    # diffusivity for grain 1
alpha2 = 1    # diffusivity for grain 2
t_max = 100     # total simulation time
z_max1 = 100    # length of grain 1
z_max2 = 200    # length of grain 2
dt = 10       # time step
dz = 0.05       # spatial step

solution, spatial_grid, time_grid = crank_nicolson_diffusion_two_grains(z_max1+z_max2, t_max, dz, dt, alpha1, alpha2, z_max1, z_max2, init_cond1, init_cond2, source_term, diff_matrix_exchange_boundary)

# Plot the diffusion equation solution
plot_diff(time_grid, spatial_grid, solution)

plt.show()
