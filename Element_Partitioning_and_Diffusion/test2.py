# Importing required libraries
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import chisquare, pearsonr


def diff_matrix_isolated_boundary_G2(N1, N, alpha_1, alpha_2):
    """
    Create the matrices A and B for the Crank-Nicolson method with isolated boundary conditions
    for two grains.

    Parameters:
        N1 (int): Number of points in the first grain.
        N (int): Total number of points.
        alpha_1 (float): Diffusion coefficient for the first grain.
        alpha_2 (float): Diffusion coefficient for the second grain.

    Returns:
        tuple: Two matrices A and B.
    """

    # Initialize matrices A and B with zeros
    A = [[0] * N for _ in range(N)]
    B = [[0] * N for _ in range(N)]

    # Fill diagonal and off-diagonal values for matrices A and B
    for i in range(N1):
        A[i][i] = 2 + 2 * alpha_1  # Diagonal element of A
        B[i][i] = 2 - 2 * alpha_1  # Diagonal element of B

        # Connect to the left neighbor (if not on the left edge)
        if i > 0:
            A[i][i - 1] = -alpha_1
            B[i][i - 1] = alpha_1

        # Connect to the right neighbor (if not on the right edge)
        if i < N - 1:
            A[i][i + 1] = -alpha_1
            B[i][i + 1] = alpha_1

    # Fill diagonal and off-diagonal values for matrices A and B
    for i in range(N1, N):
        A[i][i] = 2 + 2 * alpha_2  # Diagonal element of A
        B[i][i] = 2 - 2 * alpha_2  # Diagonal element of B

        # Connect to the left neighbor (if not on the left edge)
        if i > 0:
            A[i][i - 1] = -alpha_2
            B[i][i - 1] = alpha_2

        # Connect to the right neighbor (if not on the right edge)
        if i < N - 1:
            A[i][i + 1] = -alpha_2
            B[i][i + 1] = alpha_2


    # Boundary conditions
    A[0][0] = 2 + alpha_1
    B[0][0] = 2 - alpha_1
    A[-1][-1] = 2 + alpha_2
    B[-1][-1] = 2 - alpha_2

    return A, B


def crank_nicolson_diffusion(L_grain1, L_grain2, t_max, dt, Diff_1, Diff_2, X, init_cond_1, init_cond_2, source_term, boundary):
    """
    Solve the diffusion equation using the Crank-Nicolson method.

    Parameters:
        L_grain1 (float): Length of the first grain.
        L_grain2 (float): Length of the second grain.
        t_max (float): Maximum time.
        dl (float): Spatial step size.
        dt (float): Temporal step size.
        Diff_1 (float): Diffusion coefficient for the first grain.
        Diff_2 (float): Diffusion coefficient for the second grain.
        X (ndarray): Array of spatial points.
        init_cond_1 (callable): Initial condition for the first grain.
        init_cond_2 (callable): Initial condition for the second grain.
        source_term (callable): Source term function.
        boundary (callable): Function to create the matrices A and B.

    Returns:
        ndarray: Solution of the diffusion equation.
        ndarray: Array of spatial points.
        ndarray: Array of time points.
    """

    # Spatial grid
    N1 = int(L_grain1 / (L_grain1 + L_grain2) * len(X))
    N2 = len(X) - N1
    N = N1 + N2
    dl = (L_grain1+L_grain2) / (N)
    x = [i*dl for i in range(N)]
    t = [j*dt for j in range(int(t_max/dt))]

    alpha_1 = Diff_1 * dt / (dl**2)
    alpha_2 = Diff_2 * dt / (dl**2)

    # Initialize temperature array
    Temp = np.zeros((len(x), len(t)))

    # Initial condition
    for i in range(N1):
        Temp[i][0] = init_cond_1(X)
    for i in range(N1, len(x)):
        Temp[i][0] = init_cond_2(X)

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = boundary(N1, len(x), alpha_1, alpha_2)

    A = np.array(A)
    B = np.array(B)

    for j in range(1, len(t)):
        source_vector = np.array([source_term(xi, t[j]) for xi in x])
        Temp[:, j] = np.linalg.solve(A, np.dot(B, Temp[:, j - 1]) + dt * source_vector)

    return Temp, np.array(x), np.array(t)

def plot_diff(time_grid, spatial_grid, solution_Mg, Dist, X_Mg):
    """
    Plot the solution of the diffusion equation.

    Parameters:
        time_grid (ndarray): Array of time points.
        spatial_grid (ndarray): Array of spatial points.
        solution (ndarray): Solution of the diffusion equation.
        Dist (ndarray): Distance data.
        X_Mg (ndarray): Mg concentration data.
        X_Fe (ndarray): Fe concentration data.
    """

    # Create 2D plots
    plt.figure(figsize=(5, 4))
    plt.plot(Dist, X_Mg, 'o', label='Data')
    plt.plot(spatial_grid, solution_Mg[:, -1], linewidth=3, label=f'time = {time_grid[-1]:.1f}')
    plt.xlabel(r'Grain length ($\mu m$)')
    plt.ylabel(r'X_Mg concentration')
    plt.title('Diffusion and element partitioning in one grain')
    plt.grid()
    plt.legend()

def find_max_solution(f, a, b, tol=1e-6, max_iter=100):
    """
    Golden section search algorithm for maximizing a univariate function.

    Parameters:
        f (callable): The objective function.
        a (float): The lower bound of the search interval.
        b (float): The upper bound of the search interval.
        tol (float): Tolerance for stopping criterion (default: 1e-5).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        float: The maximum value of the function.
        float: The value of the argument at the maximum.
    """

    phi = (1 + 5 ** 0.5) / 2  # Golden ratio
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    while abs(c - d) > tol and max_iter > 0:
        if f(c) > f(d):  # Change the comparison to '>' for maximization
            b = d
        else:
            a = c
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        max_iter -= 1
    return (b + a) / 2, f((b + a) / 2)


DATA = np.loadtxt('/home/ws1/Computational-Physics-Term-Paper-Project/Element_Partitioning_and_Diffusion/PS3_Ol6a.csv', delimiter="\t", skiprows=8)
Dist = DATA[9:, 0]-75
Fo = DATA[9:, 1]
print(Dist.shape, Fo.shape)

# Constants and parameters
tol = 1e-6
t_max = 557     # total simulation time
L_grain1 = 125     # length of the grain 1
L_grain2 = 330-125     # length of the grain 2
Diff = 0.5
dt = 0.5       # time step
z_max = L_grain1 + L_grain2
dl = z_max / (len(Dist))

def source_term(x, t):
    return 0

########################################### X_Mg ###########################################

def init_X_Mg_left(X_Mg):
    # Initial condition for at the left side of the Magnesium data - average of the first 20 data points
    return np.average(X_Mg[:5])

def init_X_Mg_right(X_Mg):
    # Initial condition for at the right side of the Magnesium data - average of the last 20 data points
    return np.average(X_Mg[-5:-1])

# Function to calculate the Pearson R for Magnesium
solution_Mg, spatial_grid, time_grid = crank_nicolson_diffusion(L_grain1, L_grain2, t_max, dt, Diff, Diff, Fo, 
                                                                init_X_Mg_left, init_X_Mg_right, source_term, 
                                                                diff_matrix_isolated_boundary_G2)
pearson_R_Mg = pearsonr(Fo, solution_Mg[:, -1])[0]
print('Pearson R:', pearson_R_Mg)

print()

# Plot the diffusion equation solution
plot_diff(time_grid, spatial_grid, solution_Mg, Dist, Fo)
plt.show()
