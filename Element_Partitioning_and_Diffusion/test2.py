# Importing required libraries
from code_pack import *


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

def get_Diff(T, P, f_O2, X_Fe):
    log_D0 = -9.21 - (201000 + (P-1e5)*7e-6)/(2.303*8.314*T) + 1/6*np.log(f_O2/1e7) + 3*(X_Fe-0.1)
    return np.exp(log_D0)

DATA = np.loadtxt('/home/ws1/Computational-Physics-Term-Paper-Project/Element_Partitioning_and_Diffusion/PS3_Ol6b.csv', delimiter="\t", skiprows=8)
Dist = DATA[7:, 0]-60
Fo = DATA[7:, 1]
print(Dist.shape, Fo.shape)

# Constants and parameters
tol = 1e-6
t_max = 2.02e-2     # total simulation time
L_grain1 = 143     # length of the grain 1
L_grain2 = 360-143     # length of the grain 2
Diff = get_Diff(1083, 1e6, 1e-10, 0.1) *(3600/1e-12)
dt = 1e-6       # time step
z_max = L_grain1 + L_grain2
dl = z_max / (len(Dist))/10

def source_term(x, t):
    return 0

########################################### X_Mg ###########################################

def init_X_Mg_left(X_Mg):
    # Initial condition for at the left side of the Magnesium data - average of the first 20 data points
    return np.average(X_Mg[:10])

def init_X_Mg_right(X_Mg):
    # Initial condition for at the right side of the Magnesium data - average of the last 20 data points
    return np.average(X_Mg[-10:-1])

# Function to calculate the Pearson R for Magnesium
solution_Fo, spatial_grid, time_grid = crank_nicolson_diffusion(L_grain1, L_grain2, t_max, dt, Diff, Diff, Fo, 
                                                                init_X_Mg_left, init_X_Mg_right, source_term, 
                                                                diff_matrix_isolated_boundary_G2)
# Calculate Root Mean Square Error
rmse = np.sqrt(np.sum(((Fo - solution_Fo[:, -1])**2))/len(Fo))
print('Root Mean Square Error:', rmse)

print()

# Plot the diffusion equation solution
plot_diff(time_grid, spatial_grid, solution_Fo, Dist, Fo)
plt.show()
