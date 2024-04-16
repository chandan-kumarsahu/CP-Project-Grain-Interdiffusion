# Importing required libraries
from grain_diffusion_code_pack import *


# Function to calculate the diffusion coefficient from best fit parameters  obtained from PS2
def get_Diff(T, P, f_O2, X_Fe):
    log_D0 = -9.21 - (201000 + (P-1e5)*7e-6)/(2.303*8.314*T) + 1/6*np.log(f_O2/1e7) + 3*(X_Fe-0.1)
    return np.exp(log_D0)

# Load the data
DATA = np.loadtxt('/home/ws1/Computational-Physics-Term-Paper-Project/Element_Partitioning_and_Diffusion/Data_files/PS3_SanPedro.csv', delimiter="\t", skiprows=9)
Dist = DATA[:, 0]-30
X_Fo = DATA[:, 1]

# Constants and parameters
tol = 1e-6                                                      # tolerance
t_max = 0.0044                                                  # total simulation time
L_grain1 = 30                                                   # length of the grain 1
L_grain2 = 470-30                                               # length of the grain 2
Diff = get_Diff(1083, 2e8, 1e-10, 0.1) *(3600/1e-12)            # diffusion coefficient
dt = 1e-6                                                       # time step
z_max = L_grain1 + L_grain2                                     # total length of the grain
dl = z_max / (len(Dist))/10                                     # spatial step

# Source term
def source_term(x, t):
    return 0

########################################### X_Fo ###########################################

# Initial conditions
def init_X_Fo_left(X_Fo):
    # Initial condition for at the left side of the data
    return X_Fo[0]

def init_X_Fo_right(X_Fo):
    # Initial condition for at the right side of the data
    return np.average(X_Fo[-30:-1])

solution_Fo, spatial_grid, time_grid = crank_nicolson_diffusion(L_grain1, L_grain2, t_max, dt, Diff, Diff, X_Fo, 
                                                                init_X_Fo_left, init_X_Fo_right, source_term, 
                                                                diff_matrix_isolated_boundary_G2)

# Calculate Root Mean Square Error
rmse = np.sqrt(np.sum(((X_Fo - solution_Fo[:, -1])**2))/len(X_Fo))
print('Root Mean Square Error:', rmse)

# Plot the diffusion equation solution
plt.figure(figsize=(5, 4))
plt.plot(Dist, X_Fo, 'o', label='Data')
plt.plot(spatial_grid, solution_Fo[:, -1], linewidth=3, label=f'time = {time_grid[-1]:.1f}')
plt.xlabel(r'Grain length ($\mu m$)')
plt.ylabel(r'X_Fo concentration')
plt.title('Diffusion and element partitioning in one grain')
plt.grid()
plt.legend()

plt.show()
