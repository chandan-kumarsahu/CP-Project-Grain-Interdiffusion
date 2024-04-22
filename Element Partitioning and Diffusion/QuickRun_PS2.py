# Importing required libraries
from grain_diffusion_code_pack import *

DATA = np.loadtxt('/home/ws1/Computational-Physics-Term-Paper-Project/Element_Partitioning_and_Diffusion/Data_files/PS2_OLID_10.csv', delimiter="\t", skiprows=6)
Dist = DATA[:, 0]-201
X_Mg = DATA[:, 2]
X_Fe = DATA[:, 3]

# Constants and parameters
tol = 1e-6
Diff = 0.01
t_max = 360     # total simulation time
L_grain1 = 54.5     # length of the grain 1
L_grain2 = 101-54.5     # length of the grain 2
dt = 0.5       # time step
z_max = L_grain1 + L_grain2
dl = z_max / (len(Dist))

def source_term(x, t):
    return 0

########################################### X_Mg ###########################################

def init_X_Mg_left(X_Mg):
    # Initial condition for at the left side of the Magnesium data - average of the first 20 data points
    return np.average(X_Mg[:20])

def init_X_Mg_right(X_Mg):
    # Initial condition for at the right side of the Magnesium data - average of the last 20 data points
    return np.average(X_Mg[-20:-1])

# Function to calculate the Pearson R for Magnesium
solution_Mg, spatial_grid, time_grid = crank_nicolson_diffusion(L_grain1, L_grain2, t_max, dt, Diff, Diff, X_Mg, 
                                                                init_X_Mg_left, init_X_Mg_right, source_term, 
                                                                diff_matrix_isolated_boundary_G2)
rmse = np.sqrt(np.sum(((X_Mg - solution_Mg[:, -1])**2))/len(X_Mg))
print('Root Mean Square Error:', rmse)

# plt.plot(Dist, residuals, 'o', label='Residuals')
# plt.show()

plt.plot(Dist, X_Mg, 'o', label='Data')
plt.plot(spatial_grid, solution_Mg[:, -1], linewidth=3, label=f'time = {time_grid[-1]:.1f}')
plt.xlabel(r'Grain length ($\mu m$)')
plt.ylabel(r'X_Mg concentration')
plt.title('Diffusion and element partitioning in one grain')
plt.grid()
plt.legend()

plt.show()
