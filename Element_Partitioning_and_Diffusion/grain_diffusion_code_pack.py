# Importing required libraries
import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import chisquare, pearsonr


def get_identity(n):
    """
    Function to produce n x n identity matrix

    Parameters:
    - n: Size of matrix

    Returns:
    - Identity matrix
    """

    I=[[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        I[i][i]=1
    return I



def partial_pivot_LU (mat, vec, n):
    """
    Function for partial pivot for LU decomposition

    Parameters:
    - mat: Matrix
    - vec: Vector
    - n: Number of rows

    Returns:
    - Matrix and vector after partial pivot
    """

    for i in range (n-1):
        if mat[i][i] ==0:
            for j in range (i+1,n):
                # checks for max absolute value and swaps rows 
                # of both the input matrix and the vector as well
                if abs(mat[j][i]) > abs(mat[i][i]):
                    mat[i], mat[j] = mat[j], mat[i]
                    vec[i], vec[j] = vec[j], vec[i]
    return mat, vec



def LU_doolittle(mat,n):
    """
    LU decomposition using Doolittle's condition L[i][i]=1 without making separate L and U matrices

    Parameters:
    - mat: Matrix
    - n: Number of rows

    Returns:
    - LU decomposition of the matrix
    """

    for i in range(n):
        for j in range(n):
            if i>0 and i<=j: # changing values of upper triangular matrix
                sum=0
                for k in range(i):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=mat[i][j]-sum
            if i>j: # changing values of lower triangular matrix
                sum=0
                for k in range(j):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=(mat[i][j]-sum)/mat[j][j]
    return mat



def for_back_subs_doolittle(mat,n,vect):
    """
    Function to find the solution matrix provided a vector using forward and backward substitution respectively

    Parameters:
    - mat: Matrix
    - n: Number of rows
    - vect: Vector in RHS

    Returns:
    - Solution vector
    """

    # initialization
    y=[0 for i in range(n)]
    
    # forward substitution
    y[0]=vect[0]
    for i in range(n):
        sum=0
        for j in range(i):
            sum+=mat[i][j]*y[j]
        y[i]=vect[i]-sum
    
    # backward substitution
    vect[n-1]=y[n-1]/mat[n-1][n-1]
    for i in range(n-1,-1,-1):
        sum=0
        for j in range(i+1,n):
            sum+=mat[i][j]*vect[j]
        vect[i]=(y[i]-sum)/mat[i][i]
    del(y)
    return vect



def inverse_by_lu_decomposition (matrix, n):
    """
    Find the solution matrix using forward and backward substitution by LU decomposition

    Parameters:
    - mat: Matrix
    - n: Number of rows

    Returns:
    - Inverse of the matrix
    """

    identity=get_identity(n)
    x=[]
    
    '''
    The inverse finding process could have been done using 
    a loop for the four columns. But while partial pivoting, 
    the rows of final inverse matrix and the vector both are 
    also interchanged. So it is done manually for each row and vector.
    
    deepcopy() is used so that the original matrix doesn't change on 
    changing the copied entities. We reuire the original multiple times here
    
    1. First the matrix is deepcopied.
    2. Then partial pivoting is done for both matrix and vector.
    3. Then the decomposition algorithm is applied.
    4. Then solution is obtained.
    5. And finally it is appended to a separate matrix to get the inverse.
    Note: The final answer is also deepcopied because there is some error 
        due to which all x0, x1, x2 and x3 are also getting falsely appended.
    '''
    
    for i in range(n):
        matrix = copy.deepcopy(matrix)
        partial_pivot_LU(matrix, identity[i], n)
        matrix = LU_doolittle(matrix, n)
        x0 = for_back_subs_doolittle(matrix, n, identity[i])
        x.append(copy.deepcopy(x0))

    # The x matrix to be transposed to get the inverse in desired form
    inverse = np.transpose(x)
    return (inverse)



def polynomial_fitting(X, Y, order):
    """
    Function to fit a polynomial of given order to the given data points

    Parameters:
    - X: X values
    - Y: Y values
    - order: Order of the polynomial

    Returns:
    - Coefficients of the polynomial
    - Covariance matrix
    """

    X1 = copy.deepcopy(X)
    Y1 = copy.deepcopy(Y)
    order+=1
    
    # Finding the coefficient matrix - refer notes
    A = np.zeros((order, order))
    vector = np.zeros(order)

    for i in range(order):
        for j in range(order):
            A[i, j] = np.sum(np.power(X, i + j))

    Det = np.linalg.det(A)

    # print("Determinant is = " + str(Det))
    if Det == 0:
        print("Determinant is zero. Inverse does not exist")
    # else:
    #     print("Determinant is not zero. Inverse exists.\n")

    # Finding the coefficient vector - refer notes
    for i in range(order):
        vector[i] = np.sum(np.power(X, i) * Y)

    # Solution finding using LU decomposition using Doolittle's condition L[i][i]=1
    # partial pivoting to avoid division by zero at pivot place
    A, vector = partial_pivot_LU(A, vector, order)
    A = LU_doolittle(A,order)
    
    # Finding coefficient vector
    solution = for_back_subs_doolittle(A,order,vector)

    A_inv = inverse_by_lu_decomposition(A, len(A))

    return np.array(solution[0:order]), np.array(A_inv)



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

    return np.array(A), np.array(B)



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
    N1 = int(L_grain1 / (L_grain1 + L_grain2) * len(X))     # Number of points in the first grain
    N2 = len(X) - N1                                        # Number of points in the second grain
    N = N1 + N2                                             # Total number of points
    dl = (L_grain1+L_grain2) / (N)                  # Spatial step size
    x = [i*dl for i in range(N)]                    # Spatial points
    t = [j*dt for j in range(int(t_max/dt))]        # Time points

    alpha_1 = Diff_1 * dt / (dl**2)         # Diffusion coefficient for the first grain
    alpha_2 = Diff_2 * dt / (dl**2)         # Diffusion coefficient for the second grain

    # Initialize temperature array
    Temp = np.zeros((len(x), len(t)))

    # Initial condition
    for i in range(N1):
        Temp[i][0] = init_cond_1(X)
    for i in range(N1, len(x)):
        Temp[i][0] = init_cond_2(X)

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = boundary(N1, len(x), alpha_1, alpha_2)

    # Solve the diffusion equation using the Crank-Nicolson method
    for j in range(1, len(t)):
        source_vector = np.array([source_term(xi, t[j]) for xi in x])
        Temp[:, j] = np.linalg.solve(A, np.dot(B, Temp[:, j - 1]) + dt * source_vector)

    return Temp, np.array(x), np.array(t)



def plot_diff(time_grid, spatial_grid, solution_G1, X_G1, Dist, solution_G2=None, X_G2=None):
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

    # Create plots
    if X_G2 is None and solution_G2 is None:
        plt.figure(figsize=(6, 3.5))
        plt.subplot(1, 1, 1)
        plt.plot(Dist, X_G1, 'bo', markersize=3, label='Data')
        plt.plot(spatial_grid, solution_G1[:, -1], 'r', label='model fit')
        plt.xlabel(r'Grain length ($\mu m$)')
        plt.ylabel(r'Mg concentration')
        plt.title('Diffusion and element partitioning in Magnesium')
        plt.grid()
        plt.legend()

    else:
        plt.figure(figsize=(12, 3.5))
        plt.subplot(1, 2, 1)
        plt.plot(Dist, X_G1, 'bo', markersize=3, label='Data')
        plt.plot(spatial_grid, solution_G1[:, -1], 'r', label='model fit')
        plt.xlabel(r'Grain length ($\mu m$)')
        plt.ylabel(r'Mg concentration')
        plt.title('Diffusion and element partitioning in Magnesium')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(Dist, X_G2, 'bo', markersize=3, label='Data')
        plt.plot(spatial_grid, solution_G2[:, -1], 'r', label='model fit')
        plt.xlabel(r'Grain length ($\mu m$)')
        plt.ylabel(r'Fe concentration')
        plt.title('Diffusion and element partitioning in one Iron')
        plt.grid()
        plt.legend()

    plt.tight_layout()



def find_min_solution(f, a, b, tol=1e-6, max_iter=100):
    """
    Golden section search algorithm for maximizing a univariate function.

    Parameters:
        f (callable): The objective function.
        a (float): The lower bound of the search interval.
        b (float): The upper bound of the search interval.
        tol (float): Tolerance for stopping criterion (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        float: The maximum value of the function.
        float: The value of the argument at the maximum.
    """

    # define the new bounds based on Golden ratio
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio
    c = b - (b - a) / phi
    d = a + (b - a) / phi

    # iterate until the interval is small enough or the maximum number of iterations is reached
    while abs(c - d) > tol and max_iter > 0:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # define the new bounds based on Golden ratio
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        max_iter -= 1
    return (b + a) / 2



