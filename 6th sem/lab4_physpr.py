# import numpy as np
# import matplotlib.pyplot as plt

# def explicit_scheme(Nx, Nt, L, T, c, alpha1, beta1):
#     dx = L / Nx
#     dt = T / Nt
#     sigma = c * dt / dx  # CFL condition
    
#     u = np.ones((Nt + 1, Nx + 1))
#     u[:, 0] = 2  # Boundary condition at x=0
#     u[:, -1] = 1  # Boundary condition at x=L
    
#     for n in range(0, Nt):
#         for i in range(1, Nx):
#             u[n+1, i] = u[n, i] - sigma * (u[n, i] - u[n, i-1])
    
#     return u

# def implicit_scheme(Nx, Nt, L, T, c, alpha1, beta1):
#     dx = L / Nx
#     dt = T / Nt
#     sigma = c * dt / dx
    
#     A = np.zeros((Nx-1, Nx-1))
#     B = np.zeros(Nx-1)
#     u = np.ones((Nt+1, Nx+1))
#     u[:, 0] = 2  # Boundary condition at x=0
#     u[:, -1] = 1  # Boundary condition at x=L
    
#     for i in range(Nx-1):
#         if i > 0:
#             A[i, i-1] = -sigma / 2
#         A[i, i] = 1
#         if i < Nx-2:
#             A[i, i+1] = sigma / 2
    
#     for n in range(0, Nt):
#         B[:] = u[n, 1:Nx]
#         u[n+1, 1:Nx] = np.linalg.solve(A, B)
    
#     return u

# def plot_solution(u, L, T, title):
#     plt.imshow(u, aspect='auto', cmap='hot', origin='lower',
#                extent=[0, L, 0, T])
#     plt.colorbar(label='u(x,t)')
#     plt.xlabel('Position x')
#     plt.ylabel('Time t')
#     plt.title(title)
#     plt.show()

# # Parameters
# Nx, Nt = 20, 100  # Number of space and time steps
# L, T = 1, 1  # Length and total time
# c = 1  # Given velocity
# alpha1 = 0  # Given parameter
# beta1 = 2  # Given parameter

# # Compute solutions
# u_explicit = explicit_scheme(Nx, Nt, L, T, c, alpha1, beta1)
# u_implicit = implicit_scheme(Nx, Nt, L, T, c, alpha1, beta1)

# # Plot results
# plot_solution(u_explicit, L, T, "Explicit Scheme")
# plot_solution(u_implicit, L, T, "Implicit Scheme")
# import numpy as np
# import matplotlib.pyplot as plt

# def initial(x):
#     return 1

# def explicit_scheme(Nx, Nt, L, T, c):
#     dx = L / Nx
#     dt = T / Nt
#     sigma = c * dt / dx  
    
#     u = np.ones((Nt + 1, Nx + 1))
#     u[:, 0] = 2  # Boundary condition at x=0
#     u[:, -1] = 1  # Boundary condition at x=L
    
#     for n in range(0, Nt):
#         for i in range(1, Nx):
#             u[n+1, i] = u[n, i] - sigma * (u[n, i] - u[n, i-1])
    
#     return u

# def implicit_scheme(Nx, Nt, L, T, c):
#     dx = L / Nx
#     dt = T / Nt
#     sigma = c * dt / dx
    
#     A = -1 / (12 * dt)
#     B = 8 / (12 * dt) + 1 / dx
#     C = 5 / (12 * dt) - 1 / dx
    
#     alphas = [0]
#     betas = [2]
#     u = np.ones((Nt + 1, Nx + 1))
#     u[:, 0] = 2  # Boundary condition at x=0
#     u[:, -1] = 1  # Boundary condition at x=L
    
#     for i in range(Nx):
#         alphas.append(-A / (B + C * alphas[i]))
#         betas.append((initial(i * dx) - C * betas[i]) / (B + C * alphas[i]))
    
#     P = np.zeros(Nx + 1)
#     P[Nx] = 1
#     for i in range(Nx-1, -1, -1):
#         P[i] = alphas[i+1] * P[i+1] + betas[i+1]
    
#     A_matrix = np.zeros((Nx-1, Nx-1))
#     B_vector = np.zeros(Nx-1)
    
#     for i in range(Nx-1):
#         if i > 0:
#             A_matrix[i, i-1] = -sigma / 2
#         A_matrix[i, i] = 1
#         if i < Nx-2:
#             A_matrix[i, i+1] = sigma / 2
    
#     for n in range(0, Nt):
#         B_vector[:] = P[1:Nx]
#         P[1:Nx] = np.linalg.solve(A_matrix, B_vector)
#         u[n+1, 1:Nx] = P[1:Nx]
    
#     return u

# def plot_solution_line(u_explicit, u_implicit, L):
#     x = np.linspace(0, L, u_explicit.shape[1])
#     plt.plot(x, u_explicit[-1, :], label='Explicit Scheme', linestyle='--')
#     plt.plot(x, u_implicit[-1, :], label='Implicit Scheme', linestyle='-')
#     plt.xlabel('Position x')
#     plt.ylabel('u(x,T)')
#     plt.title('Comparison of Explicit and Implicit Schemes at Final Time Step')
#     plt.legend()
#     plt.grid()
#     plt.show()


# Nx, Nt = 20, 100  # Number of space and time steps
# L, T = 1, 1  # Length and total time
# c = 1  # Given velocity

# # Compute solutions
# u_explicit = explicit_scheme(Nx, Nt, L, T, c)
# u_implicit = implicit_scheme(Nx, Nt, L, T, c)

# # Plot results as line graph
# plot_solution_line(u_explicit, u_implicit, L)



# import numpy as np
# import matplotlib.pyplot as plt

# def explicit_scheme(Nx, Nt, L, T, c):
#     dx = L / Nx
#     dt = T / Nt
#     sigma = c * dt / dx  # CFL condition
    
#     u = np.ones((Nt + 1, Nx + 1))
#     u[:, 0] = 2  # Boundary condition at x=0
#     u[:, -1] = 1  # Boundary condition at x=L
    
#     for n in range(0, Nt):
#         for i in range(1, Nx):
#             u[n+1, i] = u[n, i] - sigma * (u[n, i] - u[n, i-1])
    
#     return u

# def implicit_scheme(Nx, Nt, L, T, c):
#     dx = L / Nx
#     dt = T / Nt
#     sigma = c * dt / dx
    
#     A = np.zeros((Nx-1, Nx-1))
#     B = np.zeros(Nx-1)
#     u = np.ones((Nt+1, Nx+1))
#     u[:, 0] = 2  # Boundary condition at x=0
#     u[:, -1] = 1  # Boundary condition at x=L
    
#     for i in range(Nx-1):
#         if i > 0:
#             A[i, i-1] = -sigma / 2
#         A[i, i] = 1
#         if i < Nx-2:
#             A[i, i+1] = sigma / 2
    
#     for n in range(0, Nt):
#         B[:] = u[n, 1:Nx]
#         u[n+1, 1:Nx] = np.linalg.solve(A, B)
    
#     return u

# def plot_solution_line(u_explicit, u_implicit, L):
#     x = np.linspace(0, L, u_explicit.shape[1])
#     plt.plot(x, u_explicit[-1, :], label='Explicit Scheme', linestyle='--')
#     plt.plot(x, u_implicit[-1, :], label='Implicit Scheme', linestyle='-')
#     plt.xlabel('Position x')
#     plt.ylabel('u(x,T)')
#     plt.title('Comparison of Explicit and Implicit Schemes at Final Time Step')
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Parameters
# Nx, Nt = 50, 200  # Adjusted resolution
# L, T = 1, 1  # Length and total time
# c = 1  # Given velocity

# # Compute solutions
# u_explicit = explicit_scheme(Nx, Nt, L, T, c)
# u_implicit = implicit_scheme(Nx, Nt, L, T, c)

# # Plot results as line graph
# plot_solution_line(u_explicit, u_implicit, L)


import numpy as np
import matplotlib.pyplot as plt

def solve(n=101, max_iter=1000, iterations_to_plot=[100]):
# Явный
    dx = 1.0 / (n - 1)
    dt = 0.01 * dx / (2.0 )  # C = 1.0
    eps = 0.00001
    
    oldP1 = np.zeros(n)
    newP1 = np.zeros(n)
    for i in range(n):
        oldP1[i] = i + 1    
    oldP1[0] = newP1[0] = 1.0
    oldP1[-1] = newP1[-1] = 2

    iter_count = 0
    max_diff = eps + 1
    
    x_vals = np.linspace(0, 1, n)
    stored_results = {}
    
    while iter_count < max_iter and max_diff > eps:
        for i in range(1, n - 1):
            newP1[i] = oldP1[i] + 2* dt * ((oldP1[i+1] - oldP1[i]) / dx)
        
        max_diff = np.max(np.abs(newP1 - oldP1))
        oldP1[:] = newP1[:]
        iter_count += 1
        
        if iter_count in iterations_to_plot:
            stored_results[iter_count] = newP1.copy()
            u_n=stored_results[iter_count]
    
# ТОМАС
    C=-2
    xlist = [i * dx for i in range(n)]
    d = np.zeros(n)
    alfa = np.zeros(n)
    betta = np.zeros(n)
    newP = np.zeros(n)
    oldP = np.zeros(n)
    for i in range(n):
        oldP[i] = i + 1
    a = 5/ (12 * dt) + C / dx
    b = 8 / (12 * dt) - C / dx 
    c = -1/ (12 * dt) 
    iter_count=0
    while iter_count <=100:
        for i in range(1, n-1):
            d[i] = (-1 * oldP[i-1] / 12 + 8 * oldP[i]/12 +5* oldP[i + 1] / 12)/(dt)
        alfa[0] = 0.0
        betta[0] = 1.0 
        
        for i in range(1, n-1):
            alfa[i+1] = -a/(b+c*alfa[i])
            betta[i+1] = (d[i]-c*betta[i])/(b+c*alfa[i])
        newP[n-1] = 2.0
        
        for i in range(n-2, -1, -1):
            newP[i] = alfa[i+1]*newP[i+1] + betta[i+1]
        max_diff = 0.0
        for i in range(n):
            if max_diff < abs(newP[i]-oldP[i]):
                max_diff = abs(newP[i]-oldP[i])
        for i in range(n):
            oldP[i] = newP[i]
        
        iter_count += 1
    error=0
    for i in range(n):
        if(abs(newP[i]-newP1[i])>error):
            error=abs(newP[i]-u_n[i])
    plt.figure(figsize=(8, 6))
    for iters, values in stored_results.items():
        plt.plot(x_vals, values, label=f'Explicit')
        plt.plot(xlist, newP, label=f'Thomas Algorithm')
    plt.xlabel('X')
    plt.ylabel('P')
    plt.legend()
    plt.grid()
    plt.show()

solve()