# import numpy as np
# import matplotlib.pyplot as plt


# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     for i in range(1, m + 1):
#         an = (-4 * ((-1)**i - 1) / (i ** 3 * np.pi ** 3)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.sin(i * np.pi * x)
#         analytical += an
#     return  analytical


# def initial(x):
#     return x-x**2


# n = 100 
# m = 10   
# dx = 1 / n
# dt = 0.0001
# iter_limit = 1000  # Number of iterations 100,1000

# # Spatial grid and initial conditions
# xlist = [i * dx for i in range(n + 1)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n + 1) 
# itt = 0


# while itt < iter_limit:
#     # coefficients for the Thomas algorithm 
#     A, B, C, D = [], [], [], []
#     for i in range(1, n):  
#         A.append(-1 / dx**2)
#         B.append(1 / dt + 2 / dx**2)
#         C.append(-1 / dx**2)
#         D.append(u_n[i] / dt)
    
    
#     alphan = [0]  
#     betan = [0]  
    
#     for i in range(1, n):
#         denom = B[i-1] + C[i-1] * alphan[i-1]
#         alphan.append(-A[i-1] / denom)
#         betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

    
#     u_n1[n] = 0 

    
#     for i in range(n-1, 0, -1):
#         u_n1[i] = alphan[i] * u_n1[i + 1] + betan[i]

#     # Update old solution with new solution
#     u_n = u_n1.copy()
    
#     itt += 1


# anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist]

# # Plotting 
# plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
# plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
# plt.xlabel("x")
# plt.ylabel("u(x)")
# plt.legend()
# plt.grid()
# plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}")
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Analytical solution
# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     for i in range(1, m + 1):
#         an = (-4 * ((-1)**i - 1) / (i ** 3 * np.pi ** 3)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.sin(i * np.pi * x)
#         analytical += an
#     return analytical

# # Initial condition
# def initial(x):
#     return x - x**2

# # Parameters
# n = 100
# m = 10   
# dx = 1 / n
# dt = 0.0001
# iter_limit = 100  # Number of iterations

# # Spatial grid and initial conditions
# xlist = [i * dx for i in range(n + 1)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n + 1)  # Initialize u_n1 with size n+1
# itt = 0

# while itt < iter_limit:
#     # Thomas algorithm coefficients for internal points
#     A, B, C, D = [], [], [], []
#     for i in range(1, n):  
#         A.append(-1 / dx**2)
#         B.append(1 / dt + 2 / dx**2)
#         C.append(-1 / dx**2)
#         D.append(u_n[i] / dt)
    
#     # Thomas algorithm forward pass
#     alphan = [0]  
#     betan = [0]  
    
#     for i in range(1, n):
#         denom = B[i-1] + C[i-1] * alphan[i-1]
#         alphan.append(-A[i-1] / denom)
#         betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

#     # Boundary conditions
#     u_n1[0] = 0  # Left boundary (Dirichlet condition)
#     u_n1[n] = 0  # Right boundary (Dirichlet condition)

#     # Thomas algorithm back substitution
#     for i in range(n-1, 0, -1):
#         u_n1[i] = alphan[i] * u_n1[i + 1] + betan[i]

#     # Update old solution with new solution
#     u_n = u_n1.copy()
    
#     itt += 1

# # Analytical solution after final time iteration
# anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist]

# # Plotting the results
# plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
# plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
# plt.xlabel("x")
# plt.ylabel("u(x)")
# plt.legend()
# plt.grid()
# plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}")
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float, t: float, m: int):
    analytical = 0
    for i in range(1, m + 1):
        an = (-4 * ((-1)**i - 1) / (i ** 3 * np.pi ** 3)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.sin(i * np.pi * x)
        analytical += an
    return analytical

def initial(x):
    return x - x**2

n = 100
m = 10   
dx = 1 / n
dt = 0.0001
iter_limit = 10000 #1000,10000

xlist = [i * dx for i in range(n + 1)]
u_n = [initial(x) for x in xlist]
u_n1 = np.zeros(n + 1)
itt = 0

while itt < iter_limit:
    A, B, C, D = [], [], [], []
    for i in range(1, n):  
        A.append(-1 / dx**2)
        B.append(1 / dt + 2 / dx**2)
        C.append(-1 / dx**2)
        D.append(u_n[i] / dt)
    
    alphan = [0]  
    betan = [0]  
    
    for i in range(1, n):
        denom = B[i-1] + C[i-1] * alphan[i-1]
        alphan.append(-A[i-1] / denom)
        betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

    u_n1[0] = 0
    u_n1[n] = 0

    for i in range(n-1, 0, -1):
        u_n1[i] = alphan[i] * u_n1[i + 1] + betan[i]

    u_n = u_n1.copy()
    
    itt += 1

anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist]

plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}")
plt.show()