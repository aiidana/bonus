
# import numpy as np
# import matplotlib.pyplot as plt


# def analytical(x: float, t: float, m: int):
#     analytical = 0
    
#     for i in range(1, m + 1):
#         an = (-4*((-1)**i -1)/(np.pi**3 *i**3)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.sin(i * np.pi * x)
#         analytical += an
#     return  analytical


# def initial(x):
#     return x-x**2


# n = 100
# m = 10
# dx = 1 / n
# dt = 0.0001


# xlist = [i * dx for i in range(n)]

# # Define the iteration limits
# iteration_limits = [10, 100, 1000]

# # Loop through each iteration limit and plot the solution
# for iter_limit in iteration_limits:
#     u_n = [initial(x) for x in xlist] 
#     u_n1 = np.zeros(n)
#     iter = 0

   
#     while iter < iter_limit:
#         for i in range(1, n - 1):
#             u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])
#         u_n1[0] =0
#         u_n1[n - 1] = 0
#         u_n = u_n1.copy()
#         iter += 1

 
#     anall_solution = [analytical(x, iter_limit * dt, m) for x in xlist]
    
#     # Plotting
#     plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
#     plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
#     plt.xlabel("x")
#     plt.ylabel("u(x)")
#     plt.legend()
#     plt.grid()
#     plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}")
#     plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x: float, t: float, m: int):
    
#     analytical = 0
#     for i in range(1, m + 1):
#         an = 32 * ((12 * (-1)**(i + 1) / ((2 * i - 1)**4 * np.pi**4)) - (3 / ((2 * i - 1)**3 * np.pi**3))) * \
#              np.exp(-(2 * i - 1)**2 * t * np.pi**2 / 4) * np.sin((2 * i - 1) * np.pi * x / 2)
#         analytical += an
#     return analytical

# def initial(x):
#     return 3 * x**2 - 2 * x**3


# n = 100 
# m = 10   
# dx = 1 / n  
# dt = 0.0001  


# xlist = [i * dx for i in range(n)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n)

# # Iterate for time steps
# # max_iter = 10  
# # max_iter = 100
# max_iter = 1000
# iter = 0

# while iter <= max_iter:
    
#     for i in range(1, n - 1):
#         u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])
    
#     # Boundary conditions
#     u_n1[0] = 0  # Dirichlet boundary condition at x = 0
#     u_n1[n - 1] = u_n1[n - 2]  # neumann boundary condition at x = 1
    
    
#     u_n = u_n1.copy()
#     iter += 1


# final_time = max_iter * dt
# anall_solution = [analytical(xi, final_time, m) for xi in xlist]

# # Plot the results
# plt.plot(xlist, u_n, label="Numerical solution")
# plt.plot(xlist, anall_solution, color='pink', label="Analytical solution")
# plt.xlabel('x')
# plt.ylabel('u(x,t)')
# plt.legend()
# plt.grid(True)
# plt.title("Numerical vs Analytical Solution")
# plt.show()




import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float, t: float, m: int):
    analytical = 0
    
    for i in range(1, m + 1):
        an = (-4*((-1)**i -1)/(np.pi**3 *i**3)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.sin(i * np.pi * x)
        analytical += an
    return  analytical


def initial(x):
    return x-x**2


n = 100
m = 10
dx = 1 / n
dt = 0.00001


xlist = [i * dx for i in range(n)]
iteration_limits = [10, 100, 10000]

# Loop through each iteration limit and plot the solution
for iter_limit in iteration_limits:
    u_n = [initial(x) for x in xlist] 
    u_n1 = np.zeros(n)
    iter = 0

    while iter < iter_limit:
        for i in range(1, n - 1):
            u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])
        u_n1[0] =0
        u_n1[n - 1] =0 
        u_n = u_n1.copy()
        iter += 1

 
    anall_solution = [analytical(x, iter_limit * dt, m) for x in xlist]

    plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
    plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid()
    plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}")
    plt.show()