import numpy as np
import matplotlib.pyplot as plt


def analytical(x: float, t: float, m: int):
    analytical = 0
    a0 = 3 / 5
    for i in range(1, m + 1):
        an = (-48 * ((-1)**i  + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
        analytical += an
    return a0 + analytical


def initial(x):
    return 3*x**4 -  8*x**3 + 6*x**2


n = 100
m = 10
dx = 1/n
dt = 0.0001


xlist = [i * dx for i in range(n)]
iteration_limits = [10, 100, 1000]


for iter_limit in iteration_limits:
    u_n = [initial(x) for x in xlist] 
    u_n1 = np.zeros(n)
    iter = 0

    # Time-stepping loop up to the specified iteration limit
    while iter < iter_limit:
        for i in range(1, n - 1):
            u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])
        u_n1[0] = u_n1[1]  
        u_n1[n - 1] = u_n1[n - 2]  
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

# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     a0 = 3 / 5
#     for i in range(1, m + 1):
#         an = (-48 * ((-1) ** i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
#         analytical += an
#     return a0 + analytical

# def initial(x):
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2

# n=100
# m=10

# dx=1/n
# dt=0.0001

# xlist=[i*dx for i in range(n)]
# u_n=[initial(x) for x in xlist]
# u_n1=np.zeros(n)
# iter=0
# anall_solution=[]
# j=0
# # max_iter=10
# max_iter=100
# # max_iter=1000
# while iter <= max_iter:
#     diff = 0
#     for i in range(1, n-1):
#         u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i+1] - 2 * u_n[i] + u_n[i-1])
#     u_n1[0] = u_n1[1]  
#     u_n1[n - 1] = u_n1[n - 2]    
#     u_n = u_n1
#     iter += 1
    
    
#     anall_solution = [analytical(xi, iter * dt, m) for xi in xlist]


# plt.plot(xlist, u_n, label="Numerical solution")
# plt.plot(xlist, anall_solution, color='pink', label="Analytical solution")
# plt.xlabel('x')
# plt.ylabel('u(x,t)')
# plt.legend()
# plt.grid(True)
# plt.title("Numerical vs Analytical Solution")
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x: float, t: float, m: int):
#     """
#     Analytical solution based on Fourier series.
#     """
#     analytical = 0
#     a0 = 3 / 5  # Constant term
#     for i in range(1, m + 1):
#         an = (-48 * ((-1) ** i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
#         analytical += an
#     return a0 + analytical

# def initial(x):
#     """
#     Initial condition: u(x, 0) = 3 * x^4 - 8 * x^3 + 6 * x^2.
#     """
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2

# # Parameters
# n = 100  # Number of spatial points
# m = 10   # Number of terms in the series
# dx = 1 / n  # Spatial step size
# dt = 0.0001  # Time step size

# # Create spatial grid
# xlist = [i * dx for i in range(n)]

# # Initial condition for the solution
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n)

# # Iterate for time steps
# iter = 0
# max_iter = 1000 # Adjust as needed for more iterations

# while iter <= max_iter:
#     # Update the solution using the explicit finite difference method
#     for i in range(1, n - 1):
#         u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])
    
#     # Neumann boundary conditions
#     u_n1[0] = u_n1[1]
#     u_n1[n - 1] = u_n1[n - 2]
    
#     # Update the solution for the next time step
#     u_n = u_n1.copy()
#     iter += 1

# # Calculate the analytical solution at the final time step
# final_time = iter * dt
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
