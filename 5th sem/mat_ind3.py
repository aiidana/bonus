import numpy as np
import matplotlib.pyplot as plt


def analytical(x: float, t: float, m: int):
    analytical = 0
    a0 = 3 / 5
    for i in range(1, m + 1):
        an = (-48 * ((-1) ** i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
        analytical += an
    return a0 + analytical


def initial(x):
    return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2


n = 100
m = 10
dx = 1 / n
dt = 0.0001


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
