import numpy as np
import matplotlib.pyplot as plt


def analytical(x: float, t: float, m: int):
    analytical = 0
    a0 = 3 / 5
    for i in range(1, m + 1):
        an = (-48 * ((-1)**i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
        analytical += an
    return a0 + analytical


def initial(x):
    return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2


n = 100 
m = 10   
dx = 1 / n
dt = 0.00001
iter_limit = 10  # Number of iterations 100,1000

# Spatial grid and initial conditions
xlist = [i * dx for i in range(n + 1)]
u_n = [initial(x) for x in xlist]
u_n1 = np.zeros(n + 1) 
itt = 0


while itt < iter_limit:
    # coefficients for the Thomas algorithm 
    A, B, C, D = [], [], [], []
    for i in range(1, n):  
        A.append(-1 / dx**2)
        B.append(1 / dt + 2 / dx**2)
        C.append(-1 / dx**2)
        D.append(u_n[i] / dt)
    
    
    alphan = [1]  
    betan = [0]  
    
    for i in range(1, n):
        denom = B[i-1] + C[i-1] * alphan[i-1]
        alphan.append(-A[i-1] / denom)
        betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

    
    u_n1[n] = betan[-1] / (1 - alphan[-1]) 

    
    for i in range(n-1, 0, -1):
        u_n1[i] = alphan[i] * u_n1[i + 1] + betan[i]

    # Update old solution with new solution
    u_n = u_n1.copy()
    
    itt += 1


anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist]

# Plotting 
plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}")
plt.show()

