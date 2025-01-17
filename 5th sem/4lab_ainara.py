# import numpy as np 
# import matplotlib.pyplot as plt 
 

# def analytical(x: float, t: float, m: int): 
#     analytical = 0 
#     for i in range(1, m + 1): 
        
#         an = (-2 * ((-1)**i - 1) / (np.pi * i)) * np.exp(-i ** 2 * np.pi ** 2 * t * 9 / 16) * np.sin(i * np.pi * x / 4) 
#         analytical += an 
#     return analytical 
 
# def initial(x): 
#     return 1 
 
# n = 100   
# m = 10   
# dx = 4 / n   
# dt = 0.00001   
 
# iter_limit = 1000 
 
 
# xlist = [i * dx for i in range(n + 1)] 
# u_n = [initial(x) for x in xlist] 
# u_n1 = np.zeros(n + 1)  # Инициализируем u_n1 размером n+1 
# itt = 0 
# anall_solution = []
 
# while itt < iter_limit: 
#     # Векторные коэффициенты для системы 
#     A, B, C, D = [], [], [], [] 
#     for i in range(n): 
#         A.append(-9 / dx / dx) 
#         B.append(1 / dt + 18 / dx / dx) 
#         C.append(-9 / dx / dx) 
#         D.append(u_n[i] / dt) 
     
    
#     alphan = [0] 
#     betan = [0] 
     
#     for i in range(n): 
#         alphan.append(-A[i] / (B[i] + C[i] * alphan[i])) 
#         betan.append(D[i] - C[i] * betan[i] / (B[i] + C[i] * alphan[i])) 
     
#     # Граничные условия 
#     u_n1[n] = 0  
 
#     # Итерации для численного решения 
#     for i in range(n-1, -1, -1): 
#         u_n1[i] = alphan[i+1] * u_n1[i+1] + betan[i+1] 
     
#     # Обновление старого решения на новое 
#     u_n = u_n1.copy()  # Чтобы не перезаписать данные в процессе вычислений 
     
#     itt += 1 
 
# # Численное решение после заданного числа итераций 
# anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist] 
 
# # Построение графика 
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
 
# Analytical solution function
def analytical(x: float, t: float, m: int):
    analytical = 0
    for i in range(1, m + 1):
        an = (-2 * ((-1)**i - 1) / (np.pi * i)) * np.exp(-i ** 2 * np.pi ** 2 * t * 9 / 16) * np.sin(i * np.pi * x / 4)
        analytical += an
    return analytical

# Initial condition function
def initial(x):
    return 1

# Parameters
n = 100
m = 10
dx = 4 / n
dt = 0.00001
iter_limit = 1000

# Initialize arrays
xlist = [i * dx for i in range(n + 1)]
u_n = [initial(x) for x in xlist]
u_n1 = np.zeros(n + 1)  # Initialize u_n1 with size n+1
itt = 0
anall_solution = []

# Main iteration loop
while itt < iter_limit:
    # Coefficients for the tridiagonal system
    A, B, C, D = [], [], [], []
    for i in range(1, n):  # Adjusted range to skip boundary points
        A.append(-9 / dx / dx)
        B.append(1 / dt + 18 / dx / dx)
        C.append(-9 / dx / dx)
        D.append(u_n[i] / dt)
    
    # Alpha and Beta arrays for the Thomas algorithm
    alphan = [0]
    betan = [0]

    # Forward elimination phase (for the interior points only)
    for i in range(1, n-1):
        denom = B[i-1] + C[i-1] * alphan[i-1]
        alphan.append(-A[i-1] / denom)
        betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

    # Apply boundary conditions
    u_n1[n] = 0  # Right boundary
    u_n1[0] = 0  # Left boundary (added this)

    # Backward substitution phase
    for i in range(n-2, 0, -1):  # Start from n-2 to 1
        u_n1[i] = alphan[i] * u_n1[i + 1] + betan[i]

    # Update for next iteration
    u_n = u_n1.copy()
    itt += 1

# Analytical solution after the specified number of iterations
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




