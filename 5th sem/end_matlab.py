# HEAT 
# import numpy as np 
# import matplotlib.pyplot as plt

# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     a0 = 3 / 5
#     for i in range(1, m + 1):
#         an = (-48 * ((-1)**i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
#         analytical += an
#     return a0 + analytical


# def initial(x):
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2

# # Parameters
# n = 100 
# m = 10   
# dx = 1 / n
# dt = 0.00001
# iter_limit = 10 #100,1000


# xlist = [i * dx for i in range(n + 1)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n + 1)  
# itt = 0

# # Main iteration loop
# while itt < iter_limit:
    
#     A, B, C, D = [], [], [], []
#     for i in range(1, n):  
#         A.append(-1 / dx / dx)
#         B.append(1 / dt + 2 / dx / dx)
#         C.append(-1 / dx / dx)
#         D.append(u_n[i] / dt)
    
    
#     alphan = [0]
#     betan = [0]
    
#     for i in range(1, n):
#         denom = B[i-1] + C[i-1] * alphan[i-1]
#         alphan.append(-A[i-1] / denom)
#         betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

#     # u_n[n] = (betan[n] / (1 - alphan[n]))
#     u_n1[0] = 0  # Left boundary
#     u_n1[n] = 0  # Right boundary

#     u_n1[1] = u_n1[0] + 0.5 * dx  # ux(0) = 0.5
#     u_n1[n] = u_n1[n-1] + 1 * dx #ux(1)=1

#     # Back-substitution for numerical solution at interior points
#     for i in range(n-1, -1, -1):
#         u_n1[i] = alphan[i+1] * u_n1[i + 1] + betan[i+1]

#     # Update old solution with new solution
#     u_n = u_n1
    
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
# plt.title(f" {iter_limit}")
# plt.show()





####################neumann and dirichet
# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x: float, t: float, m: int):
   
#     analytical = 0
#     for i in range(1, m + 1):
#         an = ((16 * (-1)**(i + 1)) / (np.pi * (2 * i - 1))) * np.exp(-(2 * i - 1)**2 * t) * np.cos((2 * i - 1) * x / 4)
#         analytical += an
#     return analytical

# def initial(x):
#     return 4

# # Parameters
# n = 100  
# m = 10   
# dx = np.sqrt(2) / n  
# dt = 0.0001  

# # Create spatial grid
# xlist = [i * dx for i in range(n)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n)


# iter = 0
# # max_iter = 10  
# # max_iter=1000
# max_iter=10000
# anall_solution = []

# while iter <= max_iter:
   
#     for i in range(1, n - 1):
#         u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i+1] - 2 * u_n[i] + u_n[i-1])
#     u_n1[0] = u_n1[1]
#     u_n1[n-1]=u_n1[n-2]

#     u_n1[0]=0
#     u_n1[n-1]=0

#     # Neumann boundary condition at x=0: u_x(0, t) = 0
#     u_n1[0] = u_n1[1]
#     # Dirichlet boundary condition at x=1: u(1, t) = 0
#     u_n1[n-1] = 0
    
#     u_n = u_n1.copy()
#     iter += 1
    
#     anall_solution = [analytical(xi, iter * dt, m) for xi in xlist]

# # Plot the results
# plt.plot(xlist, u_n, label="Numerical solution")
# plt.plot(xlist, anall_solution, color='pink', label="Analytical solution")
# plt.xlabel('x')
# plt.ylabel('u(x,t)')
# plt.legend()
# plt.grid(True)
# plt.title("Numerical vs Analytical Solution")
# plt.show()




######TRANSPORT 
# import numpy as np 
# import matplotlib.pyplot as plt

# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     a0 = 3 / 5
#     for i in range(1, m + 1):
#         an = (-48 * ((-1)**i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
#         analytical += an
#     return a0 + analytical


# def initial(x):
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2

# # Parameters
# n = 100 
# m = 10   
# dx = 1 / n
# dt = 0.00001
# iter_limit = 10 #100,1000


# xlist = [i * dx for i in range(n + 1)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n + 1)  
# itt = 0

# # Main iteration loop
# while itt < iter_limit:
    
#     A, B, C, D = [], [], [], []
#     for i in range(1, n):  
#         A.append(-c / dx )
#         B.append(1 / dt -c / dx)
#         C.append(0)
#         D.append(u_n[i] / dt)
    
    
#     alphan = [0]
#     betan = [0]
    
#     for i in range(1, n):
#         denom = B[i-1] + C[i-1] * alphan[i-1]
#         alphan.append(-A[i-1] / denom)
#         betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

#     # u_n[n] = (betan[n] / (1 - alphan[n]))
#     u_n1[0] = 0  # Left boundary
#     u_n1[n] = 0  # Right boundary

#     u_n1[1] = u_n1[0] + 0.5 * dx  # ux(0) = 0.5
#     u_n1[n] = u_n1[n-1] + 1 * dx #ux(1)=1

#     # Back-substitution for numerical solution at interior points
#     for i in range(n-1, -1, -1):
#         u_n1[i] = alphan[i+1] * u_n1[i + 1] + betan[i+1]

#     # Update old solution with new solution
#     u_n = u_n1
    
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
# plt.title(f" {iter_limit}")
# plt.show()



####WAVE
# import numpy as np 
# import matplotlib.pyplot as plt

# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     a0 = 3 / 5
#     for i in range(1, m + 1):
#         an = (-48 * ((-1)**i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
#         analytical += an
#     return a0 + analytical


# def initial(x):
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2

# # Parameters
# n = 100 
# m = 10   
# dx = 1 / n
# dt = 0.00001
# iter_limit = 10 #100,1000


# xlist = [i * dx for i in range(n + 1)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n + 1)  
# itt = 0

# # Main iteration loop
# while itt < iter_limit:
    
#     A, B, C, D = [], [], [], []
#     for i in range(1, n):  
#         A.append(-1 / dx / dx)
#         B.append(1 / dt/dt + 2 / dx / dx)
#         C.append(-1 / dx / dx)
#         D.append(2*u_n[i] / dt/dt -u_n[i-1]/dt/dt)
    
    
#     alphan = [0]
#     betan = [0]
    
#     for i in range(1, n):
#         denom = B[i-1] + C[i-1] * alphan[i-1]
#         alphan.append(-A[i-1] / denom)
#         betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

#     # u_n[n] = (betan[n] / (1 - alphan[n]))
#     u_n1[0] = 0  # Left boundary
#     u_n1[n] = 0  # Right boundary

#     u_n1[1] = u_n1[0] + 0.5 * dx  # ux(0) = 0.5
#     u_n1[n] = u_n1[n-1] + 1 * dx #ux(1)=1

#     # Back-substitution for numerical solution at interior points
#     for i in range(n-1, -1, -1):
#         u_n1[i] = alphan[i+1] * u_n1[i + 1] + betan[i+1]

#     # Update old solution with new solution
#     u_n = u_n1
    
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
# plt.title(f" {iter_limit}")
# plt.show()