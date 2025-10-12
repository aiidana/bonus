# import numpy as np
# import matplotlib.pyplot as plt
# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     for i in range(1, m + 1):
#         an = -10**7 +(10**7 * np.exp(-10**(-3)) *t) + 10**4*t
#         analytical += an
#     return analytical
# def initial(x):
#     return 4
# n = 100
# m = 10
# dx = 2* np.pi / n
# dt = 0.00001

# xlist = [i * dx for i in range(n)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n+1)
# iter = 0
# itt=0
# iter_limit=10
# # max_iter = 10 
# # max_iter=100
# max_iter=10
# anall_solution = []
# while iter <= max_iter:
#     for i in range(1, n - 1):
#         u_n1[i] = u_n[i] + (16*dt / dx / dx) * (u_n[i+1] - 2 * u_n[i] + u_n[i-1])
#     u_n1[0] = u_n1[1]
#     u_n1[n-1] = 0
#     u_n = u_n1.copy()
#     iter += 1
#     anall_solution = [analytical(xi, iter * dt, m) for xi in xlist]
# analytical_solution = [] 

# while itt < iter_limit: 
#     # Coefficient vectors for the system 
#     A, B, C, D = [], [], [], [] 
#     for i in range(n): 
#         A.append(-16 / dx / dx) 
#         B.append(1 / dt + 32 / dx / dx) 
#         C.append(-16 / dx / dx) 
#         D.append(u_n[i] / dt) 
     
   
#     alphan = [1] 
#     betan = [0] 
     
#     for i in range(n): 
#         alphan.append(-A[i] / (B[i] + C[i] * alphan[i])) 
#         betan.append((D[i] - C[i] * betan[i]) / (B[i] + C[i] * alphan[i])) 
     
     
#     u_n1[n] = 0  
    
    
#     for i in range(n - 1, -1, -1): 
#         u_n1[i] = alphan[i + 1] * u_n1[i + 1] + betan[i + 1] 
     
#     u_n = u_n1.copy() 
#     itt += 1 


# analytical_solution = [analytical(x, dt * iter_limit, m) for x in xlist]
# # Plot the results
# plt.plot(xlist, u_n, label="Numerical solution")
# plt.plot(xlist, anall_solution, color='pink', label="Analytical solution")
# plt.plot(xlist, analytical_solution, '--', label=f"Analytical solution (iter={iter_limit})")
# plt.xlabel('x')
# plt.ylabel('u(x,t)')
# plt.legend()
# plt.grid(True)
# plt.title("Numerical vs Analytical Solution")
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import sympy as sp

# # Parameters
# K = 1e-3
# b = 10
# T = 50       # simulation time
# dt = 0.01    # time step
# N = int(T/dt)

# # --- 1) Analytical solution ---
# t = sp.symbols('t', real=True)
# y = sp.Function('y')(t)
# ode = sp.Eq(y.diff(t,2) + K*y.diff(t) - b, 0)

# sol = sp.dsolve(ode, ics={y.subs(t,0):0, sp.diff(y,t).subs(t,0):0})
# y_exact = sp.lambdify(t, sol.rhs, 'numpy')

# # --- 2) Numerical solution (finite difference explicit) ---
# y_num = np.zeros(N+1)
# v_num = np.zeros(N+1)  # velocity
# time = np.linspace(0, T, N+1)

# for n in range(N):
#     # Explicit Euler for system y'' = b - K y'
#     a = b - K*v_num[n]
#     v_num[n+1] = v_num[n] + dt*a
#     y_num[n+1] = y_num[n] + dt*v_num[n]

# # --- 3) Plot ---
# plt.figure(figsize=(8,5))
# plt.plot(time, y_num, label='Numerical (Euler)', lw=2)
# plt.plot(time, y_exact(time), '--', label='Analytical (Sympy)', lw=2)
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Damped vibration model with constant force')
# plt.legend()
# plt.grid(True)
# plt.show()

# print("Analytical solution y(t) =", sol.rhs)

