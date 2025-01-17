# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x:float,t:float, m:int):
#     analytical=0
#     for i in range(1,m+1):
#         an=((16*(-1)**(m+1))/(np.pi * (2*m-1))) *np.exp(-(2*m-1)**2 * t) * np.cos((2*m-1)*x/4)
#         analytical+=an
#     return analytical

# def initial(x):
#     return 4

# n=100
# m=10

# dx=np.sqrt(2)/n
# dt=0.0001

# xlist=[i*dx for i in range(n)]
# u_n=[initial(x) for x in xlist]
# u_n1=np.zeros(n)
# iter=0
# anall_solution=[]
# j=0
# while iter <= 100:
#     diff = 0
#     for i in range(1, n-1):
#         u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i+1] - 2 * u_n[i] + u_n[i-1])
#     u_n1[0] = u_n1[1]  # Neumann condition
#     u_n1[n-1] = 0      # Dirichlet condition
#     u_n = u_n1
#     iter += 1
    
#     # Store the analytical solution for the current time step
#     anall_solution = [analytical(xi, iter * dt, m) for xi in xlist]

# for i in range(n):
#     anall_solution.append(analytical(i*dx,(j-1)*dt,m))
# plt.plot(xlist,u_n,label="numerical solution")
# plt.plot(xlist,anall_solution,color='pink', label="Analytical solution")
# plt.grid()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float, t: float, m: int):
   
    analytical = 0
    for i in range(1, m + 1):
        an = ((16 * (-1)**(i + 1)) / (np.pi * (2 * i - 1))) * np.exp(-(2 * i - 1)**2 * t) * np.cos((2 * i - 1) * x / 4)
        analytical += an
    return analytical

def initial(x):
    return 4

# Parameters
n = 100  
m = 10   
dx = np.sqrt(2) / n  
dt = 0.0001  

# Create spatial grid
xlist = [i * dx for i in range(n)]
u_n = [initial(x) for x in xlist]
u_n1 = np.zeros(n)


iter = 0
# max_iter = 10  
# max_iter=1000
max_iter=10000
anall_solution = []

while iter <= max_iter:
   
    for i in range(1, n - 1):
        u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i+1] - 2 * u_n[i] + u_n[i-1])
    
    # Neumann boundary condition at x=0: u_x(0, t) = 0
    u_n1[0] = u_n1[1]
    
    # Dirichlet boundary condition at x=1: u(1, t) = 0
    u_n1[n-1] = 0
    
    u_n = u_n1.copy()
    iter += 1
    
    anall_solution = [analytical(xi, iter * dt, m) for xi in xlist]

# Plot the results
plt.plot(xlist, u_n, label="Numerical solution")
plt.plot(xlist, anall_solution, color='pink', label="Analytical solution")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.title("Numerical vs Analytical Solution")
plt.show()
