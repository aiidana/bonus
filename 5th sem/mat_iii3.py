# import numpy as np
# import matplotlib.pyplot as plt


# def analytical(x: float, t: float, m: int):
#     analytical = 0
    
#     for i in range(1, m + 1):
#         an = (-2*((-1)**m -1)/(np.pi*m)) * np.exp(-m ** 2 * np.pi ** 2 * t*9/16) * np.sin(m * np.pi * x)
#         analytical += an
#     return  analytical


# def initial(x):
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2


# n = 100
# m = 10
# dx = np.sqrt(2) / n
# dt = 0.0001


# xlist = [i * dx for i in range(n)]

# # Define the iteration limits
# iteration_limits = [10, 1000, 100000]

# # Loop through each iteration limit and plot the solution
# for iter_limit in iteration_limits:
#     u_n = [initial(x) for x in xlist] 
#     u_n1 = np.zeros(n)
#     iter = 0

   
#     while iter < iter_limit:
#         for i in range(1, n - 1):
#             u_n1[i] = u_n[i] + (dt / dx / dx) * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])
#         u_n1[0] = u_n1[1]  
#         u_n1[n - 1] = u_n1[n - 2]  
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
import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float, t: float, m: int):
    analytical = 0
    
    for i in range(1, m + 1):
        an = (-2*((-1)**m -1)/(np.pi*m)) * np.exp(-m ** 2 * np.pi ** 2 * t*9/16) * np.sin(m * np.pi * x/4)
        analytical += an
    return  analytical

def initial(x):
    return 1

n=100
m=10

dx=np.sqrt(2)/n
dt=0.0001

xlist=[i*dx for i in range(n)]
u_n=[initial(x) for x in xlist]
u_n1=np.zeros(n)
iter=0
anall_solution=[]
j=0
while iter<=100000: #100,100000
    diff=0
    for i in range(1,n-1):
        u_n1[i]=u_n[i]+(dt/dx/dx)*(u_n[i+1]-2*u_n[i]+u_n[i-1])
    u_n1[0]=0
    u_n1[n-1]=0

    u_n=u_n1
    iter+=1
for i in range(n):
    anall_solution.append(analytical(i*dx,(j-1)*dt,m))
plt.plot(xlist,u_n,label="numerical solution")
plt.plot(xlist,anall_solution,color='pink', label="Analytical solution")
plt.grid()
plt.show()