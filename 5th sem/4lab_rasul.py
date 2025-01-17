# import numpy as np 
# import matplotlib.pyplot as plt 
 
# def analytical(x: float, t: float, m: int): 
      
#      analytical = 0 
#      for i in range(1, m + 1): 
#         an = ((16 * (-1)**(i + 1)) / (np.pi * (2 * i - 1))) * np.exp(-(2 * i - 1)**2 * t) * np.cos((2 * i - 1) * x / 4) 
#         analytical += an 
#      return analytical 
 
# def initial(x): 
#     return 4 
 
# # Parameters 
# n = 100  
# m = 10  
# dx = 2* np.pi / n 
# dt = 0.00001  
 
 
# iter_limit = 100  
 
 
# xlist = [i * dx for i in range(n + 1)] 
# u_n = [initial(x) for x in xlist] 
# u_n1 = np.zeros(n)  # Инициализируем u_n1 размером n+1 
# itt = 0 
# anall_solution = [] 
# # Основной цикл 
# while itt < iter_limit: 
#     # Векторные коэффициенты для системы 
#     A, B, C, D = [], [], [], [] 
#     for i in range(n): 
#         A.append(-16 / dx / dx) 
#         B.append(1 / dt + 32 / dx / dx) 
#         C.append(-16 / dx / dx) 
#         D.append(u_n[i] / dt) 
     
#     # Массивы для расчетов в схеме 
#     alphan = [1] 
#     betan = [0] 
     
#     for i in range(n): 
#         alphan.append(-A[i] / (B[i] + C[i] * alphan[i])) 
#         betan.append(D[i] - C[i] * betan[i] / (B[i] + C[i] * alphan[i])) 
     
    
#     u_n1[n] = 0  # Задаем значение в правой границе 
 
    
#     for i in range(n-1, -1, -1): 
#         u_n1[i] = alphan[i+1] * u_n1[i+1] + betan[i+1] 
     
  
#     u_n = u_n1.copy() 
     
#     itt += 1 
 

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
dx = 2 * np.pi / n 
dt = 0.00001  
 
iter_limit = 1000
# iter_limit = 100
# iter_limit = 1000
 
xlist = [i * dx for i in range(n + 1)] 
u_n = [initial(x) for x in xlist] 
u_n1 = np.zeros(n + 1)  # Initialize u_n1 with size n+1 to match xlist 
itt = 0 
anall_solution = [] 

while itt < iter_limit: 
    # Coefficient vectors for the system 
    A, B, C, D = [], [], [], [] 
    for i in range(n): 
        A.append(-16 / dx / dx) 
        B.append(1 / dt + 32 / dx / dx) 
        C.append(-16 / dx / dx) 
        D.append(u_n[i] / dt) 
     
   
    alphan = [1] 
    betan = [0] 
     
    for i in range(n): 
        alphan.append(-A[i] / (B[i] + C[i] * alphan[i])) 
        betan.append((D[i] - C[i] * betan[i]) / (B[i] + C[i] * alphan[i])) 
     
     
    u_n1[n] = 0  
    
    
    for i in range(n - 1, -1, -1): 
        u_n1[i] = alphan[i + 1] * u_n1[i + 1] + betan[i + 1] 
     
    u_n = u_n1.copy() 
    itt += 1 


anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist] 
 

plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})") 
plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})") 
plt.xlabel("x") 
plt.ylabel("u(x)") 
plt.legend() 
plt.grid() 
plt.title(f"Comparison of Numerical & Analytical Solutions ") 
plt.show()
