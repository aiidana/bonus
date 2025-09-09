# import matplotlib.pyplot as plt

# n = 100
# dt = 0.001
# dx = dy = 1 / n
# itt = 0


# P = []
# for i in range(n):
#     Pr = []
#     for j in range(n):
        
#         if  j== 0 and (0<= j <= 99):
#             Pr.append(1)
        
#         elif i == 0 and (30 <= j <= 60):
#             Pr.append(1)
#         else:
#             Pr.append(0)
#     P.append(Pr)

# xlist = [i * dx for i in range(n)]
# ylist = [j * dx for j in range(n)]


# while True:
#     Pn = []
#     diff = 0
#     for i in range(n):
#         Pr = []
#         for j in range(n):
            
#             if  j== 0 and (0<= j <= 99):
#                 Pr.append(1)
#             elif i == 0 and (30 <= j <= 60):
#                 Pr.append(1)
#             else:
#                 Pr.append(0)
#         Pn.append(Pr)
    
    
#     for i in range(1, n - 1):
#         for j in range(1, n - 1):
#             Pn[i][j] = 1 / 4 * (P[i + 1][j] + P[i - 1][j] + P[i][j + 1] + P[i][j - 1])
#             # Pn[i, j] = 0.25 * (P[i+1, j] + P[i-1, j] + P[i, j+1] + P[i, j-1])
#             # old = P[i, j]
#             # omega=1.5
#             # P[i, j] = (1 - omega) * old + omega * 0.25 * (P[i+1, j] + P[i-1, j] + P[i, j+1] + P[i, j-1])
               
#             diff = max(diff, abs(Pn[i][j] - P[i][j]))
    
#     itt += 1
#     P = Pn
    
    
#     if itt % 300 == 0:
#         print(itt)
#         plt.title(f"itt= {itt}")
#         plt.contourf(xlist, ylist, P)
#         plt.show()
    
#     if diff < 0.00001:
#         break

# print("Количество итераций:", itt)
# plt.contourf(xlist, ylist, P)
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

n = 100
dx = dy = 1 / n
tolerance = 1e-5
h2 = dx ** 2 
w = 1.5  # Коэффициент релаксации 


P = np.zeros((n, n))


P[0:100, 0] = 1  # Левая граница (j == 0), от 65 до 99
P[0, 35:70] = 1  # Правая граница (i=0), от 35 до 70

xlist = np.linspace(0, 1, n)
ylist = np.linspace(0, 1, n)

itt = 0
while True:
    Pn = P.copy()
    
    # Метод релаксации
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            Pn[i, j] = (w / 4) * ((P[i+1, j] + Pn[i-1, j] + P[i, j+1] + Pn[i, j-1]) - 4*(1 - 1/w) * P[i, j])
    
    
    Pn[0:100, 0] = 1
    Pn[0, 35:70] = 1
    
    diff = np.max(np.abs(Pn - P))
    P = Pn
    itt += 1
    
    if itt % 300 == 0:
        print(f"Итерация: {itt}")
        plt.title(f"Итерация {itt}")
        plt.contourf(xlist, ylist, P, levels=50, cmap='summer')
        plt.colorbar()
        plt.show()
    
    if diff < tolerance:
        break

print(f"Количество итераций: {itt}")
plt.title("Решение уравнения Лапласа (Метод релаксации)")
plt.contourf(xlist, ylist, P, levels=50, cmap='summer')
plt.colorbar()
plt.show()


# import matplotlib.pyplot as plt

# n = 100
# dt = 0.001
# dx = dy = 1 / n
# itt = 0


# P = []
# for i in range(n):
#     Pr = []
#     for j in range(n):
        
#         if  j== 0 and (0<= j <= 99):
#             Pr.append(1)
        
#         elif i == 0 and (35 <= j <= 70):
#             Pr.append(1)
#         else:
#             Pr.append(0)
#     P.append(Pr)

# xlist = [i * dx for i in range(n)]
# ylist = [j * dx for j in range(n)]


# while True:
#     Pn = []
#     diff = 0
#     for i in range(n):
#         Pr = []
#         for j in range(n):
            
#             if  j== 0 and (0<= j <= 99):
#                 Pr.append(1)
#             elif i == 0 and (35 <= j <= 70):
#                 Pr.append(1)
#             else:
#                 Pr.append(0)
#         Pn.append(Pr)
    
    
#     for i in range(1, n - 1):
#         for j in range(1, n - 1):
        
#             Pn[i][j] = 0.25 * (P[i + 1][j] + P[i - 1][j] + P[i][j + 1] + P[i][j - 1])
            
#             diff = max(diff, abs(Pn[i][j] - P[i][j]))
    
#     itt += 1
#     P = Pn
    
    
#     if itt % 300 == 0:
#         print(itt)
#         plt.title(f"itt= {itt}")
#         plt.contourf(xlist, ylist, P)
#         plt.show()
    
#     if diff < 0.00001:
#         break

# print("Количество итераций:", itt)
# plt.contourf(xlist, ylist, P)
# plt.show()