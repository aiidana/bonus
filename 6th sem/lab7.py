# import matplotlib.pyplot as plt
 
# n = 100  # Размер сетки
 
# alpha = 1  # Коэффициент
# dx = dy = 1 / n  # Шаг по пространству
# dt = dx ** 2  # Шаг по времени
# iter = 0  # Счетчик итераций
# Re=1
# # Инициализация массива значений u
# u = [[0] * (n + 1) for _ in range(n + 1)]
# v= [[0] * (n + 1) for _ in range(n + 1)]
# # Граничные условия
# for i in range(n + 1):
#     for j in range(n + 1):
#         if  (i == n and ( n / 3 <= j <= 2*n/3) ):
#             u[i][j] = 1
#         else:
#             u[i][j] = 0
# #(i == 0 and (0<= j <=  n / 3)) or
# xlist = [i * dx for i in range(n + 1)]
# ylist = [j * dx for j in range(n + 1)]
 
# A = -1 / (Re * dx**2)  # Коэффициент A
# B = 2 / dx**2 + 1 / dt +u[n]/dx# Коэффициент B
# C = -u[n]/dx -1/(Re*dx**2)  # Коэффициент C
 
# while True:
#     diff = 0
 
#     un2 = [row[:] for row in u]  # Копия массива u
#     un = [row[:] for row in u]
 
#     # Первый шаг (по x)
#     for j in range(1, n):
#         D = [0] * (n + 1)
#         for i in range(1, n):
#             D[i] = (u[i][j] / dt + v[i][j] * (u[i][j]-u[i][j-1])/dy + (
#                 (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / dy**2) 
#             )
 
#         alphan = [0] 
#         betan = [1]
#         betan[1] = u[0][j]
 
#         for i in range(1, n):
#             alphan[i+1] = -A / (B + C * alphan[i])
#             betan[i+1] = (D[i] - C * betan[i]) / (B + C * alphan[i])
 
#         for i in range(n - 1, 0, -1):
#             un2[i][j] = alphan[i + 1] * un2[i + 1][j] + betan[i + 1]
 
#     # Второй шаг (по y)
#     A = -1 / (Re * dy**2)  # Коэффициент A
#     B = 2 / dy**2 + 1 / dt +v[n]/dx# Коэффициент B
#     C = -v[n]/dy -1/(Re*dy**2)  # Коэффициент C
 
#     for i in range(1, n):
#         D = [0] * (n + 1)
#         for j in range(1, n):
#             D[j] = (u[i][j] / dt + u[i][j] * (u[i][j]-u[i-1][j])/dx + (
#                 (u[i+1][j] - 2*u[i][j] + u[i-1][j]) / dx**2) 
#             )
 
#         alphan = [0] * (n + 1)
#         betan = [0] * (n + 1)
#         betan[1] = u[i][0]
 
#         for j in range(1, n):
#             alphan[j+1] = -A / (B + C * alphan[j])
#             betan[j+1] = (D[j] - C * betan[j]) / (B + C * alphan[j])
 
#         for j in range(n - 1, 0, -1):
#             un[i][j] = alphan[j + 1] * un[i][j + 1] + betan[j + 1]
#             diff = max(diff, abs(u[i][j] - un[i][j]))
 
#     iter += 1
#     u = [row[:] for row in un]
 
#     if iter % 200 == 0:
#         print(f"Iteration {iter}")
#         plt.contourf(xlist, ylist, u)
#         plt.show()
 
#     if diff < 0.0001:
#         break
 
# print(f"Total iterations: {iter}")
# plt.contourf(xlist, ylist, u)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# n = 100  # Grid size
# Re = 1  # Reynolds number
# dx = dy = 1 / n  # Spatial step
# dt = 0.25 * min(dx, dy) ** 2  # Time step for stability
# max_iter = 5000  # Max iterations
# nu = np.ones((n+1, n+1)) / Re 
# # Initialize velocity fields
# u = np.zeros((n + 1, n + 1))
# v = np.zeros((n + 1, n + 1))

# # Boundary condition: u = 1 at the top middle part
# for j in range(n // 3, 2 * n // 3 + 1):
#     nu[n, j] = 1

# # X and Y coordinates for visualization
# xlist = np.linspace(0, 1, n + 1)
# ylist = np.linspace(0, 1, n + 1)

# # Alternating Direction Implicit (ADI) coefficients
# A_x = -1 / (Re * dx**2)
# B_x = 2 / dx**2 + 1 / dt
# C_x = -1 / (Re * dx**2)
# A_y = -1 / (Re * dy**2)
# B_y = 2 / dy**2 + 1 / dt
# C_y = -1 / (Re * dy**2)

# iter_count = 0
# while iter_count < max_iter:
#     diff = 0
#     u_old = np.copy(u)

#     # First step: Solve in x-direction
#     u_new = np.copy(u)
#     for j in range(1, n):
#         D = np.zeros(n + 1)
#         for i in range(1, n):
#             D[i] = u[i, j] / dt + v[i, j] * (u[i, j] - u[i, j - 1]) / dy \
#                    + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2

#         # Solve tridiagonal system using Thomas algorithm
#         alpha = np.zeros(n + 1)
#         beta = np.zeros(n + 1)
#         beta[1] = u[0, j]  # Boundary condition at i=0

#         for i in range(1, n):
#             alpha[i + 1] = -A_x / (B_x + C_x * alpha[i])
#             beta[i + 1] = (D[i] - C_x * beta[i]) / (B_x + C_x * alpha[i])
        
#         for i in range(n - 1, 0, -1):
#             u_new[i, j] = alpha[i + 1] * u_new[i + 1, j] + beta[i + 1]

#     # Second step: Solve in y-direction
#     u = np.copy(u_new)
#     for i in range(1, n):
#         D = np.zeros(n + 1)
#         for j in range(1, n):
#             D[j] = u[i, j] / dt + u[i, j] * (u[i, j] - u[i - 1, j]) / dx \
#                    + (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2

#         # Solve tridiagonal system
#         alpha = np.zeros(n + 1)
#         beta = np.zeros(n + 1)
#         beta[1] = u[i, 0]  # Boundary condition at j=0

#         for j in range(1, n):
#             alpha[j + 1] = -A_y / (B_y + C_y * alpha[j])
#             beta[j + 1] = (D[j] - C_y * beta[j]) / (B_y + C_y * alpha[j])
        
#         for j in range(n - 1, 0, -1):
#             u_new[i, j] = alpha[j + 1] * u_new[i, j + 1] + beta[j + 1]
#             diff = max(diff, abs(u[i, j] - u_new[i, j]))
    
#     iter_count += 1
#     u = np.copy(u_new)

#     # Plot every 200 iterations
#     if iter_count % 200 == 0:
#         print(f"Iteration {iter_count}")
#         plt.contourf(xlist, ylist, u.T, cmap='hot')
#         plt.colorbar()
#         plt.show()
    
#     # Convergence check
#     if diff < 1e-4:
#         break

# print(f"Total iterations: {iter_count}")
# plt.contourf(xlist, ylist, u.T, cmap='hot')
# plt.colorbar()
# plt.show()



import matplotlib.pyplot as plt
n = 100
Re = 5
alpha = 1  
dx = dy = 1 / n  
dt = dx ** 2  
iter = 0  
u = [[0] * (n + 1) for _ in range(n + 1)]
v = [[0] * (n + 1) for _ in range(n + 1)]
for i in range(n + 1):
    for j in range(n + 1):
        if (j == 0  and 0.3*n <= i <= 0.4 * n)or (j==n and 0.3*n <= i <= 0.4 * n ):
            u[i][j] = 1
        else:
            u[i][j] = 0
for i in range(n + 1):
    for j in range(n + 1):
        if (i == n and 0.3*n <= j <= 0.6*n):
            v[i][j] = 1
        else:
            v[i][j] = 0
xlist = [i * dx for i in range(n + 1)]
ylist = [j * dx for j in range(n + 1)]
while True:
    diff = 0
    un2 = [row[:] for row in u] 
    un = [row[:] for row in u]
    vn2 = [row[:] for row in v] 
    vn = [row[:] for row in v]
# for u in first step
    for j in range(1, n):
        Au1 = -1/(Re*dx**2) 
        Cu1 = [0] * (n+1)
        Du1 = [0] * (n + 1)
        Bu1 = [0] * (n + 1)
        for i in range(1, n):
            Du1[i] = u[i][j]/dt - v[i][j]*(u[i][j]-u[i][j-1])/dy + (u[i][j+1]-2*u[i][j]+u[i][j-1])/dy/dy/Re
            Cu1[i] = -u[i][j]/dx - 1/(Re*dx**2)
            Bu1[i] = 1/dt + u[i][j]/dx + 2/(Re*dx**2)
        alphan = [0] * (n + 1)
        betan = [0] * (n + 1)
        betan[1] = 1
        for i in range(1, n):
            alphan[i+1] = -Au1 / (Bu1[i] + Cu1[i] * alphan[i])
            betan[i+1] = (Du1[i] - Cu1[i] * betan[i]) / (Bu1[i] + Cu1[i] * alphan[i])
        if j>0.4*n and j<0.6*n:
            un2[j][n]=betan[j]/(1-alphan[j])
        for i in range(n - 1, 0, -1):
            un2[i][j] = alphan[i + 1] * un2[i + 1][j] + betan[i + 1]
# for v in first step
    for j in range(1, n):
        Av1 = -1/(Re*dx**2) 
        Cv1 = [0] * (n+1)
        Dv1 = [0] * (n + 1)
        Bv1 = [0] * (n + 1)
        for i in range(1, n):
            Dv1[i] = v[i][j]/dt - v[i][j]*(v[i][j]-v[i][j-1])/dy + (v[i][j+1]-2*v[i][j]+v[i][j-1])/dy/dy/Re 
            Cv1[i] = -u[i][j]/dx - 1/(Re*dx**2)
            Bv1[i] = 1/dt + u[i][j]/dx + 2/(Re*dx**2)
        alphan = [0] * (n + 1)
        betan = [0] * (n + 1)
        vn2[j][n]=1
        for i in range(1, n):
            alphan[i+1] = -Av1 / (Bv1[i] + Cv1[i] * alphan[i])
            betan[i+1] = (Dv1[i] - Cv1[i] * betan[i]) / (Bv1[i] + Cv1[i] * alphan[i])
        for i in range(n - 1, 0, -1):
            vn2[i][j] = alphan[i + 1] * vn2[i + 1][j] + betan[i + 1]
# for u in second step
    Au2 = -1/(Re*dy**2)
    for i in range(1, n):
        Du2 = [0] * (n + 1)
        Bu2 = [0] * (n + 1)
        Cu2 = [0] * (n + 1)
        for j in range(1, n):
            Du2[j] = un2[i][j]/dt - u[i][j]*(un2[i][j]-un2[i-1][j])/dx + (un2[i+1][j]-2*un2[i][j]+un2[i-1][j])/dx/dx/Re
            Bu2[j]= 1/dt + v[i][j]/dy + 2/(Re*dy**2)
            Cu2[j] = -v[i][j]/dy - 1/(Re*dy**2)
        alphan = [0] * (n + 1)
        betan = [0] * (n + 1)
        betan[1]=1
        if i>0.4*n and i<0.6*n:
            un[n][i]=betan[i]/(1-alphan[i])
        for j in range(1, n):
            alphan[j+1] = -Au2 / (Bu2[j] + Cu2[j] * alphan[j])
            betan[j+1] = (Du2[j] - Cu2[j] * betan[j]) / (Bu2[j] + Cu2[j] * alphan[j])
        for j in range(n - 1, 0, -1):
            un[i][j] = alphan[j + 1] * un[i][j + 1] + betan[j + 1]
            diff1 = max(diff, abs(u[i][j] - un[i][j]))
# for v in second step
    Av2 = -1/(Re*dy**2)
    for i in range(1, n):
        Dv2 = [0] * (n + 1)
        Bv2 = [0] * (n + 1)
        Cv2 = [0] * (n + 1)
        for j in range(1, n):
            Dv2[j] = vn2[i][j]/dt - u[i][j]*(vn2[i][j]-vn2[i-1][j])/dx + (vn2[i+1][j]-2*vn2[i][j]+vn2[i-1][j])/dx/dx/Re
            Bv2[j]= 1/dt + v[i][j]/dy + 2/(Re*dy**2)
            Cv2[j] = -v[i][j]/dy - 1/(Re*dy**2)
        alphan = [0] * (n + 1)
        betan = [0] * (n + 1)
        vn[j][n]=1

        for j in range(1, n):
            alphan[j+1] = -Av2 / (Bv2[j] + Cv2[j] * alphan[j])
            betan[j+1] = (Dv2[j] - Cv2[j] * betan[j]) / (Bv2[j] + Cv2[j] * alphan[j])
        for j in range(n - 1, 0, -1):
            vn[i][j] = alphan[j + 1] * vn[i][j + 1] + betan[j + 1]
            diff2 = max(diff, abs(v[i][j] - vn[i][j]))      
    iter += 1
    u = [row[:] for row in un]
    v = [row[:] for row in vn]
    if iter % 200 == 0:
        print(f"Iteration {iter}" )
        plt.contourf(xlist, ylist, u, cmap='autumn')
        plt.colorbar()  
        plt.show()
        plt.contourf(xlist, ylist, v, cmap='summer')
        plt.colorbar()  
        plt.show()
    if diff1 < 0.000001 and diff2 < 0.000001:
        break
print(f"Total iterations: {iter}")
plt.contourf(xlist, ylist, u, cmap='autumn')
plt.colorbar()  
plt.show()
plt.contourf(xlist, ylist, v, cmap='summer')
plt.colorbar()  
plt.show()