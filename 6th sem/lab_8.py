import matplotlib.pyplot as plt
import numpy as np
import copy

def burgers_star(u, v, dy, dt, dx, Re, n):
    iter = 0
    while True:
        diff = 0
        Re=1
        un = copy.deepcopy(u)
        vn = copy.deepcopy(v)
        for i in range(1, n):
            for j in range(1, n):

                u_star=un -dt(un[i][j] *(un[i][j]-un[i-1][j])/dx + vn[i][j]*(vn[i][j]-vn[i][j-1])/dy -(1/Re)*((un[i+1][j]-2*un[i][j] +un[i-1][j])/dx**2  +(un[i][j+1]-2*un[i][j] +un[i][j-1])/dy**2))
                v_star =vn -dt(un[i][j] *(vn[i][j]-vn[i-1][j])/dx + vn[i][j]*(vn[i][j]-vn[i][j-1])/dy -(1/Re)*((vn[i+1][j]-2*vn[i][j] +vn[i-1][j])/dx**2  +(vn[i][j+1]-2*vn[i][j] +vn[i][j-1])/dy**2))

        u, v = un, vn
        iter += 1
        if diff <= 0.001:
            break

    print("Burgers", iter, diff)
    return u, v

def poisson_p(P, u, v, p, dx, dy, dt, n):
    iter = 0
    while True:
        diff = 0
        Pn = copy.deepcopy(P)
        for i in range(1, n):
            for j in range(1, n):
                if i == 1 and j == 1:
                    Pn[i][j] = 1 / 2 * (P[i + 1][j] + P[i][j + 1] - p * ((u[i][j] - u[i - 1][j]) / dx + (v[i][j] - v[i][j - 1]) / dy))
                elif i == 1:
                    Pn[i][j] = 1 / 3 * (P[i + 1][j] + P[i][j + 1] + Pn[i][j - 1] - p * ((u[i][i] - u[i - 1][j]) / dx + (v[i][j] - v[i][j - 1]) / dy))
                elif j == 1 and  (0.4 * n <= i <= 0.6 * n):
                    Pn[i][j] = 1 / 3 * (P[i + 1][j] + Pn[i - 1][j] + P[i][j + 1] - p * ((u[i][j] - u[i - 1][j]) / dx + (v[i][j] - v[i][j - 1]) / dy))
                else:
                    Pn[i][j] = 1 / 4 * (P[i + 1][j] + Pn[i - 1][j] + P[i][j + 1] + Pn[i][j - 1] - p * ((u[i][j] - u[i - 1][j]) / dx + (v[i][j] - v[i][j - 1]) / dy))
                diff = max(diff, abs(Pn[i][j] - P[i][j]))
        
        for i in range(n + 1):
            Pn[i][0] = Pn[i][1]
            Pn[i][n] = Pn[i][n - 1]
            Pn [n][i] = Pn[n - 1][i]
            Pn[0][i] = Pn[1][i]
        
        Pn [0][0] = P[1][1]
        Pn [0][n] = Pn[1][n -1]
        Pn [n][0] = Pn[n - 1][1]
        Pn [n][n] = Pn[n - 1][n - 1]

        for i in range(int(0.4 * n), int(0.6 * n) + 1):
            Pn[i][0] = 0
            Pn[i][n] = 1
            Pn [0][i] = 0
        P = Pn

        iter += 1

        if diff <= 0.001:
            break
    print("Gauss_seidel", iter, diff)
    return P


n = 100
dx = dy = 1 / n
dt = dx ** 2
Re, p = 2, 4
iter = 0

xlist = [i * dx for i in range(n + 1)]
ylist = [j * dy for j in range(n + 1)]

u = np.zeros((n + 1, n + 1))
v = np.zeros((n + 1, n + 1))
P = np.zeros((n + 1, n + 1))

for j in range(int(0.4 * n), int(0.6 * n)):  
    P[n][j] = 1   
    v[n][j] = -1   



plt.contourf(xlist, ylist, P)
plt.show()

plt.contourf(xlist, ylist, u)
plt.show( )

plt.contourf(xlist, ylist, v)
plt. show( )

while True:
    diff = 0

    un = copy.deepcopy (u)
    vn = copy.deepcopy (v)
    
    us, vs = burgers_star(u, v, dy, dt, dx, Re, n)
    
    P = poisson_p(P, us, vs, p, dx, dy, dt, n)
    
    for i in range(1, n):
        for j in range(1, n):
            un[i][j] = us[i][j] - dt / (p * dx) * (P[i][j] - P[i - 1] [j])
            diff = max(diff, abs(un[i] [j] - u[i] [j]))
            
            vn[i][j] = vs[i][j] - dt / (p * dy) * (P[i][j] - P[i][j - 1])
            diff = max(diff, abs(vn[i][j] - v[i][j]))
            
    for i in range(n + 1):
        if i < int(0.45 * n) or i > int(0.55 * n):  # Остальная граница
            v[i][n] = 0   # Нет движения
            P[i][n] = P[i][n - 1]  # Поддерживаем значение

    u, v = un, vn
    
    iter += 1

    print("the end", iter, diff)
    
    if diff <= 0.0001:
        break

print(iter)

plt.contourf(xlist, ylist, u)
plt.show()

plt.contourf(xlist, ylist, v)
plt.show()

plt.contourf(xlist, ylist, P)
plt. show()


