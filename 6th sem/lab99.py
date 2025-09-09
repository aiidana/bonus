import matplotlib.pyplot as plt
import numpy as np
import copy

# Parameters
rho = 1
Re = 10
N = 100
tol = 1e-4
alpha = 0.01  # коэффициент теплопроводности

def BoundaryConditionsHorizontal(P, u, v):
    for i in range(N+1):
        if int(N/3) <= i <= int(2*N/3):
            P[i][N] = -1
            u[i][N] = -1

def BurgersStar(u, v, dx, dy, dt):
    u_star = copy.deepcopy(u)
    v_star = copy.deepcopy(v)
    iter = 0

    while True:
        diff = 0
        u_prev = copy.deepcopy(u_star)
        v_prev = copy.deepcopy(v_star)

        for i in range(1, N):
            for j in range(1, N):
                u_star[i][j] = u_prev[i][j] + dt * (-u_prev[i][j]*(u_prev[i][j] - u_prev[i - 1][j]) / dx - v_prev[i][j]*(u_prev[i][j] - u_prev[i][j - 1]) / dy + (1/Re)*((u_prev[i + 1][j] - 2 * u_prev[i][j] + u_prev[i - 1][j]) / (dx*dx) + (u_prev[i][j + 1] - 2 * u_prev[i][j] + u_prev[i][j - 1]) / (dy*dy)))
                v_star[i][j] = v_prev[i][j] + dt * (-u_prev[i][j]*(v_prev[i][j] - v_prev[i - 1][j]) / dx - v_prev[i][j]*(v_prev[i][j] - v_prev[i][j - 1]) / dy + (1/Re)*((v_prev[i + 1][j] - 2 * v_prev[i][j] + v_prev[i - 1][j]) / (dx*dx) + (v_prev[i][j + 1] - 2 * v_prev[i][j] + v_prev[i][j - 1]) / (dy*dy)))
                diff = max(diff, abs(u_star[i][j] - u_prev[i][j]), abs(v_star[i][j] - v_prev[i][j]))

        for i in range(N+1):
            u_star[i][0] = 0
            u_star[0][i] = 0
            u_star[i][N] = 0
            u_star[N][i] = 0
            v_star[i][0] = 0
            v_star[0][i] = 0
            v_star[i][N] = 0
            v_star[N][i] = 0

        for i in range(N+1):
            if int(N/3) <= i <= int(2*N/3):
                u_star[i][0] = u_star[i][1]
                v_star[0][i] = v_star[1][i]
                u_star[i][N] = 1

        iter += 1
        if diff < tol*10:
            break

    return u_star, v_star

def PoissonP(P, u, v, dx, dy, dt, w=1.5):
    P_new = copy.deepcopy(P)
    iter = 0

    while True:
        diff = 0
        P_curr = copy.deepcopy(P_new)

        for i in range(1, N):
            for j in range(1, N):
                rhs = (rho / dt) * ((u[i + 1][j] - u[i][j]) / dx + (v[i][j + 1] - v[i][j]) / dy)
                P_new[i][j] = w/4 * (P_curr[i+1][j] + P_new[i-1][j] + P_curr[i][j+1] + P_new[i][j-1] - 4*(1 - 1/w)*P_curr[i][j] - dx * dx * rhs)
                diff = max(diff, abs(P_new[i][j] - P_curr[i][j]))

        for i in range(N+1):
            P_new[i][0] = P_new[i][1]
            P_new[0][i] = P_new[1][i]
            P_new[i][N] = P_new[i][N-1]
            P_new[N][i] = P_new[1][N-1]

        for i in range(N+1):
            if int(N/3) <= i <= int(2*N/3):
                P_new[i][0] = 0
                P_new[0][i] = 0
                P[i][N] = 1

        iter += 1
        if diff < 0.1:
            break

    return P_new

# Step parameters
dx = dy = 1/N
dt = dx*dx

# Computational grid
x_list = np.array([i*dx for i in range(N+1)])
y_list = np.array([i*dy for i in range(N+1)])

# Fields
u = np.zeros((N+1, N+1))
v = np.zeros((N+1, N+1))
P = np.zeros((N+1, N+1))
T = np.zeros((N+1, N+1))  # Температура

# Граничные условия: вход сверху
T[:, N] = 1

# Нагреватель внизу
for i in range(int(N/2 - N/20), int(N/2 + N/20)):
    T[i][0] = 1

BoundaryConditionsHorizontal(P, u, v)

# Основной цикл
for iter in range(300):
    u_n = copy.deepcopy(u)
    v_n = copy.deepcopy(v)

    u_star, v_star = BurgersStar(u, v, dx, dy, dt)
    P = PoissonP(P, u_star, v_star, dx, dy, dt)

    for i in range(1, N):
        for j in range(1, N):
            u_n[i][j] = u_star[i][j] - dt/(rho * dx) * (P[i][j] - P[i-1][j])
            v_n[i][j] = v_star[i][j] - dt/(rho * dy) * (P[i][j] - P[i][j-1])

    for i in range(N+1):
        if int(N/3) <= i <= int(2*N/3):
            u_n[i][0] = u_n[i][1]
            v_n[0][i] = v_n[1][i]

    u, v = u_n, v_n

    # Обновляем температуру
    T_new = copy.deepcopy(T)
    for i in range(1, N):
        for j in range(1, N):
            convection = -u[i][j] * (T[i][j] - T[i-1][j]) / dx - v[i][j] * (T[i][j] - T[i][j-1]) / dy
            diffusion = alpha * ((T[i+1][j] - 2*T[i][j] + T[i-1][j]) / dx**2 + (T[i][j+1] - 2*T[i][j] + T[i][j-1]) / dy**2)
            T_new[i][j] = T[i][j] + dt * (convection + diffusion)

    # Граничные условия для температуры
    T_new[:, N] = 1  # Вход сверху
    T_new[:, 0] = T_new[:, 1]  # Низ (изоляция)
    for i in range(int(N/2 - N/20), int(N/2 + N/20)):
        T_new[i][0] = 1  # Нагреватель

    T_new[0, :] = T_new[1, :]      # левая стена
    T_new[N, :] = T_new[N-1, :]    # правая стена

    T = T_new

# Визуализация
plt.contourf(x_list, y_list, u)
plt.title("u-velocity")
plt.colorbar()
plt.show()

plt.contourf(x_list, y_list, v)
plt.title("v-velocity")
plt.colorbar()
plt.show()

plt.contourf(x_list, y_list, P)
plt.title("Pressure")
plt.colorbar()
plt.show()

plt.contourf(x_list, y_list, T)
plt.title("Temperature")
plt.colorbar()
plt.show()
