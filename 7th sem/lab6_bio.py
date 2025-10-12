import numpy as np
import matplotlib.pyplot as plt



A = 10.0       # production rate of healthy CD4+ T cells
mu = 0.1       # death rate of healthy T cells
beta = 0.002   # infection rate constant
mu_star = 0.5  # death rate of infected T* cells
gamma = 5.0    # rate at which infected cells produce virus
nu = 2.0       # clearance rate of free virus


T0 = 500.0     # initial healthy CD4+ T cells
Tstar0 = 0.0   # initial infected T cells
V0 = 1.0       # initial viral load


t0 = 0.0
tf = 100.0
dt = 0.1
N = int((tf - t0) / dt) + 1
t = np.linspace(t0, tf, N)

def rhs(y):
    T, T_star, V = y
    dT = A - beta*T*V - mu*T
    dT_star = beta*T*V - mu_star*T_star
    dV = gamma*T_star - nu*V
    return np.array([dT, dT_star, dV])


def rk4_step(y, h):
    k1 = rhs(y)
    k2 = rhs(y + 0.5*h*k1)
    k3 = rhs(y + 0.5*h*k2)
    k4 = rhs(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def euler_step(y, h):
    return y + h * rhs(y)


y_rk = np.zeros((N, 3))
y_eu = np.zeros((N, 3))
y_rk[0] = np.array([T0, Tstar0, V0])
y_eu[0] = np.array([T0, Tstar0, V0])

for i in range(1, N):
    y_rk[i] = rk4_step(y_rk[i-1], dt)
    y_eu[i] = euler_step(y_eu[i-1], dt)



T_dfe = A / mu
R0 = (beta * A * gamma) / (mu * mu_star * nu)
stable = "Stable" if R0 < 1 else "Unstable"

print("===== HIV Model Parameters =====")
print(f"A = {A}, mu = {mu}, beta = {beta}, mu* = {mu_star}, gamma = {gamma}, nu = {nu}")
print(f"Disease-Free Equilibrium (DFE): T = {T_dfe:.3f}, T* = 0, V = 0")
print(f"Basic Reproduction Number R0 = {R0:.3f}")
print(f"DFE Stability: {stable}")


plt.figure(figsize=(10,6))
plt.plot(t, y_rk[:,0], label='T (healthy cells)')
plt.plot(t, y_rk[:,1], label='T* (infected cells)')
plt.plot(t, y_rk[:,2], label='V (virus)')
plt.xlabel('Time')
plt.ylabel('Concentration / Density')
plt.title('HIV Model — Runge-Kutta Method')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,6))
plt.plot(t, y_eu[:,0], label='T (healthy cells)')
plt.plot(t, y_eu[:,1], label='T* (infected cells)')
plt.plot(t, y_eu[:,2], label='V (virus)')
plt.xlabel('Time')
plt.ylabel('Concentration / Density')
plt.title('HIV Model — Euler Method')
plt.legend()
plt.grid(True)
plt.show()


rel_err_V = np.abs(y_rk[:,2] - y_eu[:,2]) / np.maximum(np.abs(y_rk[:,2]), 1e-8)
print(f"Maximum relative error for V(t) between RK4 and Euler: {np.max(rel_err_V):.4e}")
