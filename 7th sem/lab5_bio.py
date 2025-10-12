import numpy as np
import matplotlib.pyplot as plt

# --- Общие параметры ---
A = 2.0            
gamma = nu = mu = 0.05  
N0 = 1000.0
I0 = 0.2 * N0
R0 = 0.0
S0 = N0 - I0 - R0

t0, tf, dt = 0, 100, 0.1
t = np.arange(t0, tf + dt, dt)

def sir_rhs(S, I, R, beta):
    dS = A - beta * S * I + gamma * R - mu * S
    dI = beta * S * I - nu * I - mu * I
    dR = nu * I - gamma * R - mu * R
    return dS, dI, dR

def euler(S0, I0, R0, t, dt, beta):
    S, I, R = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(len(t) - 1):
        dS, dI, dR = sir_rhs(S[k], I[k], R[k], beta)
        S[k+1] = S[k] + dt * dS
        I[k+1] = I[k] + dt * dI
        R[k+1] = R[k] + dt * dR
    return S, I, R


beta_stable = 0.0005  
S_stable, I_stable, R_stable = euler(S0, I0, R0, t, dt, beta_stable)
N_stable = S_stable + I_stable + R_stable


beta_unstable = 0.002  
S_unstable, I_unstable, R_unstable = euler(S0, I0, R0, t, dt, beta_unstable)
N_unstable = S_unstable + I_unstable + R_unstable


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
ax1.plot(t, S_stable, label='S (восприимчивые)', color='blue')
ax1.plot(t, I_stable, label='I (зараженные)', color='red')
ax1.plot(t, R_stable, label='R (выздоровевшие)', color='green')
ax1.plot(t, N_stable, '--', color='black', label='N = S+I+R')
ax1.set_title('СТАБИЛЬНАЯ ситуация\n(β = {})'.format(beta_stable))
ax1.set_xlabel('Время')
ax1.set_ylabel('Количество индивидуумов')
ax1.legend()
ax1.grid(True)


ax2.plot(t, S_unstable, label='S (восприимчивые)', color='blue')
ax2.plot(t, I_unstable, label='I (зараженные)', color='red')
ax2.plot(t, R_unstable, label='R (выздоровевшие)', color='green')
ax2.plot(t, N_unstable, '--', color='black', label='N = S+I+R')
ax2.set_title('НЕСТАБИЛЬНАЯ ситуация\n(β = {})'.format(beta_unstable))
ax2.set_xlabel('Время')
ax2.set_ylabel('Количество индивидуумов')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, I_stable, label='Стабильная ситуация (β = {})'.format(beta_stable), color='green', linewidth=2)
plt.plot(t, I_unstable, label='Нестабильная ситуация (β = {})'.format(beta_unstable), color='red', linewidth=2)
plt.title('Сравнение динамики зараженных')
plt.xlabel('Время')
plt.ylabel('Количество зараженных')
plt.legend()
plt.grid(True)
plt.show()