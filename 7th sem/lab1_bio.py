import numpy as np
import matplotlib.pyplot as plt

# parameters
K = 1e-3 
b = 10.0 
dt = 0.01  
T = 1.0  
N = int(T/dt)

y_n_1 = 0.0   # y^{n-1}
y_n = 0.0     # y^n

time = np.linspace(0, T, N+1)
y_values = [y_n]

# numerical solution, backward
for n in range(1, N+1):
    y_next = (b - K * (y_n - y_n_1) / dt) * dt**2 + 2 * y_n - y_n_1
    y_values.append(y_next)
    y_n_1 = y_n
    y_n = y_next

# аnalytical
def analytical(t, b=b, K=K):
    return (b / K**2) * (np.exp(-K * t) - 1) + (b / K) * t

y_values_analytical = [analytical(t, b, K) for t in time]


plt.figure(figsize=(5,5))
plt.plot(time, y_values, label='Numerical (backward scheme)')
plt.plot(time, y_values_analytical, '--', label='Analytical')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Comparison: Numerical vs Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()
