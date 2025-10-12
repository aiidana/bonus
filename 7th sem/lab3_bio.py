import numpy as np
import matplotlib.pyplot as plt


lambda_ = 0.1  # коэффициент роста
T0 = 2000      
k1 = 1
k2 = 100.0
e0 = 2
T_init = 1000

h = 0.01         
t_max = 10       
t = np.arange(0, t_max + h, h)

T_model_3_1 = np.zeros(len(t))
T_model_3_2 = np.zeros(len(t))

T_model_3_1[0] = T_init
T_model_3_2[0] = T_init

# numerical solution by Euler method
for i in range(1, len(t)):
    # model 3.1
    dTdt_3_1 = lambda_ * T_model_3_1[i-1] * (1 - T_model_3_1[i-1] / T0)
    T_model_3_1[i] = T_model_3_1[i-1] + h * dTdt_3_1

    # model 3.2
    reaction_term = (k1 * k2 * e0 * T_model_3_2[i-1]) / (k2 + k1 * T_model_3_2[i-1])
    dTdt_3_2 = lambda_ * T_model_3_2[i-1] * (1 - T_model_3_2[i-1] / T0) - reaction_term
    T_model_3_2[i] = T_model_3_2[i-1] + h * dTdt_3_2


T_analytical = T0 / (((T0 - T_init) / T_init) * np.exp(-lambda_ * t) + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, T_model_3_1, label="model 3.1 (Euler)")
plt.plot(t, T_model_3_2, label="model 3.2 (Euler)", linestyle='--')
plt.plot(t, T_analytical, label="model 3.1 (analytical)", linestyle=':')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.title("Comparison of numerical and analytical solutions")
plt.legend()
plt.grid(True)
plt.show()
