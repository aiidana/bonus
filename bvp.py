# import numpy as np
# import matplotlib.pyplot as plt
# import math
# def f(x):
#     if -4 <= x < 0:
#         return -x
#     elif 0 <= x < 4:
#         return 2
#     elif x == 0:
#         return 1
# def fourier_series(x, n):
#     result = 0
#     a0 = 2
#     for i in range(1, n+1):
#         an = (4/(math.pi * i)**2) * (-1 + (-1)**i)
#         bn = 2 * (((-1)**i) + 1) / (math.pi * i)
#         result +=  (an * math.cos(math.pi * i * x / 4)) + (bn * math.sin(math.pi * i * x / 4))
#     return result + a0
# x_values = np.linspace(-4, 4, 1000)
# y_values_f = np.array([f(x) for x in x_values])
# # Calculate Fourier series approximation values for n=5, 10, 20
# y_values_n5 = np.array([fourier_series(x, 5) for x in x_values])
# y_values_n10 = np.array([fourier_series(x, 10) for x in x_values])
# y_values_n20 = np.array([fourier_series(x, 20) for x in x_values])
# plt.figure(figsize=(10, 6))
# plt.plot(x_values, y_values_f, label='Original Function', color='red')
# plt.plot(x_values, y_values_n5, label='Fourier Series (n=5)', linestyle='--')
# plt.plot(x_values, y_values_n10, label='Fourier Series (n=10)', linestyle='--')
# plt.plot(x_values, y_values_n20, label='Fourier Series (n=20)', linestyle='--')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Function and its Fourier Series Approximations')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math
def f(x):
    if -4 <= x < 0:
        return -x
    elif 0 <= x < 4:
        return 2
    elif x == 0:
        return 1
def fourier_series(x, n):
    result = 0
    a0 = 2
    for i in range(1, n+1):
        an = (4/(math.pi * i)**2) * (-1 + (-1)**i)
        bn = 2 * (((-1)**i) + 1) / (math.pi * i)
        result +=  (an * math.cos(math.pi * i * x / 4)) + (bn * math.sin(math.pi * i * x / 4))
    return result + a0
x_values = np.linspace(-4, 4, 1000)
y_values_f = np.array([f(x) for x in x_values])
# Calculate Fourier series approximation values for n=5, 10, 20
y_values_n5 = np.array([fourier_series(x, 5) for x in x_values])
y_values_n10 = np.array([fourier_series(x, 10) for x in x_values])
y_values_n20 = np.array([fourier_series(x, 20) for x in x_values])
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(x_values, y_values_f, label='Original Function', color='red')
axs[0, 0].set_title('Original Function')
axs[0, 0].grid(True)
axs[0, 1].plot(x_values, y_values_f, label='Original Function', color='red')
axs[0, 1].plot(x_values, y_values_n5, label='Fourier Series (n=5)', linestyle='--')
axs[0, 1].set_title(' Fourier Series n=5')
axs[0, 1].grid(True)
axs[1, 0].plot(x_values, y_values_f, label='Original Function', color='red')
axs[1, 0].plot(x_values, y_values_n10, label='Fourier Series n=10', linestyle='--')
axs[1, 0].set_title(' Fourier Series n=10')
axs[1, 0].grid(True)
axs[1, 1].plot(x_values, y_values_f, label='Original Function', color='red')
axs[1, 1].plot(x_values, y_values_n20, label='Fourier Series (n=20)', linestyle='--')
axs[1, 1].set_title(' Fourier Series n=20')
axs[1, 1].grid(True)
plt.tight_layout()
plt.show()


