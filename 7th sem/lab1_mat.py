# import numpy as np
# import matplotlib.pyplot as plt

# def f(x, y):
#     return -2*y/x + 3/(x**3) + 1

# def exact_solution(x):
#     return (3*np.log(x)/x**2) + x/3 + 2/(3*x**2)


# def eulermethod(f, y0, x0, x_end, h):
#     x_val = np.arange(x0, x_end + h, h)
#     y_val = [y0]

#     for x in x_val[:-1]:
#         y0 = y0 + h * f(x, y0)
#         y_val.append(y0)
    
#     return x_val, y_val


# x0 = 1
# y0 = 1
# h=0.1
# # h = 0.05
# # h=0.025
# x_end = 2
# x_euler, y_euler = eulermethod(f, y0, x0, x_end, h)


# x_exact = np.linspace(x0, x_end, 1000)
# y_exact = exact_solution(x_exact)


# print(f'x={x_euler}, y={y_euler}')


# plt.figure(figsize=(10, 6))
# plt.plot(x_exact, y_exact, label="Exact (Analytical) Solution", color='green', linewidth=2)
# plt.plot(x_euler, y_euler, label="Euler's Method", linestyle='--', color='blue', linewidth=2)

# plt.title('Lab 1: Comparison of Euler Method and Exact Solution', fontsize=14)
# plt.xlabel("x", fontsize=12)
# plt.ylabel(f'y      (h={h})', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# def f(x, y):
#     return -2*y/x + 3/(x**3) + 1
# def exact_solution(x):
#     return (3*np.log(x)/x**2) + x/3 + 2/(3*x**2)
# def eulermethod(f, y0, x0, x_end, h):
#     n = int((x_end - x0) / h)  # Number of steps
#     x_values = np.linspace(x0, x_end, n+1)
#     y_values = np.zeros(n+1)
#     y_values[0] = y0
#     for i in range(n):
#         y_values[i+1] = y_values[i] + h * f(x_values[i], y_values[i])
#     return x_values, y_values
# x0 = 1
# y0 = 1
# # h=0.1
# # h = 0.05
# h=0.025
# x_end = 2
# x_euler, y_euler = eulermethod(f, y0, x0, x_end, h)
# x_exact = np.linspace(x0, x_end, 1000)
# y_exact = exact_solution(x_exact)
# print(f'x={x_euler}, y={y_euler}')
# plt.figure(figsize=(10, 6))
# plt.plot(x_exact, y_exact, label="Exact (Analytical) Solution", color='green', linewidth=2)
# plt.plot(x_euler, y_euler, label="Euler's Method", linestyle='--', color='blue', linewidth=2)
# plt.title('Lab 1: Comparison of Euler Method and Exact Solution', fontsize=14)
# plt.xlabel("x", fontsize=12)
# plt.ylabel(f'y      (h={h})', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
def f(x,y):
    return 7*np.exp(4*x) -(3*y)
def exact(x):
    return  np.exp(4*x) + (1/np.exp(3*x))
def euler(f,y0,x0,xn,h):
    x_val=np.arange(x0,xn+h,h)
    y_val=[y0]
    for x in x_val[:-1]:
        y0=y0+ h* f(x,y0)
        y_val.append(y0)
    return x_val,y_val
x0, xn, h =0,1,0.1
# x0, xn, h =0,1,0.05
# x0, xn, h =0,1,0.025
y0 = 2
x_euler, y_euler = euler(f, y0, x0, xn, h)
x_exact = np.linspace(x0, xn, 100)
y_exact = exact(x_exact)
print(f'x={x_euler},y={y_euler}')
# for i in range(len(x_euler)):
#     print(f'x={x_euler},y={y_euler}')
#Plot the results
plt.plot(x_exact, y_exact, label="Exact Solution", linewidth=2)
plt.plot(x_euler, y_euler, label="Euler's Method", marker='o', linestyle='dashed')
# Add labels and legend
plt.xlabel(" ")
plt.ylabel(f'solution for h={h}')
plt.legend()
plt.show()
result=euler(f, y0, x0, xn, h)
print('\n',result)
