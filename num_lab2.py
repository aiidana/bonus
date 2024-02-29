#lab2
import numpy as np
import matplotlib.pyplot as plt
def f(x,y):
    return 5+x-y
def exact(x):
    return 4+x+(-5* np.exp(2))/ np.exp(x)

def euler(f,y0,x0,xn,h):
    x_val=np.arange(x0,xn+h,h)
    y_val=[y0]

    for x in x_val[:-1]:
        y0=y0+ h* f(x,y0)
        y_val.append(y0)
    
    return x_val,y_val
x0, xn, h =2,4,0.5
y0 = 1
def heuns(f, y0, x0, xn, h):
    x_val = np.arange(x0, xn + h, h)
    y_val = [y0]
    for x in x_val[:-1]:
        k1 = f(x, y0)
        k2 = f(x + h, y0 + h * k1)
        y0 = y0 + 0.5 * h * (k1 + k2)
        y_val.append(y0)
    return x_val,y_val


x_euler, y_euler = euler(f, y0, x0, xn, h)
x_heun, y_heun = heuns(f, y0, x0, xn, h)
x_exact = np.linspace(x0, xn, 100)
y_exact = exact(x_exact)
for i in range(len(x_euler)):
    print(f'x={x_euler},y={y_euler}')
for i in range(len(x_heun)):
    print(f'x={x_heun},y={y_heun}')

#Plot the results
plt.plot(x_exact, y_exact, label="Exact Solution", linewidth=2)
plt.plot(x_euler, y_euler, label="Euler's Method", marker='o', linestyle='dashed')
plt.plot(x_heun, y_heun, label="Heun's Method", marker='s', linestyle='dashed')

# Add labels and legend
plt.xlabel(" ")
plt.ylabel("Solution")
plt.legend()
plt.show()

result=euler(f, y0, x0, xn, h)
resultt=heuns(f, y0, x0, xn, h)
print('\n',result)
print('\n',resultt)

#
# import numpy as np
# from scipy.integrate import quad
# def f(x):
#     return 3 * x**2 -x -1
# def midpoint(f, a, b, n):
#     deltax = (b - a) / n
#     integral = 0
#     for i in range(n):
#         mid_point = a + (i + 0.5) * deltax
#         integral += deltax * f(mid_point)
#     return integral
# a,b,n=1,3,8
# result=midpoint(f,a,b,n)

# def simpson(f, a, b, n):
#     deltax = (b - a) / n
#     integral = f(a) + f(b)  # Initial and final terms
#     for i in range(1, n, 2):  # Odd indices
#         integral += 4 * f(a + i * deltax)
#     for i in range(2, n-1, 2):  # Even indices
#         integral += 2 * f(a + i * deltax)
#     integral =(integral*deltax)/3
#     return integral
# res=simpson(f,a,b,n)
# print('integral of f(x) by simpsons rule=',res)
# exact_result, _ = quad(f, 1, 3)
# midpoint_error = abs(result - exact_result)
# simpsons_error = abs(res - exact_result)
# print('mid error:',midpoint_error)