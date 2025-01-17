#lab1
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
# x0, xn, h =0,1,0.1
# x0, xn, h =0,1,0.05
x0, xn, h =0,1,0.025
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



