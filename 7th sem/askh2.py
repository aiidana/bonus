# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x:float,t:float, m:int):
#     analytical=0
#     for i in range(1,m+1):
#         an=((16*(-1)**(m+1))/(np.pi * (2*m-1))) *np.exp(-(2*m-1)**2 * t) * np.cos((2*m-1)*x/4)
#         analytical+=an
#     return analytical

# def initial(x):
#     return 4

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
def f(x):
    return x*(1-x)
def fourier(x,m):
    approx=np.zeros_like(x)
    for n in range(1,m+1):
        an=(-4*(-1)**n +4)/(n**3 *pi**3)
        approx+=an *np.sin(n*pi*x)
    return approx



x_values = np.linspace(0,  1, 100)  
f_values=f(x_values)

approx_5=fourier(x_values,m=5)
approx_10=fourier(x_values,m=10)
approx_20=fourier(x_values,m=20)

plt.figure(figsize=(10, 6))
plt.plot(x_values,f_values,label='initial function: f(x)=x(1-x)',color="black")
plt.plot(x_values,approx_5,label="Fourier Approximation m=5",linestyle='--')
plt.plot(x_values,approx_10,label="Fourier Approximation m=10",linestyle='-.')
plt.plot(x_values,approx_20,label="Fourier Approximation m=20",linestyle=':')
plt.title("Fourier series Approximation of f(x)= x(1-x) with Different m (5, 10, 20")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

def u(x, t, m=10):
    sum_result = np.zeros_like(x)  
    for n in range(1, m + 1):
        an=(-4*(-1)**n +4)/(n**3 *pi**3)
        sin_term = np.sin(n*pi*x)
        e_term = np.exp(-n**2 * pi**2  *t)
        sum_result += an * sin_term * e_term
    return sum_result
time_val=[0, 0.01, 0.05, 0.1]
plt.figure(figsize=(10,6))
for t in time_val:
    u_vals=u(x_values,t,m=20)
    plt.plot(x_values,u_vals,label=f't={t}')
plt.title("Solution u(x,t) ")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()