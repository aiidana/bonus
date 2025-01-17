import numpy as np
import matplotlib.pyplot as plt


pi = np.pi
sqrt_3 = np.sqrt(3)
def f(x):
    return x**3 - 3*x* pi**2
def fourier(x,m):
    approx=np.zeros_like(x)
    for n in range(1,m+1):
        an=(-1)**(n+1) *(-192/(pi*(2*n-1)**4) ) 
        approx+=an *np.sin((2*n -1)*x/2)
    return approx



x_values = np.linspace(0,  pi, 100)  # 0 to π
f_values=f(x_values)

approx_5=fourier(x_values,m=5)
approx_10=fourier(x_values,m=10)
approx_20=fourier(x_values,m=20)

plt.figure(figsize=(10, 6))
plt.plot(x_values,f_values,label='initial function: f(x)=x^3 - 3π^2x',color="black")
plt.plot(x_values,approx_5,label="Fourier Approximation m=5",linestyle='--')
plt.plot(x_values,approx_10,label="Fourier Approximation m=10",linestyle='-.')
plt.plot(x_values,approx_20,label="Fourier Approximation m=20",linestyle=':')
plt.title("Fourier series Approximation of f(x)= x^3- 3xπ^2 with Different m (5, 10, 20")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

def u(x, t, m=10):
    sum_result = np.zeros_like(x)  
    for n in range(1, m + 1):
        an=(-1)**(n+1) *(-192/(pi*(2*n-1)**4) )
        cos_term = np.cos((2*n -1) * sqrt_3 * t / 2)
        sin_term = np.sin((2*n -1) * x / 2)
        sum_result += an * cos_term * sin_term
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
# # --- Second Plot: Dynamic Visualization over Time ---
# T = 0.1  # End of the time interval
# time_values = np.linspace(0, T, 100)  # Time interval for animation

# plt.figure(figsize=(8, 6))

# for t in time_values:
#     u_values = u(x_values, t, 20)  # Using N=20 for good approximation
#     plt.plot(x_values, u_values, color='b')
#     plt.title(f'Solution u(x,t) at t={t:.2f}')
#     plt.xlabel('x')
#     plt.ylabel('u(x,t)')
#     plt.grid(True)
#     plt.pause(0.05)  # Pause to simulate animation
#     plt.clf()  # Clear the figure for next time point

# plt.show()
