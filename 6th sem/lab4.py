import numpy as np
import matplotlib.pyplot as plt
L = 1.0
T = 1.0
C = 1.0
nx = 100
nt = 200
dx = L / nx
dt = T / nt
x = np.linspace(0, L, nx)
u0 = np.zeros(nx) # u(t=0, x) = 0
def apply_boundary_conditions(u):
    u[0] = 2 
    u[-1] = 1.0 # u(t, x=1) = 1
def explicit_scheme(u0, C, dx, dt, nt):
    u = u0.copy()
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx-1):
            u[i] = un[i] - C * dt / dx * (un[i] - un[i-1]) # Против потока
        apply_boundary_conditions(u)
    return u
def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n-1)
    d_ = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n-1):
        c_[i] = c[i] / (b[i] - a[i-1] * c_[i-1])
    for i in range(1, n):
        d_[i] = (d[i] - a[i-1] * d_[i-1]) / (b[i] - a[i-1] * c_[i-1])
    u = np.zeros(n)
    u[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        u[i] = d_[i] - c_[i] * u[i+1]
    return u
def implicit_scheme(u0, C, dx, dt, nt):
    u = u0.copy()
    for n in range(nt):
        a = np.full(nx-1, -C * dt / (2 * dx))
        b = np.full(nx, 1.0)
        c = np.full(nx-1, C * dt / (2 * dx))
        d = u.copy()
        u = thomas_algorithm(a, b, c, d)
        apply_boundary_conditions(u)
    return u
u_explicit = explicit_scheme(u0, C, dx, dt, nt)
u_implicit = implicit_scheme(u0, C, dx, dt, nt)
def calculate_error(u1, u2):
    linf_error = np.max(np.abs(u1 - u2))
    return linf_error
linf_error = calculate_error(u_explicit, u_implicit)
print(f"Error: {linf_error}")
time_steps = [0, nt//4, nt//2]
plt.figure(figsize=(12, 10))
for i, t in enumerate(time_steps):
    plt.subplot(2, 3, i+1)
    plt.plot(x, u_explicit, 'b-', label='Explicit Scheme')
    plt.plot(x, u_implicit, 'r-', label='Implicit Scheme')
    plt.title(f'Time Step {t}')
    plt.xlabel('Distance (x)')
    plt.ylabel('Amplitude (u)')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()
