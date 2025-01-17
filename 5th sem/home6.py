import matplotlib.pyplot as plt 
import numpy as np 

L1 = 0.6  # Length of the first part of the rod (meters) 
L2 = 0.6  # Length of the second part of the rod (meters) 
L_total = L1 + L2  # Total length of the rod 
 
T_end = 1.0  # Total time (seconds) 
 
# Thermal diffusivities for each section 
a1 = 0.206  # Thermal diffusivity for the first section (0 to 0.6 m) 
a2 = 0.0945  # Thermal diffusivity for the second section (0.6 to 1.2 m) 
 
# Discretization for both sections 
Nx1 = 20  # Number of spatial points for the first section 
Nx2 = 20  # Number of spatial points for the second section 
 
Nx_total = Nx1 + Nx2 - 1  # Total number of spatial points 
Nt = 1000  
dx1 = L1 / (Nx1 - 1)  # Spatial step size for the first section 
dx2 = L2 / (Nx2 - 1)  # Spatial step size for the second section 
dt = T_end / Nt  # Time step size 
 
# Stability condition for both sections 
alpha1 = a1**2 * dt / dx1**2 
alpha2 = a2**2 * dt / dx2**2 
 
if alpha1 > 0.5 or alpha2 > 0.5: 
    print(f"Stability condition violated: alpha1 = {alpha1}, alpha2 = {alpha2}") 
else: 
    print(f"Stability condition satisfied: alpha1 = {alpha1}, alpha2 = {alpha2}") 
 
# Initialize the temperature array for the whole rod 
u = np.zeros((Nx_total, Nt+1)) 
 
# Initial condition: u(x, 0) = 25 (everywhere) 
u[:, 0] = 25 
 
# Boundary conditions: u(0, t) = 25, u(0.6, t) = 200, and u(1.2, t) = 25 
u[0, :] = 25 
u[Nx1-1, :] = 200  # Boundary at x = 0.6 m 
u[-1, :] = 25      # Boundary at x = 1.2 m 
 
# Explicit finite difference method 
for n in range(0, Nt): 
    # Update for the first section (0 to 0.6 m) 
    for i in range(1, Nx1-1): 
        u[i, n+1] = u[i, n] + alpha1 * (u[i+1, n] - 2*u[i, n] + u[i-1, n]) 
     
    # Update for the second section (0.6 to 1.2 m) 
    for i in range(Nx1, Nx_total-1): 
        u[i, n+1] = u[i, n] + alpha2 * (u[i+1, n] - 2*u[i, n] + u[i-1, n]) 

x1 = np.linspace(0, L1, Nx1) 
x2 = np.linspace(L1, L_total, Nx2) 
x_total = np.concatenate((x1, x2[1:]))  

plt.figure(figsize=(8, 6)) 
for n in range(0, Nt, Nt//10): 
    plt.plot(x_total, u[:, n], label=f"t = {n*dt:.2f}s") 
 
plt.title("Temperature distribution along the rod (0 to 1.2 m) over time") 
plt.xlabel("Position along the rod (m)") 
plt.ylabel("Temperature (Â°C)") 
plt.legend() 
plt.grid(True) 
plt.show()