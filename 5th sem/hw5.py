import numpy as np
import matplotlib.pyplot as plt

# Define constants
n = 51
pi = 3.14159265359

# Initialize parameters
dx = 1.0 / (n - 1)
dt = 0.5 * dx * dx
eps = 0.0000000001
oldP = np.zeros(n)
newP = np.zeros(n)
f = np.zeros(n)
d = np.zeros(n)
alfa = np.zeros(n)
betta = np.zeros(n)

# Set initial conditions
for i in range(n):
    oldP[i] = np.sin(pi * i * dx)

# Set right-hand side
for i in range(n):
    f[i] = 0.0

# Coefficients for the Thomas algorithm
a = -1.0 / (dx * dx)
b = 1.0 / dt + 2.0 / (dx * dx)
c = -1.0 / (dx * dx)

# Main iteration loop
iter = 0
max_diff = float('inf')

while iter < 10000 and max_diff > eps:
    # Fill the d vector
    for i in range(n):
        d[i] = oldP[i] / dt + f[i]

    # Forward substitution (Thomas algorithm)
    alfa[1] = 0.0
    betta[1] = 0.0
    for i in range(1, n - 1):
        alfa[i + 1] = -a / (b + c * alfa[i])
        betta[i + 1] = (d[i] - c * betta[i]) / (b + c * alfa[i])

    # Backward substitution (Thomas algorithm)
    newP[n - 1] = 0.0  # Boundary condition
    for i in range(n - 2, -1, -1):
        newP[i] = alfa[i + 1] * newP[i + 1] + betta[i + 1]

    # Check for convergence
    max_diff = np.max(np.abs(newP - oldP))

    # Update oldP for the next iteration
    oldP[:] = newP
    iter += 1

# Print results
print("Iterations:", iter)
print("X\tP\tTrue")
for i in range(n):
    x = i * dx
    true_value = np.exp(-pi * pi * dt * iter) * np.sin(pi * x)
    print(x, "\t", newP[i], "\t", true_value)

# Create the plot
x = [i * dx for i in range(n)]
y = newP
true_y = [np.exp(-pi * pi * dt * iter) * np.sin(pi * i * dx) for i in range(n)]

plt.plot(x, y, label='Numerical Solution', color='blue')
plt.plot(x, true_y, label='Analytical Solution', color='red')

# Add title and labels
plt.title("Temperature Distribution using Thomas Algorithm")
plt.xlabel("X")
plt.ylabel("P")
plt.legend()
plt.show()
#2thomas 
# import numpy as np
# import matplotlib.pyplot as plt


# n = 101
# pi = 3.14159265359
# C = 1.0

# dx = 1.0 / (n - 1)
# dt = 0.01 * dx / (2.0 * C)
# eps = 0.00001
# oldP = np.zeros(n)
# newP = np.zeros(n)
# d = np.zeros(n)
# alfa = np.zeros(n)
# betta = np.zeros(n)

# for i in range(n):
#     if i <= n // 2:
#         oldP[i] = 1.0


# a = C / dx
# b = 1.0 / dt - C / dx
# c = 0.0

# # Main iteration loop
# iter = 0
# max_diff = float('inf')

# while iter < 100 and max_diff > eps:
#     # Fill the d vector
#     for i in range(n):
#         d[i] = oldP[i] / dt

#     # Forward substitution (Thomas algorithm)
#     alfa[1] = 0.0
#     betta[1] = 1.0
#     for i in range(1, n - 1):
#         alfa[i + 1] = -a / (b + c * alfa[i])
#         betta[i + 1] = (d[i] - c * betta[i]) / (b + c * alfa[i])

    
#     newP[n - 1] = 0.0 
#     for i in range(n - 2, -1, -1):
#         newP[i] = alfa[i + 1] * newP[i + 1] + betta[i + 1]

    
#     max_diff = np.max(np.abs(newP - oldP))

    
#     oldP[:] = newP
#     iter += 1

# # Print results
# print("Iterations:", iter)
# print("VARIABLES = \"X\", \"P\"")
# print("ZONE I=", n, ", F=POINT")
# for i in range(n):
#     print(i * dx, "\t", newP[i])

# # Create the plot
# x = [i * dx for i in range(n)]
# y = newP

# plt.plot(x, y, label='graph ', color='blue')
# plt.title("Temperature Distribution using Thomas Algorithm")
# plt.xlabel("X")
# plt.ylabel("P")
# plt.legend()
# plt.show()
# #1exact
# import numpy as np
# import matplotlib.pyplot as plt

# # Define constants
# n = 51
# pi = 3.14159265359

# # Initialize parameters
# dx = 1.0 / (n - 1)
# dt = 0.5 * dx * dx
# eps = 0.00001

# # Initialize arrays
# oldP = np.zeros(n)
# newP = np.zeros(n)

# # Set initial conditions
# for i in range(n):
#     oldP[i] = np.sin(pi * i * dx)

# # Main loop
# iter = 0
# max_diff = float('inf')  # Initialize max_diff to a large value

# while max_diff > eps:
#     # Boundary conditions
#     oldP[0] = 0.0
#     newP[0] = 0.0
#     oldP[n - 1] = 0.0
#     newP[n - 1] = 0.0

#     # Update interior points
#     for i in range(1, n - 1):
#         newP[i] = oldP[i] + dt * ((oldP[i + 1] - 2.0 * oldP[i] + oldP[i - 1]) / (dx * dx))

#     # Calculate the maximum difference
#     max_diff = np.max(np.abs(newP - oldP))

#     # Update oldP
#     oldP[:] = newP

#     iter += 1

# # Print results
# print("VARIABLES = \"X\", \"P\", \"True\"")
# print("ZONE I=", n, ", F=POINT")
# for i in range(n):
#     true_value = np.exp(-pi * pi * dt * iter) * np.sin(pi * i * dx)
#     print(i * dx, "\t", newP[i], "\t", true_value)

# # Create the plot
# x = [i * dx for i in range(n)]
# y = newP
# true_y = [np.exp(-pi * pi * dt * iter) * np.sin(pi * i * dx) for i in range(n)]

# plt.plot(x, y, label='Numerical Solution', color='blue')
# plt.plot(x, true_y, label='Analytical Solution', color='red')

# # Add title and labels
# plt.title("Temperature Distribution")
# plt.xlabel("X")
# plt.ylabel("P")

# plt.legend()
# plt.show()

#2exact
# import numpy as np
# import matplotlib.pyplot as plt

# # Define constants
# n = 101
# pi = 3.1415
# C = 1.0

# # Initialize arrays
# dx = 1.0 / (n - 1)
# dt = 0.01 * dx / (2.0 * C)
# eps = 0.00001
# oldP = np.zeros(n)
# newP = np.zeros(n)

# # Set initial conditions
# for i in range(n):
#     if i <= n // 2:
#         oldP[i] = 1.0

# # Set boundary conditions
# oldP[0] = 1.0
# oldP[n - 1] = 0.0

# # Main loop
# iter = 0
# while iter < 10000:
#     newP[0] = 1.0
#     newP[n - 1] = 0.0

#     # Update interior points
#     for i in range(1, n - 1):
#         newP[i] = oldP[i] - C * dt * ((oldP[i] - oldP[i - 1]) / dx)

#     # Update max difference
#     max_diff = np.max(np.abs(newP - oldP))

#     # Update oldP
#     oldP[:] = newP

#     iter += 1

# # Print results
# print("Iterations:", iter)

# # Print output in a format similar to the C++ code
# print("VARIABLES = \"X\", \"P\"")
# print("ZONE I=", n, ", F=POINT")
# for i in range(n):
#     print(i * dx, "\t", newP[i])

# # Create the plot
# x = [i * dx for i in range(n)]
# y = newP

# plt.plot(x, y, label='Numerical Solution', color='blue')

# # Add title and labels
# plt.title("Temperature Distribution")
# plt.xlabel("X")
# plt.ylabel("P")

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()