


import numpy as np
import math
import time

# Function to minimize
def func_to_minimize_new(x):
    return (
        math.sin(x[0] + x[1])**2 + 5
        + math.cos(x[1])**2
        - math.exp(-((x[0]**2 + x[1]**2 + 146*x[1] + 54*x[0] + 6058) / 25))
    )

# Gradient of the function
def gradient_new(x):
    df_dx = (
        2 * math.sin(x[0] + x[1]) * math.cos(x[0] + x[1])
        - (2 * x[0] + 54)
        * math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
        / 25
    )
    df_dy = (
        2 * math.sin(x[0] + x[1]) * math.cos(x[0] + x[1])
        - 2 * math.sin(x[1]) * math.cos(x[1])
        - (2 * x[1] + 146)
        * math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
        / 25
    )
    return np.array([df_dx, df_dy])

# Hessian of the function
def hessian_new(x):
    h11 = (
        -2 * math.sin(x[0] + x[1])**2 + 2 * math.cos(x[0] + x[1])**2
        - (2 - 4 * x[0]**2 / 25)
        * math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
        / 25
    )
    h12 = h21 = (
        -2 * math.sin(x[0] + x[1])**2 + 2 * math.cos(x[0] + x[1])**2
        - 4 * x[0] * x[1] / (25 * 25)
    )
    h22 = (
        -2 * math.sin(x[0] + x[1])**2 + 2 * math.cos(x[0] + x[1])**2
        - (2 - 4 * x[1]**2 / 25)
        * math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
        / 25
    )
    return np.array([[h11, h12], [h21, h22]])

# Backtracking line search
def backtracking_line_search(x, grad, alpha=0.3, beta=0.8, max_iter=100):
    t = 1.0
    while max_iter > 0:
        new_x = x - t * grad
        if func_to_minimize_new(new_x) < func_to_minimize_new(x) - alpha * t * np.dot(grad, grad): 
            return t
        t *= beta
        max_iter -= 1
    return t

# Gradient Descent
def gradient_descent_new(start_point, domain_size=100, max_iter=3000, tolerance=1e-5):
    x = start_point
    iteration_count = 0
    for _ in range(max_iter):
        iteration_count += 1
        grad = gradient_new(x)
        if np.linalg.norm(grad) < tolerance:
            break
        step_size = backtracking_line_search(x, grad)
        x = x - step_size * grad
    return x, iteration_count

# Newton's Method
def newton_method(start_point, alpha=0.3, beta=0.8, eps=1e-6, stop_iteration=100):
    x = np.array(start_point, dtype=float)
    iteration = 0
    while iteration < stop_iteration:
        grad = gradient_new(x)
        hess = hessian_new(x)
        if np.linalg.det(hess) == 0:
            print("Hessian is singular. Cannot proceed with Newton's method.")
            break
        delta = np.linalg.solve(hess, grad)
        step_size = backtracking_line_search(x, grad, alpha, beta)
        x = x - step_size * delta
        if np.linalg.norm(grad) < eps:
            break
        iteration += 1
    return x, iteration

# Main execution
num_start_points = 10000
start_points = [np.random.uniform(-5000, 5000, 2) for _ in range(num_start_points)]

# Gradient Descent
best_gd_value = float('inf')
total_gd_iterations = 0
start_time = time.time()
for sp in start_points:
    final_point, iterations = gradient_descent_new(sp)
    total_gd_iterations += iterations
    final_value = func_to_minimize_new(final_point)
    if final_value < best_gd_value:
        best_gd_value = final_value
        best_gd_point = final_point
elapsed_gd_time = time.time() - start_time

# Newton's Method
best_newton_value = float('inf')
total_newton_iterations = 0
start_time = time.time()
for sp in start_points:
    final_point, iterations = newton_method(sp)
    total_newton_iterations += iterations
    final_value = func_to_minimize_new(final_point)
    if final_value < best_newton_value:
        best_newton_value = final_value
        best_newton_point = final_point
elapsed_newton_time = time.time() - start_time

# Results
print("Gradient Descent Results:")
print(f"Best point: {best_gd_point}")
print(f"Best value: {best_gd_value}")
print(f"Total iterations: {total_gd_iterations}")
print(f"Elapsed time: {elapsed_gd_time:.6f} seconds")

print("\nNewton's Method Results:")
print(f"Best point: {best_newton_point}")
print(f"Best value: {best_newton_value}")
print(f"Total iterations: {total_newton_iterations}")
print(f"Elapsed time: {elapsed_newton_time:.6f} seconds")

