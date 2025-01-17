import numpy as np
import math
import time

# Функция, которую минимизируем
def func_to_minimize_new(x):
    return (
        math.sin(x[0] + x[1])**2 + 5
        + math.cos(x[1])**2
        - math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
    )

# Градиент функции
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

# Гессиан функции (матрица вторых производных)
def hessian_new(x):
    d2f_dx2 = (
        -2 * math.sin(x[0] + x[1])**2 + 2 * math.cos(x[0] + x[1])**2
        - (2 - (4 * x[0]**2 + 108 * x[0] + 2916) / 625)
        * math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
    )
    d2f_dy2 = (
        -2 * math.sin(x[0] + x[1])**2 + 2 * math.cos(x[0] + x[1])**2
        - 2 * math.cos(2 * x[1])
        - (2 - (4 * x[1]**2 + 292 * x[1] + 21316) / 625)
        * math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
    )
    d2f_dxdy = (
        2 * (math.cos(x[0] + x[1])**2 - math.sin(x[0] + x[1])**2)
        - (4 * x[0] * x[1] + 54 * x[1] + 146 * x[0] + 7884) / 625
        * math.exp(-((x[0]**2 + x[1]**2 + 146 * x[1] + 54 * x[0] + 6058) / 25))
    )
    return np.array([
        [d2f_dx2, d2f_dxdy],
        [d2f_dxdy, d2f_dy2]
    ])

# Линейный поиск для шага
def backtracking_line_search(x, grad, direction, alpha=0.3, beta=0.8, max_iter=100):
    t = 1.0
    while max_iter > 0:
        new_x = x + t * direction
        if func_to_minimize_new(new_x) < func_to_minimize_new(x) + alpha * t * np.dot(grad, direction):
            return t
        t *= beta
        max_iter -= 1
    return t

# Проверка, находится ли точка в пределах домена
def is_within_domain(x, domain_size=100):
    return np.all(np.abs(x) <= domain_size)

# Градиентный спуск
def gradient_descent_new(start_point, domain_size=100, max_iter=3000, tolerance=1e-5):
    x = start_point
    iteration_count = 0
    for _ in range(max_iter):
        iteration_count += 1
        if not is_within_domain(x, domain_size):
            break
        grad = gradient_new(x)
        if np.linalg.norm(grad) < tolerance:
            break
        step_size = backtracking_line_search(x, grad, -grad)
        x = x - step_size * grad
    return x, iteration_count

# Метод Ньютона
def newton_method(start_point, domain_size=100, max_iter=300, tolerance=1e-5):
    x = start_point
    iteration_count = 0
    for _ in range(max_iter):
        iteration_count += 1
        if not is_within_domain(x, domain_size):
            break
        grad = gradient_new(x)
        hess = hessian_new(x)
        if np.linalg.norm(grad) < tolerance:
            break
        try:
            direction = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Гессиан необратим, метод Ньютона не может продолжаться.")
            break
        step_size = backtracking_line_search(x, grad, direction)
        x = x + step_size * direction
    return x, iteration_count

# Основной код
num_start_points = 10
start_points = [np.random.uniform(-100, 100, 2) for _ in range(num_start_points)]

# Градиентный спуск
best_gd_point, gd_iterations = None, None
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

gd_time = time.time() - start_time

# Метод Ньютона
best_newton_point, newton_iterations = None, None
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

newton_time = time.time() - start_time

# Вывод результатов
print("Градиентный спуск:")
print(f"Лучшее значение: {best_gd_value}")
print(f"Точка: {best_gd_point}")
print(f"Итерации: {total_gd_iterations}")
print(f"Время: {gd_time:.6f} секунд")

print("\nМетод Ньютона:")
print(f"Лучшее значение: {best_newton_value}")
print(f"Точка: {best_newton_point}")
print(f"Итерации: {total_newton_iterations}")
print(f"Время: {newton_time:.6f} секунд")
