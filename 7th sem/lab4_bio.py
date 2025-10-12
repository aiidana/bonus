import numpy as np

# Define the system F(x) = 0
def F(x):
    x1, x2 = x
    return np.array([
        x1**3 + x2 - 1,   # f1(x1, x2)
        x2**3 - x1 + 1    # f2(x1, x2)
    ])

# Jacobian of F
def J(x):
    x1, x2 = x
    return np.array([
        [3*x1**2, 1],
        [-1, 3*x2**2]
    ])


def newton(x0, tol=1e-6, max_iter=50):
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        delta = np.linalg.solve(Jx, -Fx)
        err_abs = np.linalg.norm(delta)
        x = x + delta

        print(f"Iter {k+1}: |Δx| = {err_abs:.6e}")

        if err_abs < tol:
            print("Converged!")
            return x, k+1  # возвращаем количество сделанных шагов
    
    print("Max iterations reached (no convergence).")
    return x, max_iter



sol, steps = newton([10, 0.5])  # initial guess
print(f"\nSolution: {sol}, steps = {steps}")
