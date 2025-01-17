import numpy as np

def optimality(C):
    """Check if the solution is optimal."""
    return all(c >= 0 for c in C)

def find_pivot_column(C):
    """Find the index of the most negative value in the cost row (pivot column)."""
    return np.argmin(C)

def find_pivot_row(A, B, pivot_col):
    """Find the pivot row using the minimum ratio test."""
    ratios = []
    for i in range(len(B)):
        if A[i][pivot_col] > 0:
            ratios.append(B[i] / A[i][pivot_col])
        else:
            ratios.append(float('inf'))
    return np.argmin(ratios)

def pivot(A, B, C, pivot_row, pivot_col):
    """Perform pivot operation."""
    pivot_value = A[pivot_row][pivot_col]
    
    # Divide the pivot row by the pivot value
    A[pivot_row] = A[pivot_row] / pivot_value
    B[pivot_row] = B[pivot_row] / pivot_value
    
    # Update the rest of the rows
    for i in range(len(A)):
        if i != pivot_row:
            factor = A[i][pivot_col]
            A[i] = A[i] - factor * A[pivot_row]
            B[i] = B[i] - factor * B[pivot_row]
    
    # Update the cost function row
    factor = C[pivot_col]
    C = C - factor * A[pivot_row]
    
    return A, B, C

def simplex_method(A, B, C, maximize=True):
    """Simplex method implementation."""
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    
    # Convert to maximization if needed by negating the cost function
    if not maximize:
        C = -C
    
    while not optimality(C):
        pivot_col = find_pivot_column(C)
        pivot_row = find_pivot_row(A, B, pivot_col)
        
        if np.all(A[:, pivot_col] <= 0):
            print("Unbounded solution")
            return None
        
        A, B, C = pivot(A, B, C, pivot_row, pivot_col)
    
    # Retrieve optimal solution
    solution = np.zeros(len(C))
    for i in range(len(B)):
        if np.count_nonzero(A[i]) == 1:
            pivot_col = np.nonzero(A[i])[0][0]
            solution[pivot_col] = B[i]
    
    optimal_value = sum(C[i] * solution[i] for i in range(len(solution)))
    
    if maximize:
        return solution, optimal_value
    else:
        return solution, -optimal_value

# Example usage
C = [9, 8] 
A = [
    [1, 2],
    [1, 3],
    [3, 2]
] 
B = [40, 90, 60]

solution, optimal_value = simplex_method(A, B, C, maximize=False)

print("Optimal Solution:", solution)
print("Optimal Value:", optimal_value)
