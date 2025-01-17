import numpy as np 

def simplex_method_maximize(objective_coeffs, constraints, rhs): 
    num_variables = len(objective_coeffs)  # Number of decision variables
    num_constraints = len(rhs)  # Number of constraints
    
    # Create the table
    simplex_table = np.zeros((num_constraints + 1, num_variables + num_constraints + 1)) 
    simplex_table[0, :num_variables] = np.array(objective_coeffs)  # Objective function coefficients
    
    # Fill in the constraints
    simplex_table[1:num_constraints + 1, :num_variables] = constraints 
    simplex_table[1:num_constraints + 1, num_variables:num_variables + num_constraints] = np.identity(num_constraints) 
    simplex_table[1:num_constraints + 1, -1] = rhs  # Right-hand side values

    iteration = 0 
    print(f"Initial table (Iteration number {iteration}):\n{format_table(simplex_table)}\n") 

    while np.max(simplex_table[0, :-1]) > 0:  # Find the maximum element in the objective row
        iteration += 1 
        # Find the pivot column
        pivot_col = np.argmax(simplex_table[0, :-1]) 

        # Check for unboundedness
        if all(simplex_table[1:, pivot_col] <= 0): 
            raise Exception("The problem has no solution (unboundedness)") 

        # Determine the pivot row
        ratios = [] 
        for i in range(1, num_constraints + 1):  
            if simplex_table[i, pivot_col] > 0: 
                ratios.append(simplex_table[i, -1] / simplex_table[i, pivot_col]) 
            else: 
                ratios.append(np.inf) 
        pivot_row = np.argmin(ratios) + 1  # Offset to account for the header row

        # Normalize the pivot row
        pivot_value = simplex_table[pivot_row, pivot_col] 
        simplex_table[pivot_row, :] /= pivot_value 

        # Update the other rows
        for i in range(num_constraints + 1): 
            if i != pivot_row:   
                simplex_table[i, :] -= simplex_table[i, pivot_col] * simplex_table[pivot_row, :] 

        simplex_table[0] = -simplex_table[0]  # Negate the objective row for the next iteration

        print(f"Tableau after iteration number {iteration} (Pivot element in row {pivot_row}, column {pivot_col + 1}):\n{format_table(simplex_table)}\n") 

        simplex_table[0] = -simplex_table[0]  # Restore the objective row for the next iteration

    # Extract the optimal solution 
    optimal_solution = np.zeros(num_variables) 
    for i in range(1, num_constraints + 1): 
        basic_var_index = np.where(simplex_table[i, :num_variables] == 1)[0] 
        if len(basic_var_index) == 1: 
            optimal_solution[basic_var_index[0]] = simplex_table[i, -1] 

    # Return the maximum value of the objective function and the optimal solution 
    return -simplex_table[0, -1], optimal_solution  # Maximum value of the objective function 

def format_table(simplex_table): 
    """Formats the tableau, replacing negative zeros with regular zeros.""" 
    return np.where(np.abs(simplex_table) < 1e-10, 0, simplex_table) 

# Example 
objective_coeffs = [3, 4]  # Coefficients of the objective function 
constraints = [ 
    [2, 1], 
    [1, 1], 
    [0, 1], 
    [1, 0], 
]  # Left-hand side of the constraints 
rhs = [16, 10, 6, 7]  # Right-hand side values 

max_value, optimal_solution = simplex_method_maximize(objective_coeffs, constraints, rhs) 
print(f"Maximum value of the objective function: {max_value}") 
print(f"Optimal solution: {optimal_solution}") 