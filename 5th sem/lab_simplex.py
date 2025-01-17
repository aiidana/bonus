import numpy as np
def maximize_linear_function(coefficients, constraints, right_hand_side):
   
    num_variables = len(coefficients)
    num_constraints = len(constraints)

    simplex_table = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
    simplex_table[0, :num_variables] = np.array(coefficients)  # Use positive sign for maximization


    simplex_table[1:num_constraints+1, :num_variables] = constraints
    simplex_table[1:num_constraints+1, num_variables:num_variables + num_constraints] = np.identity(num_constraints) #slack variables
    simplex_table[1:num_constraints+1, -1] = right_hand_side

    iteration = 0
    print(f"Initial table (Iteration {iteration}):\n{simplex_table}\n")

    while np.max(simplex_table[0, :-1]) > 0:  # For maximization, look for positive coefficients
        iteration += 1
        
        pivot_column = np.argmax(simplex_table[0, :-1])

        # Check for unboundedness
        if all(simplex_table[1:, pivot_column] <= 0):
            raise Exception("No solution exists (unbounded).")

        # Choose the pivot row
        ratios = []
        for i in range(1,num_constraints+1):
            if simplex_table[i, pivot_column] > 0:
                ratios.append(simplex_table[i, -1] / simplex_table[i, pivot_column])
            else:
                ratios.append(np.inf)
        pivot_row = np.argmin(ratios)+1

        # Make the pivot element equal to 1
        pivot_value = simplex_table[pivot_row, pivot_column]
        simplex_table[pivot_row, :] /= pivot_value

        # Update the other rows
        for i in range(num_constraints + 1):
            if i != pivot_row:
                simplex_table[i, :] -= simplex_table[i, pivot_column] * simplex_table[pivot_row, :]

        print(f"Table after iteration {iteration} (Pivot element in row {pivot_row}, column {pivot_column+1}):\n{simplex_table}\n")

   
    optimal_solution = np.zeros(num_variables)
    for i in range(1,num_constraints+1):
        basic_var_index = np.where(simplex_table[i, :num_variables] == 1)[0]
        if len(basic_var_index) == 1:
            optimal_solution[basic_var_index[0]] = simplex_table[i, -1]

    
    return -simplex_table[0, -1], optimal_solution  # Return the negative of the maximum value

# Example 1
coefficients = [3, 4]  # Coefficients of the objective function
constraints = [
    [2, 1],
    [1, 1],
    [0, 1],
    [1, 0]
]  
right_hand_side = [16, 10, 6, 7] 

max_value, optimal_solution = maximize_linear_function(coefficients, constraints, right_hand_side)
print(f"Maximum value = {max_value}")
print(f"Optimal solutions = {optimal_solution}")