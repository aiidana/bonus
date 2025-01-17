# import math
# import numpy as np
# c = [1, 1, 0, 0, 0]
# A = [
#     [-1, 1, 1, 0, 0],
#     [ 1, 0, 0, 1, 0],
#     [ 0, 1, 0, 0, 1]
# ]
# b = [2, 4, 4]

# def to_tableau(c, A, b):
#     xb = [eq + [x] for eq, x in zip(A, b)]
#     z = c + [0]
#     return xb + [z]
# def can_be_improved(tableau):
#     z = tableau[-1]
#     return any(x > 0 for x in z[:-1])
# def pivot(table):
#     z=table[-1]
#     column = next(i for i, x in enumerate(z[:-1]) if x > 0)
#     restrictions = []
#     for eq in table[:-1]:
#         el = eq[column]
#         restrictions.append(math.inf if el <= 0 else eq[-1] / el)

#     row = restrictions.index(min(restrictions))
#     return row, column

# def pivot_step(tableau, pivot_position):
#     new_tableau = [[] for eq in tableau]

#     i, j = pivot_position
#     pivot_value = tableau[i][j]
#     new_tableau[i] = np.array(tableau[i]) / pivot_value

#     for eq_i, eq in enumerate(tableau):
#         if eq_i != i:
#             multiplier = np.array(new_tableau[i]) * tableau[eq_i][j]
#             new_tableau[eq_i] = np.array(tableau[eq_i]) - multiplier

#     return new_tableau
# def is_basic(column):
#     return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

# def get_solution(tableau):
#     columns = np.array(tableau).T
#     solutions = []
#     for column in columns[:-1]:
#         solution = 0
#         if is_basic(column):
#             one_index = column.tolist().index(1)
#             solution = columns[-1][one_index]
#         solutions.append(solution)

#     return solutions
# def simplex(c, A, b):
#     tableau = to_tableau(c, A, b)

#     while can_be_improved(tableau):
#         pivot_position = pivot(tableau)
#         tableau = pivot_step(tableau, pivot_position)

#     return get_solution(tableau)
# print(simplex(c,A,b))

class Simplex:
    def __init__(self, restriction_coefficients: list, restriction_constants: list, objective_function_coefficients: list, max: bool) -> None:
        if max == True:
            self.A = restriction_coefficients
            self.B = restriction_constants
            self.C = [-objective_function_coefficients[i]
                      for i in range(len(objective_function_coefficients))]
            self.objective_function = [
                objective_function_coefficients[i] for i in range(len(objective_function_coefficients))]
            for i in range(len(restriction_coefficients)):
                self.C.append(0)
                self.objective_function.append(0)
                for j in range(len(restriction_coefficients)):
                    self.A[i].append(1 if i == j else 0)
            self.rows = len(self.A)
            self.columns = len(self.A[0])
            self.maximize = max
            self.bounded = True
            self.optimal_value = 0
            self.optimal_solution = [
                0 for i in range(self.columns - self.rows)]
            self.basis = [i + 1 for i in range(self.rows, self.columns)]
            self.basis_value = [0 for i in range(self.rows)]
        else:
            restriction_coefficients = [[restriction_coefficients[j][i] for j in range(
                len(restriction_coefficients))] for i in range(len(restriction_coefficients[0]))]
            self.A = restriction_coefficients
            self.B = objective_function_coefficients
            self.C = [-restriction_constants[i]
                      for i in range(len(restriction_constants))]
            self.objective_function = [
                restriction_constants[i] for i in range(len(restriction_constants))]
            self.initial_objective_function = [objective_function_coefficients[i] for i in range(
                len(objective_function_coefficients))]
            for i in range(len(restriction_coefficients)):
                self.C.append(0)
                self.objective_function.append(0)
                for j in range(len(restriction_coefficients)):
                    self.A[i].append(1 if i == j else 0)
            self.rows = len(self.A)
            self.columns = len(self.A[0])
            self.maximize = max
            self.bounded = True
            self.optimal_value = 0
            self.optimal_solution = [
                0 for i in range(len(self.initial_objective_function))]
            self.basis = [i + 1 for i in range(self.rows, self.columns)]
            self.basis_value = [0 for i in range(self.rows)]

    def optimality(self) -> bool:
        for i in range(self.columns):
            if self.C[i] < 0:
                return False
        return True

    def find_pivot_column(self) -> int:
        index = 0
        minimum = self.C[0]
        for i in range(self.columns):
            if self.C[i] < minimum:
                index = i
                minimum = self.C[i]
        return index

    def find_pivot_row(self, pivot_column: int) -> int:
        ratios = [(self.B[i] / self.A[i][pivot_column] if (self.A[i]
                   [pivot_column] > 0 and self.B[i] > 0)else float("inf")) for i in range(self.rows)]
        index = 0
        minimum = ratios[0]
        amount_of_inf = 0
        for i in range(self.rows):
            if ratios[i] == float("inf"):
                amount_of_inf += 1
            else:
                if minimum > ratios[i]:
                    minimum = ratios[i]
                    index = i
        if amount_of_inf == self.rows:
            self.bounded = False
        return index
    

    def pivotting(self, pivot_row: int, pivot_column: int) -> None:
        pivot_value = self.A[pivot_row][pivot_column]
        self.B[pivot_row] /= pivot_value
        self.A[pivot_row] = [x / pivot_value for x in self.A[pivot_row]]
        for i in range(self.rows):
            if i != pivot_row:
                factor = self.A[i][pivot_column]
                self.B[i] -= factor * self.B[pivot_row]
                self.A[i] = [self.A[i][j] - factor * self.A[pivot_row][j]
                             for j in range(self.columns)]
        pivot_factor = self.C[pivot_column]
        self.C = [self.C[j] - pivot_factor * self.A[pivot_row][j]
                  for j in range(self.columns)]
        self.basis[pivot_row] = pivot_column + 1
        self.basis_value[pivot_row] = self.objective_function[pivot_column]
        self.optimal_value = 0
        for i in range(self.rows):
            self.optimal_value += self.B[i] * self.basis_value[i]

    def calculate(self):
        while not self.optimality():
            pivot_column = self.find_pivot_column()
            pivot_row = self.find_pivot_row(pivot_column)
            if not self.bounded:
                print("The solution is undounded")
                return None
            self.pivotting(pivot_row, pivot_column)

        if (self.maximize):
            for i in range(self.columns - self.rows):
                if self.basis[i] <= self.columns - self.rows:
                    self.optimal_solution[self.basis[i] -
                                          1] = self.basis_value[i]
        else:
            print(self.C)
            print(self.rows)
            print(self.columns)
            for i in range(len(self.initial_objective_function)):
                self.optimal_solution[i] = self.C[self.columns - self.rows + i]
            self.optimal_value = 0
            for i in range(len(self.initial_objective_function)):
                self.optimal_value += self.optimal_solution[i] * \
                    self.initial_objective_function[i]
        print("Optimal solution found.")
        print(f"Optimal value: {self.optimal_value}")
        print(f"Optimal solution: {self.optimal_solution}")
        return self.optimal_solution, self.optimal_value


# A = [[2, 3, 4, 0, 0, 0], [0, 1, 2, 3, 0, 0], [
#     1, 0, 1, 1, 1, 0], [0, 2, 1, 0, 3, 1], [1, 1, 1, 1, 1, 1, ]]
# B = [100, 90, 80, 120, 150]
# C = [4, 5, 3, 7, 2, 6]
# S = Simplex(A, B, C)
# S.calculate()

# A = [[1, 1, 1], [0, 1, 2], [-1, 2, 2]]
# B = [6, 8, 4]
# C = [2, 10, 8]
A = [[1.5, 1, 0, 2], [0, 2, 6, 4], [1, 1, 1, 1], [0.5, 0, 2.5, 1.5]]
B = [35, 120, 50, 75]
C = [1, 0.5, 2.5, 3]
S = Simplex(A, B, C, False)
S.calculate()
