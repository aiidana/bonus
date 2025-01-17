# import numpy as np
# import sympy as sp

# def classification(mat, symbolic, variable_value=None):
#     if symbolic:
#         mat = sp.Matrix(mat)  
#         if variable_value is not None:
#             mat = mat.subs(symbol, variable_value)  # Подстановка значения переменной
        
#         eigenvalues = mat.eigenvals() 
#         eigenvalues = list(eigenvalues.keys()) 
#         print(f'Eigenvalues: {eigenvalues}')

#         signs = [sp.sign(ev) for ev in eigenvalues] 

#         if all(sign == 1 for sign in signs) or all(sign == -1 for sign in signs):
#             return "Elliptic"
#         elif any(sign == 0 for sign in signs):
#             return "Parabolic"
#         else:
#             return "Hyperbolic"
#     else:
#         mat = np.array(mat)  
#         eigenvalues, _ = np.linalg.eig(mat)  
#         if all(ev > 0 for ev in eigenvalues) or all(ev < 0 for ev in eigenvalues):
#             return "Elliptic"
#         elif any(ev == 0 for ev in eigenvalues):
#             return "Parabolic"
#         else:
#             return "Hyperbolic"

# def input_mat(symbolic):
#     n = int(input("Enter the size of matrix: "))
#     matrix = []

#     if symbolic:
#         global symbol  # Используем глобальную переменную для символа
#         variables = input("Enter your variable (for example: x): ")
#         symbol = sp.symbols(variables)

#         for i in range(n):
#             row = input().split()
#             matrix.append([sp.sympify(x, locals={variables: symbol}) for x in row])
#     else:
#         for i in range(n):
#             row = input().split()
#             matrix.append([float(x) for x in row]) 
#     return matrix  


# # Основная часть программы
# symbolic = input("Использовать символическое вычисление? да/нет: ").strip().lower() == "да"
# inp = input_mat(symbolic)

# if symbolic:
#     variable_value = float(input(f"Введите значение для переменной {symbol}: "))  # Запрос значения переменной
#     result = classification(inp, symbolic, variable_value)
# else:
#     result = classification(inp, symbolic)

# print(f'Classification of matrix: {result}')
import sympy as sp

def classification_intervals(mat, symbol):
    mat = sp.Matrix(mat)
    eigenvalues = mat.eigenvals()
    eigenvalues = list(eigenvalues.keys())
    
    
    critical_points = []
    for ev in eigenvalues:
        
        roots = sp.solve(ev, symbol)
        critical_points.extend(roots)
    
    
    critical_points = sorted(set(critical_points))
    
    print(f'Critical points: {critical_points}')
    
    
    intervals = []
    for i in range(len(critical_points) + 1):
        if i == 0:
            test_value = critical_points[0] - 1  
        elif i == len(critical_points):
            test_value = critical_points[-1] + 1  
        else:
            test_value = (critical_points[i - 1] + critical_points[i]) / 2 
        
       
        signs = [sp.sign(ev.subs(symbol, test_value)) for ev in eigenvalues]
        
        if all(sign == 1 for sign in signs) or all(sign == -1 for sign in signs):
            classification = "Elliptic"
        elif any(sign == 0 for sign in signs):
            classification = "Parabolic"
        else:
            classification = "Hyperbolic"
        
        intervals.append((test_value, classification))
    
    return intervals

def input_mat(symbolic):
    n = int(input("Enter the size of matrix: "))
    matrix = []

    if symbolic:
        variables = input("Enter your variable (for example: x): ")
        symbol = sp.symbols(variables)

        for i in range(n):
            row = input().split()
            matrix.append([sp.sympify(x, locals={variables: symbol}) for x in row])
        return matrix, symbol
    else:
        for i in range(n):
            row = input().split()
            matrix.append([float(x) for x in row]) 
        return matrix, None


symbolic = input("Использовать символическое вычисление? да/нет: ").strip().lower() == "да"
if symbolic:
    inp, symbol = input_mat(symbolic)
    intervals = classification_intervals(inp, symbol)

    
    for value, classification in intervals:
        print(f'For {symbol} = {value}, the matrix is {classification}')
else:
    print("This functionality is only available for symbolic computation.")
