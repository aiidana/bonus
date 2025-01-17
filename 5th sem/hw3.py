import numpy as np
import sympy as sp

def classification(mat, symbolic):
    if symbolic:
        mat = sp.Matrix(mat)  
        eigenvalues = mat.eigenvals() 
        eigenvalues = list(eigenvalues.keys()) 

        signs = [sp.sign(ev) for ev in eigenvalues] 

        if all(sign == 1 for sign in signs) or all(sign == -1 for sign in signs):
            return "Elliptic"
        elif any(sign == 0 for sign in signs):
            return "Parabolic"
        else:
            return "Hyperbolic"
    else:
        mat = np.array(mat)  
        eigenvalues, _ = np.linalg.eig(mat)  
        if all(ev > 0 for ev in eigenvalues) or all(ev < 0 for ev in eigenvalues):
            return "Elliptic"
        elif any(ev == 0 for ev in eigenvalues):
            return "Parabolic"
        else:
            return "Hyperbolic"

def input_mat(symbolic):
    n = int(input("Enter the size of matrix: "))
    matrix = []

    if symbolic:
        variables = input("Enter your variable (for example: x): ")
        symbol = sp.symbols(variables)

        for i in range(n):
            row = input().split()
            matrix.append([sp.sympify(x, locals={variables: symbol}) for x in row])
    else:
        for i in range(n):
            row = input().split()
            matrix.append([float(x) for x in row]) 
    return matrix  


symbolic = input("Использовать символическое вычисление? да/нет: ").strip().lower() == "да"

inp = input_mat(symbolic)


result = classification(inp, symbolic)
print(f'Classification of matrix: {result}')
