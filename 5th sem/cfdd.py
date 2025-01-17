import numpy as np
import sympy as sp

def classification_intervals(mat, symbol=None):
    if symbol is not None:
        mat = sp.Matrix(mat)
        eigenvalues = mat.eigenvals()
        eigenvalues = list(eigenvalues.keys())

        critical_points = []
        for ev in eigenvalues:
            roots = sp.solve(ev, symbol)
            critical_points.extend(roots)

        critical_points = sorted(set(critical_points), key=lambda x: sp.re(x))

        if not critical_points:
            print("No critical points found (no roots for eigenvalues).")
        
        intervals = []
        bounds = [-sp.oo] + critical_points + [sp.oo]

        for i in range(len(bounds) - 1):
            lower_bound = bounds[i]
            upper_bound = bounds[i + 1]

            if i != 0 and sp.simplify(upper_bound - lower_bound) == 0:
                classification = "Parabolic"
            else:
                if lower_bound == -sp.oo:
                    test_value = upper_bound - 1
                elif upper_bound == sp.oo:
                    test_value = lower_bound + 1
                else:
                    test_value = (lower_bound + upper_bound) / 2

                signs = [sp.sign(ev.subs(symbol, test_value)) for ev in eigenvalues]

                if all(sign == 1 for sign in signs) or all(sign == -1 for sign in signs):
                    classification = "Elliptic"
                else:
                    classification = "Hyperbolic"

            intervals.append((lower_bound, upper_bound, classification))
        
        return intervals
    
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
        return matrix, symbol
    else:
        for i in range(n):
            row = input().split()
            matrix.append([float(x) for x in row])
        return matrix, None


symbolic = input("Использовать символическое вычисление? yes/no: ").strip().lower() == "yes"
if symbolic:
    inp, symbol = input_mat(symbolic)
    intervals = classification_intervals(inp, symbol)

    for lower_bound, upper_bound, classification in intervals:
        if lower_bound == -sp.oo:
            lower_bound_str = "-∞"
        else:
            lower_bound_str = str(lower_bound)
        
        if upper_bound == sp.oo:
            upper_bound_str = "∞"
        else:
            upper_bound_str = str(upper_bound)
        
        print(f'From {lower_bound_str} to {upper_bound_str}, the matrix is {classification}')
else:
    inp, _ = input_mat(symbolic)
    result = classification_intervals(inp)  
    print(f'Classification of matrix: {result}')

