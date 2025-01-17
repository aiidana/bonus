import numpy as np
import sympy as sp

def classification(mat,symbolic):
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
mat=[]
def input_mat(symbolic):
    n=int(input("enter the size of matrix:"))
    matrix=[]
    if symbolic:
        variables=input("your variable( for example:x) ")
        symbol=sp.symbols(variables)
    for i in range(n):
         row=input().split()
         if symbolic:
             matrix.append([sp.sympify(x,locals={variables:symbol}) for x in row])
         else:
             matrix.append(float(x) for x in row)
          
    mat.append(matrix)
    return mat 
symbolic=False
choice=input("использовать символическое вычисление? да/нет: ").strip()
if choice=="да":
    symbolic=True
else:
    symbolic=False 
inp=input_mat(symbolic)
  
result=classification(inp,symbolic)
print(f'classification of matrix: {result}')
# # Example 1: uxx + uyy + uzz = 0 (A = 1, C = 1, F = 1) 

# coeff_matrix1 =        ([[1, 0, 0], 
#                          [0, 1, 0], 
#                          [0, 0, 1]]) 
# print(f'uxx + uyy + uzz is', classification(coeff_matrix1,True))

# #example 2: utt-c^2(uxx+uyy)=0   
# c = sp.symbols('c')                       
# coeff_matrix2 = ([[1, 0, 0,0], 
#                   [0, -c**2, 0,0], 
#                   [0, 0, -c**2,0],
#                   [0,0,0,-c**2]]) 
# print(f'utt-c^2(uxx+uyy)=0  is', classification(coeff_matrix2,True))

# #example 3: ut-a(uxx+uyy)=0 (A=-a and C=-a) 
# a = sp.symbols('a')
# coeff_matrix3 = ([[0, 0, 0], 
#                          [0, -a, 0], 
#                          [0, 0, -a]]) 
# print(f' ut-a(uxx+uyy)=0  is', classification(coeff_matrix3,True))

# #example4: uxx-uxy+uyy=f(x,y) 
# coeff_matrix4 = ([[1, -1/2], 
#                          [-1/2, 1], 
#                          ]) 
# print(f'uxx-uxy+uyy=f(x,y) is', classification(coeff_matrix4,False))