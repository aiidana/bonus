#include <iostream>
#include <vector>
#include <random>
using namespace std;


vector<vector<int>> multiplyMatrix(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();       
    int m = A[0].size();     
    int k = B.size();        
    int x = B[0].size();     

    vector<vector<int>> C(n, vector<int>(x, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < x; j++) {
            for (int t = 0; t < m; t++) {
                C[i][j] += A[i][t] * B[t][j];
            }
        }
    }
    return C;
}

vector<int> multiplyMatrixVector(const vector<vector<int>> A, const vector<int> v) {
    int n = A.size();
    int m = A[0].size();

    vector<int> res(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res[i] += A[i][j] * v[j];
        }
    }
    return res;
}

int main() {
    int n, m, k, x;

    cout << "Enter the size of matrix 1 (rows cols): \n";
    cin >> n >> m;

    cout << "Enter the size of matrix 2 (rows cols): \n";
    cin >> k >> x;

    if (n <= 0 || m <= 0 || k <= 0 || x <= 0) {
        cout << "Error: size of matrix must be greater than zero\n";
        return 0;
    }
    if (m != k) {
        cout << "Error: number of columns of matrix1 must equal number of rows of matrix2 \n";
        return 0;
    }
    

    vector<vector<int>> A(n, vector<int>(m));
    vector<vector<int>> B(k, vector<int>(x));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            A[i][j] = rand() % 10; // числа от 0 до 9

    for (int i = 0; i < k; i++)
        for (int j = 0; j < x; j++)
            B[i][j] = rand() % 10;

    cout<< "Matrix 1: \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            cout << A[i][j] << " ";
        cout << "\n";
    }

    cout<< "Matrix2: \n";
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < x; j++)
            cout << B[i][j] << " ";
        cout << "\n";
    }

    // vector<vector<int>> A(n, vector<int>(m));
    // cout << "Enter matrix1:\n";
    // for (int i = 0; i < n; i++)
    //     for (int j = 0; j < m; j++)
    //         cin >> A[i][j];

    

    // vector<vector<int>> B(k, vector<int>(x));
    // cout << "Enter matrix2:\n";
    // for (int i = 0; i < k; i++)
    //     for (int j = 0; j < x; j++)
    //         cin >> B[i][j];

    
    

    cout << "\n result of A * B:\n";
    auto C = multiplyMatrix(A, B);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < x; j++)
            cout << C[i][j] << " ";
        cout << "\n";
    }

    // умножение матрицы на вектор
    cout << "\n enter size of vector ";
    int size;
    cin >> size;

    vector<int> v(size);
    cout << "enter vector:\n";
    for (int i = 0; i < size; i++)
        cin >> v[i];

    if (m != size) {
        cout << "error: number of column != size of vector!\n";
        return 0;
    }

    cout << "\nresult of  A * v:\n";
    auto Av = multiplyMatrixVector(A, v);
    for (int i = 0; i < Av.size(); i++)
        cout << Av[i] << " ";
    cout << "\n";

    return 0;
}
