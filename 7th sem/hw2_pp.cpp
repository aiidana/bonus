#include <iostream>
using namespace std;

int main() {
    int n, m; 
    int k, x; 
    cout << "Enter the size of matrix 1 (rows cols): \n";
    cin >> n >> m;
    cout << "Enter the size of matrix 2 (rows cols): \n";
    cin >> k >> x;

    if (n <= 0 || m <= 0 || k <= 0 || x <= 0) {
        cout << "Error: size of matrix must be greater than zero\n";
        return 0;
    }
    if (m != k) {
        cout << "Error: number of columns of matrix1 must equal number of rows of matrix2\n";
        return 0;
    }

    int mat1[n][m];
    int mat2[k][x];
    int res[n][x]; // результат умножения

    cout << "Enter matrix1:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> mat1[i][j];
        }
    }

    cout << "Enter matrix2:\n";
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < x; j++) {
            cin >> mat2[i][j];
        }
    }

    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < x; j++) {
            res[i][j] = 0;
            for (int t = 0; t < m; t++) {
                res[i][j] += mat1[i][t] * mat2[t][j];
            }
        }
    }

    cout << "Result matrix:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < x; j++) {
            cout << res[i][j] << " ";
        }
        cout << endl;
    }

   


    return 0;
}
