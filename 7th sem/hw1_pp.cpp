#include <iostream>
#include <iomanip>
using namespace std;

int main() {
   
    double e = 1.0;
    double fact = 1.0;
    int terms_e = 100; // количество членов ряда для e

    for (int n = 1; n < terms_e; n++) {
        fact *= n;           
        e += 1.0 / fact;     
    }

    
    double pi = 0;
    int terms_pi = 2000000; // количество членов ряда для pi

    for (int n = 0; n < terms_pi; n++) {
        if (n % 2 == 0)
            pi += 1.0 / (2.0 * n + 1.0);
        else
            pi -= 1.0/ (2.0 * n + 1.0);
    }
    pi *= 4;

    cout <<  setprecision(15);
    cout << "e  =" << e << endl;
    cout << "pi = " << pi << endl;

    return 0;
}
