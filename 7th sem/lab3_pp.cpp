#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm> // for max
using namespace std;

const int N = 100;       
const int ITER = 10000;    
const double PI = 3.14159265359;


void initial_cond(double x[], double u[]) {
    for (int i = 0; i < N; i++) {
        u[i] = sin(2 * PI * x[i]);
    }
}

// explicit 
void explicitStep(double u_old[], double u_new[], double C, double dt, double dx) {
    for (int i = 1; i < N; i++) {
        u_new[i] = u_old[i] - (C * dt / dx) * (u_old[i] - u_old[i - 1]);
    }
    u_new[0] = 0.0; 
    // u_new[0] = u_old[0];
}

// implicit 
void implicitStep(double u_old[], double u_new[], double c, double dt, double dx) {
    double alpha[N], beta[N];

    
    double A = 0;
    double B = 1.0 / dt + c / dx;
    double C=-c/dx;

    // Thomas algorithm
    alpha[1] = 0.0;
    beta[1] = 0.0;
    for (int i = 1; i < N - 1; i++) {
        double denom = B + C * alpha[i];
        alpha[i + 1] = -A / denom;
        double D = u_old[i] / dt;
        beta[i + 1] = (D - C * beta[i]) / denom;
    }

    u_new[N - 1] = 0.0;

    for (int i = N - 2; i >= 0; i--) {
        u_new[i] = alpha[i + 1] * u_new[i + 1] + beta[i + 1];
    }
}

int main() {
    double dx = 1.0 / (N - 1);
    double dt = 0.0005 * dx; // Time step size
    double C = 1.0;           

    double x[N], u_explicit_old[N], u_implicit_old[N], u_explicit_new[N], u_implicit_new[N];

    // initial_cond spatial grid
    for (int i = 0; i < N; i++) {
        x[i] = i * dx;
    }

    // initial_cond wave profiles
    initial_cond(x, u_explicit_old);
    initial_cond(x, u_implicit_old);

    for (int t = 0; t < ITER; t++) {
        explicitStep(u_explicit_old, u_explicit_new, C, dt, dx);
        
        for (int i = 0; i < N; i++) {
            u_explicit_old[i] = u_explicit_new[i];
        }
    }

    for (int t = 0; t < ITER; t++) {
        implicitStep(u_implicit_old, u_implicit_new, C, dt, dx);
        
        for (int i = 0; i < N; i++) {
            u_implicit_old[i] = u_implicit_new[i];
        }
    }

    
    ofstream fout("result.dat");
    double total_time = ITER * dt;
    for (int i = 0; i < N; i++) {
        double current_x = x[i];
        double analytic = sin(2 * PI * (current_x - C * total_time));
        fout << current_x << "\t" << u_explicit_new[i] << "\t" << u_implicit_new[i] << "\t" << analytic << "\n";
    }
    fout.close();


    double max_error_explicit = 0.0;
    double max_error_implicit = 0.0;
    for (int i = 0; i < N; i++) {
        double current_x = x[i];
        double analytic = sin(2 * PI * (current_x - C * total_time));
        max_error_explicit = max(max_error_explicit, fabs(u_explicit_new[i] - analytic));
        max_error_implicit = max(max_error_implicit, fabs(u_implicit_new[i] - analytic));
    }

    cout << "Max error (Explicit): " << max_error_explicit << endl;
    cout << "Max error (Implicit): " << max_error_implicit << endl;

    return 0;
}