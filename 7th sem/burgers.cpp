#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;

double phi(double x, double t, double nu) {
    double pi = 3.141592653589793;
    return exp(-(x - 4 * t) * (x - 4 * t) / (4 * nu * (t + 1))) +
        exp(-(x - 4 * t - 2 * pi) * (x - 4 * t - 2 * pi) / (4 * nu * (t + 1)));
}

double analytical(double x, double t, double nu) {
    double ph = phi(x, t, nu);
    double pi = 3.141592653589793;
    double dphdx = (-(x - 4 * t) / (2 * nu * (t + 1))) *
        exp(-(x - 4 * t) * (x - 4 * t) / (4 * nu * (t + 1))) +
        (-(x - 4 * t - 2 * pi) / (2 * nu * (t + 1))) *
        exp(-(x - 4 * t - 2 * pi) * (x - 4 * t - 2 * pi) / (4 * nu * (t + 1)));
    return -2 * nu * dphdx / ph + 4;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double pi = 3.141592653589793;
    const double L = 2.0 * pi;
    const double nu = 0.1;
    const int N = 1000;
    const double dx = L / (N - 1);
    const double dt = 1e-4;
    const double Tmax = 0.3;
    const int Nt = Tmax / dt;
    const double t_sim = Nt * dt;

    
    if (rank == 0) {
        vector<double> u(N, 0.0), u_new(N, 0.0);

        for (int i = 0; i < N; ++i) {
            double x = i * dx;
            double phi0 = exp(-x * x / (4 * nu)) +
                exp(-(x - 2 * pi) * (x - 2 * pi) / (4 * nu));
            double dphdx = (-x / (2 * nu)) * exp(-x * x / (4 * nu)) +
                (-(x - 2 * pi) / (2 * nu)) *
                exp(-(x - 2 * pi) * (x - 2 * pi) / (4 * nu));
            u[i] = -2 * nu * dphdx / phi0 + 4;
        }

        double t_start = MPI_Wtime();
        for (int n = 0; n < Nt; ++n) {
            for (int i = 1; i < N - 1; ++i) {
                double convection = u[i] * (u[i] - u[i - 1]) / dx;
                double diffusion = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx * dx);
                u_new[i] = u[i] - dt * convection + nu * dt * diffusion;
            }
            
            u_new[0] = u_new[N - 2];
            u_new[N - 1] = u_new[1];
            u.swap(u_new);
        }
        double t_end = MPI_Wtime();
        double seq_time = t_end - t_start;

        cout << "\n--- Sequential version ---\n";
        cout << "Sequential time = " << seq_time << " s\n";

        ofstream fs("burgers_seq.csv");
        fs << "x,numerical\n";
        for (int i = 0; i < N; ++i) {
            double x = i * dx;
            fs << x << "," << u[i] << "\n";
        }
        fs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
 

    int base = N / size;
    int rem = N % size;
    int local_N = (rank < rem) ? base + 1 : base;
    int start = (rank < rem) ? rank * (base + 1)
        : rem * (base + 1) + (rank - rem) * base;

    vector<double> u(local_N + 2, 0.0), u_new(local_N + 2, 0.0);

    for (int i = 1; i <= local_N; ++i) {
        double x = (start + i - 1) * dx;
        double phi0 = exp(-x * x / (4 * nu)) +
            exp(-(x - 2 * pi) * (x - 2 * pi) / (4 * nu));
        double dphdx = (-x / (2 * nu)) * exp(-x * x / (4 * nu)) +
            (-(x - 2 * pi) / (2 * nu)) *
            exp(-(x - 2 * pi) * (x - 2 * pi) / (4 * nu));
        u[i] = -2 * nu * dphdx / phi0 + 4;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int n = 0; n < Nt; ++n) {
        if (local_N == 0) continue;

        int left = (rank - 1 + size) % size;
        int right = (rank + 1) % size;

        double send_right = u[local_N];
        double recv_left;
        MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, right, 0,
            &recv_left, 1, MPI_DOUBLE, left, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        u[0] = recv_left;

        double send_left = u[1];
        double recv_right;
        MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, left, 1,
            &recv_right, 1, MPI_DOUBLE, right, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        u[local_N + 1] = recv_right;

        for (int i = 1; i <= local_N; ++i) {
            double convection = u[i] * (u[i] - u[i - 1]) / dx;
            double diffusion = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx * dx);
            u_new[i] = u[i] - dt * convection + nu * dt * diffusion;
        }

        u.swap(u_new);
    }

    double t1 = MPI_Wtime();
    double parallel_time = t1 - t0;

    vector<double> global_u;
    if (rank == 0) global_u.resize(N);

    if (rank == 0) {
        for (int i = 0; i < local_N; ++i) global_u[i] = u[i + 1];
        for (int src = 1; src < size; ++src) {
            int recv_N = (src < rem) ? base + 1 : base;
            int start_idx = (src < rem) ? src * (base + 1)
                : rem * (base + 1) + (src - rem) * base;
            if (recv_N > 0) {
                MPI_Recv(&global_u[start_idx], recv_N, MPI_DOUBLE,
                    src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else {
        if (local_N > 0)
            MPI_Send(&u[1], local_N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        ofstream fout("burgers_result.csv");
        fout << "x,numerical,analytical,error\n";
        double sumsq = 0.0;
        double maxerr = 0.0;
        for (int i = 0; i < N; ++i) {
            double x = i * dx;
            double ana = analytical(x, t_sim, nu);
            double err = fabs(global_u[i] - ana);
            sumsq += err * err;
            if (err > maxerr) maxerr = err;
            fout << x << "," << global_u[i] << "," << ana << "," << err << "\n";
        }
        fout.close();

        double rms = sqrt(sumsq / N);
        cout << "\n--- Parallel version ---\n";
        cout << fixed << setprecision(8);
        cout << "RMS error = " << rms << "\n";
        cout << "Max error = " << maxerr << "\n";
        cout << "Parallel time = " << parallel_time << " s\n";
    }

    MPI_Finalize();
    return 0;
}
