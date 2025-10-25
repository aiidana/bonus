#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <mpi.h>

using namespace std;

const int Nx = 200;   
const double pi = 3.14159265358979323846;


double analytic_solution(double x, double t) {
    if (x >= 2 * t) {
        return x * t - 0.5 * t * t + cos(pi * (x - 2 * t));
    }
    else {
        double tau = t - x / 2.0;
        return x * t - 0.5 * t * t + exp(-tau) + 0.5 * tau * tau;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double dx = 1.0 / (Nx - 1);
    double dt = 0.001; 
    int Nt = (int)(1.0 / dt) + 1;

    
    int base = Nx / size;
    int remainder = Nx % size;
    int local_Nx = base + (rank < remainder ? 1 : 0);

    int offset = 0;
    for (int i = 0; i < rank; i++) {
        offset += base + (i < remainder ? 1 : 0);
    }

    vector<vector<double>> u(Nt, vector<double>(local_Nx));

    double start_time = MPI_Wtime();


    for (int i = 0; i < local_Nx; i++) {
        double x = (offset + i) * dx;
        u[0][i] = cos(pi * x);
    }

    
    for (int n = 0; n < Nt - 1; n++) {
        double t = n * dt;

      
        if (rank == 0) {
            u[n + 1][0] = exp(-t);
        }
        
        for (int i = 1; i < local_Nx; i++) {
            double x = (offset + i) * dx;
            u[n + 1][i] = u[n][i] - 2.0 * (dt / dx) * (u[n][i] - u[n][i - 1])
                + dt * (x + t);
        }

        
        if (size > 1) {
            MPI_Status status;
            double send_val, recv_val;

            
            if (rank < size - 1) {
                send_val = u[n + 1][local_Nx - 1];
                MPI_Send(&send_val, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            }

            if (rank > 0) {
                MPI_Recv(&recv_val, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
                u[n + 1][0] = recv_val;
            }
        }
    }

    double end_time = MPI_Wtime();

 
    vector<vector<double>> global_u;
    if (rank == 0) {
        global_u.resize(Nt, vector<double>(Nx));
    }

    
    for (int n = 0; n < Nt; n++) {
        if (rank == 0) {
            
            for (int i = 0; i < local_Nx; i++) {
                global_u[n][offset + i] = u[n][i];
            }

            for (int src = 1; src < size; src++) {
                int src_nx = base + (src < remainder ? 1 : 0);
                int src_offset = 0;
                for (int j = 0; j < src; j++) {
                    src_offset += base + (j < remainder ? 1 : 0);
                }

                vector<double> recv_buf(src_nx);
                MPI_Status status;
                MPI_Recv(recv_buf.data(), src_nx, MPI_DOUBLE, src, n, MPI_COMM_WORLD, &status);

                for (int i = 0; i < src_nx; i++) {
                    global_u[n][src_offset + i] = recv_buf[i];
                }
            }
        }
        else {
            MPI_Send(u[n].data(), local_Nx, MPI_DOUBLE, 0, n, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        ofstream file3d("results_3d.txt");
        if (file3d.is_open()) {
            for (int n = 0; n < Nt; n++) {
                double t = n * dt;
                for (int i = 0; i < Nx; i++) {
                    double x = i * dx;
                    double a = analytic_solution(x, t);
                    file3d << x << " " << t << " " << global_u[n][i] << " " << a << endl;
                }
            }
            file3d.close();
            cout << "3D data saved to results_3d.txt" << endl;
        }

        ofstream file2d("results_2d.txt");
        if (file2d.is_open()) {
            for (int i = 0; i < Nx; i++) {
                double x = i * dx;
                double a = analytic_solution(x, 1.0);
                file2d << x << " " << global_u[Nt - 1][i] << " " << a << endl;
            }
            file2d.close();
            cout << "2D data saved to results_2d.txt" << endl;
        }

        
        double max_err = 0.0, rms_err = 0.0;
        for (int i = 0; i < Nx; i++) {
            double x = i * dx;
            double a = analytic_solution(x, 1.0);
            double e = abs(global_u[Nt - 1][i] - a);
            max_err = max(max_err, e);
            rms_err += e * e;
        }
        rms_err = sqrt(rms_err / Nx);

        cout << "Results for " << size << " processes:" << endl;
        cout << "Maximum error: " << max_err << endl;
        cout << "RMS error: " << rms_err << endl;
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
