#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
using namespace std;
 

double f(double x) {
    return 2 * exp(x) * cos(x);
}


double p_exact(double x) {
    return -exp(x) * sin(x) + (exp(1.0) * sin(1.0) - 1) * x + 1;
}

double solve_poisson_relax(int N, int rank, int size, bool save_result = false) {
    double a = 0.0, b = 1.0;
    double h = (b - a) / N;
    double omega = 1.7;     
    double tol = 1e-5;       
    int max_iter = 3500;     

  
    int base = (N + 1) / size;
    int remainder = (N + 1) % size;
    int local_n = base + (rank < remainder ? 1 : 0);
    int start = rank * base + min(rank, remainder);
    int end = start + local_n - 1;

    vector<double> x(local_n);
    for (int i = 0; i < local_n; ++i)
        x[i] = (start + i) * h;

    vector<double> p(local_n, 0.0);

    if (rank == 0) p[0] = 1.0;           
    if (rank == size - 1) p[local_n - 1] = 0.0; 

    bool converged = false;

    for (int iter = 0; iter < max_iter && !converged; ++iter) {
        double diff = 0.0;

        double left_val = 1.0, right_val = 0.0;
        MPI_Request reqs[4];
        int req_count = 0;

        if (rank > 0) {
            MPI_Isend(&p[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&left_val, 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (rank < size - 1) {
            MPI_Irecv(&right_val, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&p[local_n - 1], 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }

       
        for (int i = 1; i < local_n - 1; ++i) {
            double new_val = 0.5 * (p[i - 1] + p[i + 1] + h * h * f(x[i]));
            double relaxed = (1 - omega) * p[i] + omega * new_val;
            diff = max(diff, fabs(relaxed - p[i]));
            p[i] = relaxed;
        }

       
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

       
        if (local_n > 1) {
            if (rank > 0) {
                double new_val = 0.5 * (left_val + p[1] + h * h * f(x[0]));
                double relaxed = (1 - omega) * p[0] + omega * new_val;
                diff = max(diff, fabs(relaxed - p[0]));
                p[0] = relaxed;
            }

            if (rank < size - 1) {
                double new_val = 0.5 * (p[local_n - 2] + right_val + h * h * f(x[local_n - 1]));
                double relaxed = (1 - omega) * p[local_n - 1] + omega * new_val;
                diff = max(diff, fabs(relaxed - p[local_n - 1]));
                p[local_n - 1] = relaxed;
            }
        }

       
        double global_diff = diff;
        if (rank != 0)
            MPI_Send(&diff, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        else {
            for (int src = 1; src < size; ++src) {
                double tmp;
                MPI_Recv(&tmp, 1, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_diff = max(global_diff, tmp);
            }
            for (int dest = 1; dest < size; ++dest)
                MPI_Send(&global_diff, 1, MPI_DOUBLE, dest, 3, MPI_COMM_WORLD);
        }
        if (rank != 0)
            MPI_Recv(&global_diff, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (global_diff < tol) converged = true;
    }


    if (save_result) {
        if (rank == 0) {
            ofstream fout_num("num.txt");
            ofstream fout_ana("analytical.txt");

            for (int i = 0; i < local_n; ++i) {
                fout_num << x[i] << " " << p[i] << "\n";
                fout_ana << x[i] << " " << p_exact(x[i]) << "\n";
            }

            for (int src = 1; src < size; ++src) {
                int recv_n;
                MPI_Recv(&recv_n, 1, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<double> x_buf(recv_n), p_buf(recv_n);
                MPI_Recv(x_buf.data(), recv_n, MPI_DOUBLE, src, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(p_buf.data(), recv_n, MPI_DOUBLE, src, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < recv_n; ++i) {
                    fout_num << x_buf[i] << " " << p_buf[i] << "\n";
                    fout_ana << x_buf[i] << " " << p_exact(x_buf[i]) << "\n";
                }
            }

            fout_num.close();
            fout_ana.close();
            cout << "Results saved to num.txt and analytical.txt\n";
        }
        else {
            int send_n = local_n;
            MPI_Send(&send_n, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
            MPI_Send(x.data(), local_n, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
            MPI_Send(p.data(), local_n, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
        }
    }

   
    double local_err = 0.0;
    for (int i = 0; i < local_n; ++i)
        local_err = max(local_err, fabs(p[i] - p_exact(x[i])));

    if (rank != 0)
        MPI_Send(&local_err, 1, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
    else {
        for (int src = 1; src < size; ++src) {
            double tmp;
            MPI_Recv(&tmp, 1, MPI_DOUBLE, src, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_err = max(local_err, tmp);
        }
    }

    return local_err;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> N_values = { 50, 100, 200 };

    if (rank == 0) {
        cout << "  Poisson Equation ( Relaxation)\n";
        cout << setw(10) << "N" << setw(20) << "Max Error" << setw(20) << "seconds\n";
    }

    for (int N : N_values) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        double err = solve_poisson_relax(N, rank, size, N == 200);

        double elapsed = MPI_Wtime() - start;

        if (rank == 0) {
            cout << setw(10) << N << setw(20) << scientific << err
                << setw(20) << fixed << setprecision(6) << elapsed << endl;
        }
    }

    MPI_Finalize();
    return 0;
}



