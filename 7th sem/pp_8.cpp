#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;
//cos
//double analytical(double x, double t, double nu, int Nterms = 1000) {
//    double L = 2.0;
//    double pi = 3.141592653589793;
//    double a0 = 1.25;
//    double sum = 0.0;
//    for (int n = 1; n <= Nterms; ++n) {
//        double an = (2.0 / (n * pi)) * (sin(n * pi / 2.0) - sin(n * pi / 4.0));
//        double lambda = (n * pi / L);
//        sum += an * cos(lambda * x) * exp(-nu * lambda * lambda * t);
//    }
//    return a0 + sum;
//}
//sin
//double analytical(double x, double t, double nu, int Nterms = 100) {
//    double L = 2.0;
//    double sum = 0.0;
//    double pi = 3.141592653589793;
//    for (int n = 1; n <= Nterms; n++) {
//        double An = (- 2 * cos(n * pi / 2) / (n * pi)) + (2 * cos(n * pi / 4) / (n * pi))- (2*cos(n*pi)/(n*pi))+(2/(n*pi));
//        sum += An * exp(-nu * pow(pi,2) * pow(n,2) * t/4) * sin(n * pi * x / L);
//    }
//    return  sum; 
//}
// mixed sine
double analytical(double x, double t, double nu, int Nterms = 1000) {
    double L = 2.0;
    double pi = 3.141592653589793;
    
    double sum = 0.0;
    for (int n = 0; n <= Nterms; ++n) {
        
        double an = 4 * cos((2 * n + 1) * pi / 8) / ((2 * n + 1) * pi) - 4 * cos((2 * n + 1) * pi / 4) / ((2 * n + 1) * pi);
        sum += an * sin((2*n+1)*pi * x/4) * exp(-nu * pow((2*n+1),2)*pow(pi,2) * t/16);
    }
    return 1+sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double L = 2.0;
    const double nu = 0.3;
    const int N = 101;
    const double dx = L / (N - 1);
    const double dt = 1e-4;
    const double r = nu * dt / (dx * dx);
    const double Tmax = 0.3;
    const int Nt = (int)round(Tmax / dt);

    if (rank == 0) {
        cout << fixed << setprecision(6);
        cout << "N=" << N << " dx=" << dx << " dt=" << dt << " Nt=" << Nt << " r=" << r << "\n";
    }

    vector<double> times_to_save = { 0.03, 0.08, 0.3 };

    int base = N / size;
    int rem = N % size;
    int local_N = (rank < rem) ? base + 1 : base;
    int start_idx = (rank < rem)
        ? rank * (base + 1)
        : rem * (base + 1) + (rank - rem) * base;

    vector<double> u(local_N + 2, 1.0), u_new(local_N + 2, 1.0);

    
    for (int i = 1; i <= local_N; ++i) {
        int gi = start_idx + (i - 1);
        double x = gi * dx;
        if (x >= 0.5 && x <= 1.0) {
            u[i] = 2.0;
        }
        else {
            u[i] = 1.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double current_time = 0.0;
    size_t save_idx = 0;
    for (int n = 0; n <= Nt; ++n) {
        if (save_idx < times_to_save.size()) {
            double ttarget = times_to_save[save_idx];
            if (fabs(current_time - ttarget) < 0.5 * dt || current_time > ttarget) {

                vector<double> local_res(local_N);
                for (int i = 0; i < local_N; ++i) local_res[i] = u[i + 1];

                if (rank == 0) {
                    vector<double> global_res(N);
                    
                    for (int i = 0; i < local_N; ++i) {
                        global_res[i] = local_res[i];
                    }
                   
                    for (int src = 1; src < size; ++src) {
                        int recv_N = (src < rem) ? base + 1 : base;
                        int start_j = (src < rem) ? src * (base + 1)
                            : rem * (base + 1) + (src - rem) * base;
                        MPI_Recv(&global_res[start_j], recv_N, MPI_DOUBLE, src, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }

                    
                    double twrite = current_time;
                    string fname = string("t_") + to_string((int)round(twrite*1000)) + ".csv";
                    ofstream fout(fname);
                    fout << "x,numerical,analytical,abs_error\n";
                    double sumsq = 0.0, maxe = 0.0;
                    for (int gi = 0; gi < N; ++gi) {
                        double x = gi * dx;
                        double num = global_res[gi];
                        double ana = analytical(x, twrite, nu, 500);
                        double err = fabs(num - ana);
                        sumsq += err * err;
                        if (err > maxe) maxe = err;
                        fout << x << "," << num << "," << ana << "," << err << "\n";
                    }
                    fout.close();
                    double rms = sqrt(sumsq / N);
                    cout << "t=" << twrite
                        << "  RMS_error=" << rms << "  max_err=" << maxe << "\n";
                }
                else {
                    MPI_Send(local_res.data(), local_N, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD);
                }

                save_idx++;
                
            }
        }

        if (n == Nt) break;

       
        /*if (rank == 0) u[0] = u[1];*/
        if (rank == 0) u[0] = 1;
        if (rank == size - 1) u[local_N + 1] = u[local_N];
        /*u_new[0] = 1;
        u_new[local_N - 1] = 1;*/
        

        MPI_Request reqs[4];
        int rc = 0;
        if (rank > 0) {
            MPI_Irecv(&u[0], 1, MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &reqs[rc++]);
            MPI_Isend(&u[1], 1, MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &reqs[rc++]);
        }
        if (rank < size - 1) {
            MPI_Irecv(&u[local_N + 1], 1, MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &reqs[rc++]);
            MPI_Isend(&u[local_N], 1, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &reqs[rc++]);
        }
        if (rc > 0) MPI_Waitall(rc, reqs, MPI_STATUSES_IGNORE);

        for (int i = 1; i <= local_N; ++i)
            u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);

        u.swap(u_new);
        current_time += dt;
    }

    double t1 = MPI_Wtime();
    double parallel_time = t1 - t0;

    double sequential_time = 0.0;

    if (rank == 0) {
        
        double seq_t0 = MPI_Wtime();

        vector<double> u_seq(N, 1.0), u_new_seq(N, 1.0);
        for (int i = 0; i < N; ++i) {
            double x = i * dx;
            if (x >= 0.5 && x <= 1.0) u_seq[i] = 2.0;
            else u_seq[i] = 1.0;
        }

        double tcur = 0.0;
        for (int n = 0; n < Nt; ++n) {
            for (int i = 1; i < N - 1; ++i)
                u_new_seq[i] = u_seq[i] + r * (u_seq[i + 1] - 2 * u_seq[i] + u_seq[i - 1]);
            u_new_seq[0] = u_new_seq[1];
            u_new_seq[N - 1] = u_new_seq[N - 2];
            u_seq.swap(u_new_seq);
            tcur += dt;
        }

        double seq_t1 = MPI_Wtime();
        sequential_time = seq_t1 - seq_t0;

        double speedup = sequential_time / parallel_time;

        
        cout << "Sequential time: " << sequential_time << " s\n";
        cout << "Parallel (MPI) time: " << parallel_time << " s\n";
        cout << "Speedup = " << speedup << "x faster\n";
    }

    MPI_Finalize();
    return 0;
}

