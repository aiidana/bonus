
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


num = np.loadtxt('num.csv', delimiter=',')
exact = np.loadtxt('exact.csv', delimiter=',')
err = np.loadtxt('err.csv', delimiter=',')

NY, NX = num.shape
x = np.linspace(0, 2.0, NX)
y = np.linspace(0, 2.0, NY)
X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, num, cmap='viridis')
ax1.set_title('Numerical Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x,y)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)


ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Y, exact, cmap='plasma')
ax2.set_title('Analytical Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u(x,y)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)


ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X, Y, err, cmap='inferno')
ax3.set_title(' Error (num - exact)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Error')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()



# #include <mpi.h>
# #include <iostream>
# #include <vector>
# #include <cmath>
# #include <fstream>
# #include <iomanip>
# using namespace std;

# const double PI = 3.141592653589793;
# const double Lx = 2.0;
# const double Ly = 2.0;
# const double nu = 0.1;
# const double t_final = 0.5;
# const int NX = 101;
# const int NY = 101;


# inline int idx(int ix, int iy) { return iy * NX + ix; }

# double init_u(double x, double y) {
#     if (x > 0.5 && x < 1.0 && y > 0.5 && y < 1.0) return 2.0;
#     return 1.0;
# }

# double compute_Amn(int m, int n) {
#     double mp = double(m);
#     double np = double(n);
#     double termx = cos(PI * mp * 0.5 / Lx) - cos(PI * mp * 1.0 / Lx);
#     double termy = cos(PI * np * 0.5 / Ly) - cos(PI * np * 1.0 / Ly);
#     double Amn = (4.0 / (PI * PI * mp * np)) * termx * termy;
#     return Amn;
# }

# double analytical_u(double x, double y, double t) {
#     double v = 0.0;
#     for (int m = 1; m <= 40; ++m) {
#         double sin_mx = sin(PI * m * x / Lx);
#         for (int n = 1; n <= 40; ++n) {
#             double sin_ny = sin(PI * n * y / Ly);
#             double Amn = compute_Amn(m, n);
#             double lambda = (double(m * m) + double(n * n)) * (PI * PI) / (Lx * Lx);
#             double decay = exp(-nu * lambda * t);
#             v += Amn * sin_mx * sin_ny * decay;
#         }
#     }
#     return 1.0 + v;
# }

# void write_csv(const string& filename, const vector<double>& arr, int NXglob = NX, int NYglob = NY) {
#     ofstream f(filename);
#     f << fixed << setprecision(9);
#     for (int iy = 0; iy < NYglob; ++iy) {
#         for (int ix = 0; ix < NXglob; ++ix) {
#             f << arr[iy * NXglob + ix];
#             if (ix < NXglob - 1) f << ",";
#         }
#         f << "\n";
#     }
#     f.close();
# }

# void solve_sequential(vector<double>& u_out, double& seq_elapsed) {
#     double dx = Lx / (NX - 1);
#     double dy = Ly / (NY - 1);

#     double dt_stable = (dx * dx * dy * dy) / (2.0 * nu * (dx * dx + dy * dy));
#     double dt = 0.25 * dt_stable;
#     int nsteps = max(1, (int)ceil(t_final / dt));
#     dt = t_final / double(nsteps);

#     vector<double> u(NX * NY), u_new(NX * NY);

#     for (int iy = 0; iy < NY; ++iy) {
#         double y = iy * dy;
#         for (int ix = 0; ix < NX; ++ix) {
#             double x = ix * dx;
#             u[idx(ix, iy)] = init_u(x, y);
#         }
#     }

#     double t0 = MPI_Wtime();
#     for (int step = 0; step < nsteps; ++step) {
#         for (int iy = 1; iy < NY - 1; ++iy) {
#             for (int ix = 1; ix < NX - 1; ++ix) {
#                 double uij = u[idx(ix, iy)];
#                 double u_xx = (u[idx(ix + 1, iy)] - 2.0 * uij + u[idx(ix - 1, iy)]) / (dx * dx);
#                 double u_yy = (u[idx(ix, iy + 1)] - 2.0 * uij + u[idx(ix, iy - 1)]) / (dy * dy);
#                 u_new[idx(ix, iy)] = uij + dt * nu * (u_xx + u_yy);
#             }
#         }
#         // Dirichlet  = 1
#         for (int ix = 0; ix < NX; ++ix) {
#             u_new[idx(ix, 0)] = 1.0;
#             u_new[idx(ix, NY - 1)] = 1.0;
#         }
#         for (int iy = 0; iy < NY; ++iy) {
#             u_new[idx(0, iy)] = 1.0;
#             u_new[idx(NX - 1, iy)] = 1.0;
#         }
#         u.swap(u_new);
#     }
#     double t1 = MPI_Wtime();
#     seq_elapsed = t1 - t0;
#     u_out = move(u);
# }

# void solve_parallel(int rank, int size, vector<double>& local_u_out, double& local_elapsed) {
#     double dx = Lx / (NX - 1);
#     double dy = Ly / (NY - 1);

#     double dt_stable = (dx * dx * dy * dy) / (2.0 * nu * (dx * dx + dy * dy));
#     double dt = 0.25 * dt_stable;
#     int nsteps = max(1, (int)ceil(t_final / dt));
#     dt = t_final / double(nsteps);

#     int base = NY / size;
#     int rem = NY % size;
#     int local_ny = base + (rank < rem ? 1 : 0);
#     int y_start = rank * base + min(rank, rem);
#     int y_end = y_start + local_ny - 1;

    
#     vector<double> u((local_ny + 2) * NX, 1.0);
#     vector<double> u_new((local_ny + 2) * NX, 1.0);

#     auto local_idx = [&](int ix, int iy_local) { return iy_local * NX + ix; };

    
#     for (int iy_local = 1; iy_local <= local_ny; ++iy_local) {
#         int iy_global = y_start + (iy_local - 1);
#         double y = iy_global * dy;
#         for (int ix = 0; ix < NX; ++ix) {
#             double x = ix * dx;
#             u[local_idx(ix, iy_local)] = init_u(x, y);
#         }
#     }
   
#     for (int iy_local = 1; iy_local <= local_ny; ++iy_local) {
#         u[local_idx(0, iy_local)] = 1.0;
#         u[local_idx(NX - 1, iy_local)] = 1.0;
#     }
#     if (y_start == 0) for (int ix = 0; ix < NX; ++ix) u[local_idx(ix, 1)] = 1.0;
#     if (y_end == NY - 1) for (int ix = 0; ix < NX; ++ix) u[local_idx(ix, local_ny)] = 1.0;

#     int up = (rank == 0 ? MPI_PROC_NULL : rank - 1);
#     int down = (rank == size - 1 ? MPI_PROC_NULL : rank + 1);

#     MPI_Barrier(MPI_COMM_WORLD);
#     double t0 = MPI_Wtime();

   
#     for (int step = 0; step < nsteps; ++step) {
       
#         MPI_Request reqs[4];
#         int rcount = 0;

       
#         if (up != MPI_PROC_NULL) {
#             MPI_Irecv(&u[local_idx(0, 0)], NX, MPI_DOUBLE, up, 101, MPI_COMM_WORLD, &reqs[rcount++]);
#             MPI_Isend(&u[local_idx(0, 1)], NX, MPI_DOUBLE, up, 102, MPI_COMM_WORLD, &reqs[rcount++]);
#         }
#         else {
#             for (int ix = 0; ix < NX; ++ix) u[local_idx(ix, 0)] = 1.0;
#         }

        
#         if (down != MPI_PROC_NULL) {
#             MPI_Irecv(&u[local_idx(0, local_ny + 1)], NX, MPI_DOUBLE, down, 102, MPI_COMM_WORLD, &reqs[rcount++]);
#             MPI_Isend(&u[local_idx(0, local_ny)], NX, MPI_DOUBLE, down, 101, MPI_COMM_WORLD, &reqs[rcount++]);
#         }
#         else {
#             for (int ix = 0; ix < NX; ++ix) u[local_idx(ix, local_ny + 1)] = 1.0;
#         }

#         if (rcount > 0) MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);

       
#         for (int iy_local = 1; iy_local <= local_ny; ++iy_local) {
#             int iy_global = y_start + (iy_local - 1);
#             if (iy_global == 0 || iy_global == NY - 1) continue;
#             for (int ix = 1; ix < NX - 1; ++ix) {
#                 double uij = u[local_idx(ix, iy_local)];
#                 double u_xx = (u[local_idx(ix + 1, iy_local)] - 2.0 * uij + u[local_idx(ix - 1, iy_local)]) / (dx * dx);
#                 double u_yy = (u[local_idx(ix, iy_local + 1)] - 2.0 * uij + u[local_idx(ix, iy_local - 1)]) / (dy * dy);
#                 u_new[local_idx(ix, iy_local)] = uij + dt * nu * (u_xx + u_yy);
#             }
            
#             u_new[local_idx(0, iy_local)] = 1.0;
#             u_new[local_idx(NX - 1, iy_local)] = 1.0;
#         }

  
#         if (y_start == 0) for (int ix = 0; ix < NX; ++ix) u_new[local_idx(ix, 1)] = 1.0;
#         if (y_end == NY - 1) for (int ix = 0; ix < NX; ++ix) u_new[local_idx(ix, local_ny)] = 1.0;

#         u.swap(u_new);
#     }

#     double t1 = MPI_Wtime();
#     local_elapsed = t1 - t0;

  
#     local_u_out.resize(local_ny * NX);
#     for (int iy_local = 1; iy_local <= local_ny; ++iy_local) {
#         for (int ix = 0; ix < NX; ++ix) {
#             local_u_out[(iy_local - 1) * NX + ix] = u[local_idx(ix, iy_local)];
#         }
#     }
# }

# void collect(int rank, int size, const vector<double>& local_u, double parallel_time, double seq_time) {
    
#     int base = NY / size;
#     int rem = NY % size;
#     vector<int> rows_per_rank(size);
#     for (int r = 0; r < size; ++r) rows_per_rank[r] = base + (r < rem ? 1 : 0);

#     vector<double> global_u;
#     if (rank == 0) global_u.resize(NX * NY);

   
#     if (rank == 0) {
        
#         int offset = 0;
#         int my_rows = rows_per_rank[0];
#         for (int i = 0; i < my_rows * NX; ++i) global_u[offset + i] = local_u[i];
#         offset += my_rows * NX;

        
#         for (int r = 1; r < size; ++r) {
#             int cnt = rows_per_rank[r] * NX;
#             MPI_Recv(&global_u[offset], cnt, MPI_DOUBLE, r, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#             offset += cnt;
#         }
#     }
#     else {
#         MPI_Send(local_u.data(), (int)local_u.size(), MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
#     }

   
#     double parallel_max = 0.0;
#     if (rank == 0) {
#         parallel_max = parallel_time; 
#         for (int r = 1; r < size; ++r) {
#             double tmp;
#             MPI_Recv(&tmp, 1, MPI_DOUBLE, r, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#             if (tmp > parallel_max) parallel_max = tmp;
#         }
#     }
#     else {
#         MPI_Send(&parallel_time, 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
#     }

    
#     if (rank == 0) {
#         vector<double> exact(NX * NY);
#         vector<double> err(NX * NY);
#         double dx = Lx / (NX - 1);
#         double dy = Ly / (NY - 1);
#         double err_sum = 0.0;
#         double err_max = 0.0;
#         for (int iy = 0; iy < NY; ++iy) {
#             double y = iy * dy;
#             for (int ix = 0; ix < NX; ++ix) {
#                 double x = ix * dx;
#                 exact[idx(ix, iy)] = analytical_u(x, y, t_final);
#                 err[idx(ix, iy)] = fabs(global_u[idx(ix, iy)] - exact[idx(ix, iy)]);
#                 err_sum += err[idx(ix, iy)];
#                 err_max = max(err_max, err[idx(ix, iy)]);
#             }
#         }
#         double average_err = err_sum / (NX * NY);

#         write_csv("num.csv", global_u);
#         write_csv("exact.csv", exact);
#         write_csv("err.csv", err);

#         cout << "Wrote num.csv, exact.csv, err.csv\n";
#         cout << fixed << setprecision(9);
#         cout << "Average absolute error = " << average_err << "\n";
#         cout << "Maximum absolute error = " << err_max << "\n";
#         cout << fixed << setprecision(6);
#         cout << "Parallel time (max across ranks): " << parallel_max << " s\n";
#         if (parallel_max > 1e-12) cout << "Speedup (serial/parallel) = " << (seq_time / parallel_max) << "\n";
#         else cout << "Parallel time too small to compute speedup\n";
#     }
# }

# int main(int argc, char* argv[]) {
#     MPI_Init(&argc, &argv);

#     int rank = 0, size = 1;
#     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#     MPI_Comm_size(MPI_COMM_WORLD, &size);

#     double seq_time = 0.0;
#     vector<double> serial_u;
#     if (rank == 0) {
#         double t0 = MPI_Wtime();
#         solve_sequential(serial_u, seq_time);
#         double t1 = MPI_Wtime();
#         seq_time = t1 - t0;
#         cout << "Sequential solver (on rank 0) finished in " << seq_time << " s\n";
#     }

#     MPI_Barrier(MPI_COMM_WORLD);

#     vector<double> local_u;
#     double local_par_time = 0.0;
#     solve_parallel(rank, size, local_u, local_par_time);

    
#     collect(rank, size, local_u, local_par_time, seq_time);

#     MPI_Finalize();
#     return 0;
# }

# ///