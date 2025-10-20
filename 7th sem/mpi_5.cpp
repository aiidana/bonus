#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
////1
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int number = rank;
    int received_number;
    if (size < 2) {
        if (rank == 0) {
            cout << "you should use at least 2 proccesses!" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    /*for (int i = 0; i < size; i++) {
        MPI_Ssend(&number, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
        MPI_Recv(&received_number, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        number = received_number + rank;
        cout << "Process " << rank << " received " << received_number << " and sent " << number << endl;
    }*/
    if (rank % 2 == 0) {

        MPI_Ssend(&number, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
        MPI_Recv(&received_number, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else {

        MPI_Recv(&received_number, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Ssend(&number, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
    }
    number = received_number + rank;
    cout << "Process " << rank << " received " << received_number << " and sent " << number << endl;

    MPI_Finalize();
    return 0;
}

//2



//double f(double x, double y) {
//    return -x + y * y;
//}
//
//double compute_integral_part(double a, double b, int N, int rank, int size) {
//    double h_x = (b - a) / N;
//    double local_sum = 0.0;
//
//   
//    for (int i = rank; i < N; i += size) {
//        double x = a + (i + 0.5) * h_x;  
//        double y_low = 0.0;
//        double y_high = 1.0 + x;
//
//        
//        int M = 1000;
//        double h_y = (y_high - y_low) / M;
//        double inner_sum = 0.0;
//
//        for (int j = 0; j < M; j++) {
//            double y = y_low + (j + 0.5) * h_y;  
//            inner_sum += f(x, y);
//        }
//
//        local_sum += inner_sum * h_y;
//    }
//
//    return local_sum * h_x;
//}
//
//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    double a = 0.0, b = 1.0;
//    int test_N[] = { 10, 100, 1000, 1000000 };
//    int num_tests = 4;
//
//    if (rank == 0) {
//        cout << "Double Integral Calculation:  (-x + y^2) dxdy" << endl;
//        cout << "Domain-> x :[0, 1], y : [0, 1+x]" << endl;
//        cout << "Number of processes: " << size << endl;
//        cout << "========================================" << endl;
//    }
//
//    if (rank == 0) {
//        
//        for (int test = 0; test < num_tests; test++) {
//            int N = test_N[test];
//
//            double start_time = MPI_Wtime();
//            double total_result = compute_integral_part(a, b, N, 0, size);
//
//            for (int i = 1; i < size; i++) {
//                double part_result;
//                MPI_Recv(&part_result, 1, MPI_DOUBLE, i, test, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                total_result += part_result;
//            }
//
//            double end_time = MPI_Wtime();
//
//            cout << "N = " << setw(5) << N << " | Result: "
//                << fixed << setprecision(5) << total_result
//                << " | Time: " << setprecision(4) << (end_time - start_time)
//                << " s" << endl;
//        }
//
//        
//        cout << "========================================" << endl;
//        cout << "Final high-precision calculation:" << endl;
//
//        int N_final = 1000000;
//        double start_time = MPI_Wtime();
//        double final_result = compute_integral_part(a, b, N_final, 0, size);
//
//     
//        for (int i = 1; i < size; i++) {
//            double part_result;
//            MPI_Recv(&part_result, 1, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            final_result += part_result;
//        }
//
//        double end_time = MPI_Wtime();
//
//        cout << "Final result: " << fixed << setprecision(5) << final_result << endl;
//        cout << "Execution time: " << setprecision(3) << (end_time - start_time) << " seconds" << endl;
//
//
//    }
//    else {
//        for (int test = 0; test < num_tests; test++) {
//            int N = test_N[test];
//            double part_result = compute_integral_part(a, b, N, rank, size);
//            MPI_Send(&part_result, 1, MPI_DOUBLE, 0, test, MPI_COMM_WORLD);
//        }
//
//       
//        int N_final = 1000000;
//        double final_part = compute_integral_part(a, b, N_final, rank, size);
//        MPI_Send(&final_part, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
//    }
//
//    MPI_Finalize();
//    return 0;
//}