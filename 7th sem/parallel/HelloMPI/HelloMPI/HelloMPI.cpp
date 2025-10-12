#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>
using namespace std;

//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    if (size < 5) {
//        if (rank == 0) {
//            cout << "The program requires at least 5 processes to work!" << endl;
//        }
//        MPI_Finalize();
//        return 1;
//    }
//
//    const int data_size = 1000;
//    vector<double> send_data(data_size);
//    vector<double> recv_data(data_size);
//
//    
//    if (rank == 0) {
//        for (int i = 0; i < data_size; i++) {
//            send_data[i] = i * 1.5;
//        }
//
//        int buffer_size;
//        MPI_Pack_size(data_size, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
//        buffer_size += MPI_BSEND_OVERHEAD;
//        vector<char> buffer(buffer_size);
//        MPI_Buffer_attach(buffer.data(), buffer_size);
//
//       
//        MPI_Send(send_data.data(), data_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
//        MPI_Ssend(send_data.data(), data_size, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
//        MPI_Bsend(send_data.data(), data_size, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
//        MPI_Rsend(send_data.data(), data_size, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD);
//
//        
//        for (int i = 1; i <= 4; i++) {
//            double confirmation;
//            MPI_Recv(&confirmation, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            cout << "proccess " << i << "confirmed receipt of the data " << endl;
//        }
//
//        MPI_Buffer_detach(buffer.data(), &buffer_size);
//    }
//    else if (rank >= 1 && rank <= 4) {
//        double start_time = MPI_Wtime();
//
//        MPI_Recv(recv_data.data(), data_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//        double end_time = MPI_Wtime();
//        double elapsed_time = end_time - start_time;
//
//
//        bool correct = true;
//        for (int i = 0; i < data_size; i++) {
//            if (abs(recv_data[i] - i * 1.5) > 1e-10) {
//                correct = false;
//                break;
//            }
//        }
//
//        cout << "proccess " << rank << " -> " << elapsed_time
//            << " second. Correct: " << (correct ? "yes" : "no") << endl;
//
//       
//        double confirmation = 1.0;
//        MPI_Send(&confirmation, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
//    }
//
//    MPI_Finalize();
//    return 0;
//}

//2


//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv); 
//
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
//    int size;
//    MPI_Comm_size(MPI_COMM_WORLD, &size); 
//
//    if (rank == 0) {
//        
//        vector<int> data1 = { 1, 2, 3, 4, 5 }; 
//        vector<int> data2 = { 6, 7, 8 }; 
//        vector<int> data3 = { 9, 10, 11, 12, 13, 14 }; 
//
//       
//        MPI_Send(data1.data(), data1.size(), MPI_INT, 1, 0, MPI_COMM_WORLD);
//        MPI_Send(data2.data(), data2.size(), MPI_INT, 2, 0, MPI_COMM_WORLD);
//        MPI_Send(data3.data(), data3.size(), MPI_INT, 3, 0, MPI_COMM_WORLD);
//    }
//    else {
//        
//        MPI_Status status;
//
//       
//        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
//
//        int count;
//        MPI_Get_count(&status, MPI_INT, &count);
//
//        vector<int> receivedData(count);
//
//        MPI_Recv(receivedData.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//       
//        int sum = 0;
//        for (int num : receivedData) {
//            sum += num;
//        }
//
//        
//        cout << "Process " << rank << " received " << count << " elements. Sum = " << sum << std::endl;
//    }
//
//    MPI_Finalize(); // Finalize the MPI environment
//    return 0;
//}

//3

//vector<double> multiply_matrix_vector(vector<vector<double>> matrix,
//    vector<double> input_vector) {
//    int rows = matrix.size();
//    int cols = input_vector.size();
//    vector<double> result(rows, 0.0);
//
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            result[i] += matrix[i][j] * input_vector[j];
//        }
//    }
//
//    return result;
//}
//
//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    srand(time(nullptr)); 
//
//    if (rank == 0) {
//        cout << "=== MATRIX VECTOR MULTIPLICATION ===" << endl;
//
//        int rows, cols;
//        cout << "Enter matrix size (rows columns): ";
//        cin >> rows >> cols;
//        cout << "Matrix size: " << rows << " x " << cols << endl;
//        cout << "Number of processes: " << size << endl;
//
//        
//        vector<vector<double>> A(rows, vector<double>(cols));
//        vector<double> v(cols);
//
//        
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < cols; j++) {
//                A[i][j] = rand() % 10; 
//            }
//        }
//
//        for (int i = 0; i < cols; i++) {
//            v[i] = rand() % 10;
//        }
//
//        
//        cout << "\nMatrix A[" << rows << "][" << cols << "]:" << endl;
//        for (int i = 0;i<rows; i++) { 
//            for (int j = 0; j <cols; j++) { 
//                cout << A[i][j] << "\t";
//            }
//            
//        }
//        
//
//        cout << "\nVector v[" << cols << "]:" << endl;
//        for (int i = 0; i <cols; i++) { 
//            cout << v[i] << "\t";
//        }
//        
//
//        
//        double start_time = MPI_Wtime();
//        auto sequential_result = multiply_matrix_vector(A, v);
//        double end_time = MPI_Wtime();
//        double sequential_time = end_time - start_time;
//
//        cout << "\n1. SEQUENTIAL CALCULATION:" << endl;
//        cout << "Time: " << sequential_time << " seconds" << endl;
//
//        start_time = MPI_Wtime();
//
//        
//        for (int i = 1; i < size; i++) {
//            MPI_Send(&cols, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//        }
//
//        
//        for (int i = 1; i < size; i++) {
//            MPI_Send(v.data(), cols, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
//        }
//
//        
//        int rows_per_process = rows / (size - 1);
//        int extra_rows = rows % (size - 1);
//
//        cout << "\nROW DISTRIBUTION:" << endl;
//        cout << "Rows per process: " << rows_per_process << endl;
//        cout << "Extra rows: " << extra_rows << endl;
//
//        int current_row = 0;
//        for (int worker = 1; worker < size; worker++) {
//            int rows_to_send = rows_per_process;
//            if (worker <= extra_rows) {
//                rows_to_send++;
//            }
//
//            MPI_Send(&rows_to_send, 1, MPI_INT, worker, 2, MPI_COMM_WORLD);
//
//            for (int i = 0; i < rows_to_send; i++) {
//                MPI_Send(A[current_row].data(), cols, MPI_DOUBLE, worker, 3, MPI_COMM_WORLD);
//                current_row++;
//            }
//            cout << "Sent " << rows_to_send << " rows to process " << worker << endl;
//        }
//
//        
//        vector<double> parallel_result(rows);
//        current_row = 0;
//
//        for (int worker = 1; worker < size; worker++) {
//            int rows_received;
//            MPI_Recv(&rows_received, 1, MPI_INT, worker, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//            for (int i = 0; i < rows_received; i++) {
//                double value;
//                MPI_Recv(&value, 1, MPI_DOUBLE, worker, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                parallel_result[current_row++] = value;
//            }
//        }
//
//        end_time = MPI_Wtime();
//        double parallel_time = end_time - start_time;
//
//        cout << "\n2. PARALLEL CALCULATION:" << endl;
//        cout << "Time: " << parallel_time << " seconds" << endl;
//
//        
//        cout << "\n3. COMPARISON RESULTS:" << endl;
//        if (parallel_time > 0) {
//            cout << "Speedup: " << sequential_time / parallel_time << " times" << endl;
//        }
//        else {
//            cout << "Speedup: Very large (parallel time too small)" << endl;
//        }
//
//        
//        double max_error = 0.0;
//        for (int i = 0; i < rows; i++) {
//            double error = abs(sequential_result[i] - parallel_result[i]);
//            if (error > max_error) {
//                max_error = error;
//            }
//        }
//
//        cout << "Maximum error: " << max_error << endl;
//
//       
//        cout << "\nSequential result (A * v):" << endl;
//        for (int i = 0; i < min(rows, 10); i++) {
//            cout << sequential_result[i] << "\t";
//        }
//        if (rows > 10) cout << "...";
//        cout << endl;
//
//        cout << "\nParallel result (A * v):" << endl;
//        for (int i = 0; i < min(rows, 10); i++) {
//            cout << parallel_result[i] << "\t";
//        }
//        if (rows > 10) cout << "...";
//        cout << endl;
//
//        
//        if (max_error < 1e-10) {
//            cout << "\n? RESULTS ARE CORRECT!" << endl;
//        }
//        else {
//            cout << "\n? RESULTS ARE INCORRECT!" << endl;
//        }
//
//    }
//    else {
//        
//        int cols;
//        MPI_Recv(&cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//      
//        vector<double> vec(cols);
//        MPI_Recv(vec.data(), cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//        
//        int rows_to_process;
//        MPI_Recv(&rows_to_process, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//        vector<double> local_results(rows_to_process);
//        vector<double> row(cols);
//
//       
//        for (int i = 0; i < rows_to_process; i++) {
//            MPI_Recv(row.data(), cols, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//            double sum = 0.0;
//            for (int j = 0; j < cols; j++) {
//                sum += row[j] * vec[j];
//            }
//            local_results[i] = sum;
//        }
//
//       
//        MPI_Send(&rows_to_process, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
//        for (double result : local_results) {
//            MPI_Send(&result, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
//        }
//
//        cout << "Process " << rank << " processed " << rows_to_process << " rows" << endl;
//    }
//
//    MPI_Finalize();
//    return 0;
//}

//4

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int TAG = 0;
    int message;

    if (size != 2) {
        cout << "This program requires exactly two processes." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {

        cout << "Process 0: Waiting to receive message from Process 1..." << endl;
        MPI_Recv(&message, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Process 0: Received message from Process 1." << endl;

    }
    else if (rank == 1) {
        int message = 1;
        MPI_Send(&message, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD);

    }


    MPI_Finalize();
    return 0;
}