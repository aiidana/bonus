#include<mpi.h>
#include<iostream>
#include<vector>
#include <random>
#include <iomanip>
using namespace std;
//1

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "Process " << rank << " of " << size << endl;
    printf("hello mpi =%d, total=%d\n", rank, size);

    MPI_Finalize();
    return 0;
}

//2

//int main(int argc, char* argv[]) {
//	MPI_Init(&argc, &argv);
//	int n;
//	int rank, size;
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//
//	if (rank == 0) {
//		cin >> n;
//	}
//	// Разослать n всем процессам
//	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//	vector<int> m(n);
//	int sum1 = 0, sum2 = 0,total_sum=0;
//	if (rank == 0) {
//		for (int i = 0; i < n; i++) {
//			m[i] = rand() % 100;
//		}
//		cout << "Array: ";
//		for (int i = 0; i < n; i++) {
//			cout << m[i] << " ";
//		}
//		cout << endl;
//	}
//
//	int elements = n / size;
//	vector<int> my_part(elements);
//
//	MPI_Scatter(
//		m.data(), elements, MPI_INT,
//		my_part.data(), elements, MPI_INT,
//		0, MPI_COMM_WORLD
//	);
//
//	cout << "Process " << rank << " elements: ";
//	for (int i = 0; i < elements; i++) {
//		cout << my_part[i] << " ";
//		sum1 += my_part[i];
//	}
//	
//	MPI_Allreduce(&sum1, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//
//
//	cout << "| local sum = " << sum1
//		<< " | global sum = " << total_sum << endl;
//
//	//Сравнение с последовательным решением
//	if (rank == 0) {
//		int seq_sum = 0;
//		for (int x : m) seq_sum += x;
//		cout << "\nSequential sum = " << seq_sum << endl;
//		cout << "\nTotal sum = " << total_sum << endl;
//	}
//	
//	MPI_Finalize();
//	return 0;
//}


//3

//double calculate_pi(int terms, int rank, int size) {
//    double pi = 0.0;
//
//    for (int i = rank; i < terms; i += size) {
//        if (i % 2 == 0) {
//            pi += 1.0 / (2.0 * i + 1.0);
//        }
//        else {
//            pi -= 1.0 / (2.0 * i + 1.0);
//        }
//    }
//
//    double global_pi;
//    MPI_Reduce(&pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//
//    return global_pi * 4;
//}
//
//int main(int argc, char* argv[]) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    vector<int> test_terms = { 100000, 500000, 1000000, 2000000 };
//
//    if (rank == 0) {
//        cout << "pi =" << endl;
//        cout << "number of processors " << size << endl << endl;
//    }
//    for (int terms : test_terms) {
//        MPI_Barrier(MPI_COMM_WORLD); 
//
//        double start_time = MPI_Wtime();
//        double pi_approx = calculate_pi(terms, rank, size);
//        double end_time = MPI_Wtime();
//
//        if (rank == 0) {
//            double error = abs(pi_approx - 3.141592653589793);
//            double time = end_time - start_time;
//
//            cout << " \npi = " << fixed << setprecision(10) << pi_approx;
//            cout << " error= "  << error;
//            cout << " time= " << fixed << setprecision(4) << time << "с";
//            cout << endl;
//        }
//    }
//    
//
//    MPI_Finalize();
//    return 0;
//
//}


//4
// compute_e

//
//int main(int argc, char* argv[]) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    int n; 
//    if (rank == 0) {
//        cout << "Enter n (number of terms, e ~= sum_{k=0..n} 1/k!): ";
//        cin >> n;
//    }
//
//    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//    double local_sum = 0.0;
//    for (int k = rank; k <= n; k += size) {
//       
//        double fact = 1.0;
//        for (int j = 1; j <= k; ++j) fact *= j;
//        local_sum += 1.0 / fact;
//    }
//
//   
//    if (rank == 0) {
//        double total = local_sum;
//        for (int p = 1; p < size; ++p) {
//            double tmp;
//            MPI_Recv(&tmp, 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            total += tmp;
//        }
//        double seq = 0.0;
//        for (int k = 0; k <= n; ++k) {
//            double fact = 1.0;
//            for (int j = 1; j <= k; ++j) fact *= j;
//            seq += 1.0 / fact;
//        }
//
//        cout.precision(12);
//        cout << "Parallel sum (approx e) = " << total << endl;
//        cout << "Sequential sum (check)   = " << seq << endl;
//    }
//    else {
//        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
//    }
//
//    MPI_Finalize();
//    return 0;
//}


//2

//int main(int argc, char* argv[]) {
//    MPI_Init(&argc, &argv);
//    int n;
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    vector<int> m;
//
//    if (rank == 0) {
//        cout << "Enter array size: ";
//        cin >> n;
//
//        m.resize(n);
//        for (int i = 0; i < n; i++) {
//            m[i] = rand() % 100;
//        }
//
//        cout << "Array: ";
//        for (int i = 0; i < n; i++) {
//            cout << m[i] << " ";
//        }
//        cout << endl;
//
//        for (int i = 1; i < size; i++) {
//            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//        }
//    }
//    else {
//        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    }
//
//    int elements_per_process = n / size;
//    vector<int> my_part(elements_per_process);
//
//    if (rank == 0) {
//       
//        for (int i = 1; i < size; i++) {
//            int start_index = i * elements_per_process;
//            MPI_Send(&m[start_index], elements_per_process, MPI_INT, i, 1, MPI_COMM_WORLD);
//        }
//
//        for (int i = 0; i < elements_per_process; i++) {
//            my_part[i] = m[i];
//        }
//    }
//    else {
//       
//        MPI_Recv(my_part.data(), elements_per_process, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    }
//
//    int local_sum = 0;
//    cout << "Process " << rank << " elements: ";
//    for (int i = 0; i < elements_per_process; i++) {
//        cout << my_part[i] << " ";
//        local_sum += my_part[i];
//    }
//    cout << "| local sum = " << local_sum << endl;
//
//   
//    int total_sum = 0;
//
//    if (rank == 0) {
//        total_sum = local_sum;
//
//       
//        for (int i = 1; i < size; i++) {
//            int received_sum;
//            MPI_Recv(&received_sum, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            total_sum += received_sum;
//        }
//
//       
//        cout << "\nTotal sum = " << total_sum << endl;
//
//        
//        int seq_sum = 0;
//        for (int x : m) seq_sum += x;
//        cout << "Sequential sum = " << seq_sum << endl;
//    }
//    else {
//        
//        MPI_Send(&local_sum, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
//    }
//
//    MPI_Finalize();
//    return 0;
//}


//3
//double calculate_pi(int terms, int rank, int size) {
//    double local_pi = 0.0;
//
//    for (int i = rank; i < terms; i += size) {
//        if (i % 2 == 0) {
//            local_pi += 1.0 / (2.0 * i + 1.0);
//        }
//        else {
//            local_pi -= 1.0 / (2.0 * i + 1.0);
//        }
//    }
//
//    double global_pi = 0.0;
//
//    if (rank == 0) {
//        global_pi = local_pi;
//
//        for (int i = 1; i < size; i++) {
//            double received_pi;
//            MPI_Recv(&received_pi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            global_pi += received_pi;
//        }
//
//        global_pi *= 4;  
//    }
//    else {
//       
//        MPI_Send(&local_pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
//    }
//
//    return global_pi;
//}
//
//int main(int argc, char* argv[]) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    vector<int> test_terms = { 100000, 500000, 1000000, 2000000 };
//
//    if (rank == 0) {
//        cout << "pi calculation" << endl;
//        cout << "number of processors: " << size << endl << endl;
//    }
//
//    for (int terms : test_terms) {
//     
//        MPI_Barrier(MPI_COMM_WORLD);
//
//        double start_time = MPI_Wtime();
//        double pi_approx = calculate_pi(terms, rank, size);
//        double end_time = MPI_Wtime();
//
//        if (rank == 0) {
//            double error = abs(pi_approx - 3.141592653589793);
//            double time = end_time - start_time;
//
//            cout << "Terms: " << terms << endl;
//            cout << "pi = " << fixed << setprecision(10) << pi_approx << endl;
//            cout << "error = " << scientific << error << endl;
//            cout << "time = " << fixed << setprecision(4) << time << "s" << endl;
//            cout << "------------------------" << endl;
//        }
//    }
//
//    MPI_Finalize();
//    return 0;
//}