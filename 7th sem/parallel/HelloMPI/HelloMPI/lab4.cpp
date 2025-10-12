#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);  // ����: MPL_Init
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // ����: MPL_Conn_rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // ����: MPL_Conn_size
    cout << "Hello from process " << rank << " of " << size << endl;
    MPI_Finalize();  // ����: MPL_Finalize
    return 0;
}