#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);  // Áûכמ: MPL_Init
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Áûכמ: MPL_Conn_rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Áûכמ: MPL_Conn_size
    cout << "Hello from process " << rank << " of " << size << endl;
    MPI_Finalize();  // Áûכמ: MPL_Finalize
    return 0;
}