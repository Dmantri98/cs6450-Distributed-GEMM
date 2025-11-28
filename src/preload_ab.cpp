#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unistd.h>   // fork
#include <sys/wait.h> // waitpid
#include <iostream>

// Your HWCCEntry and helpers will be adapted:
#include "../include/hwcc.hpp"


void worker_main(int rank, int num_workers,
                 const float* A, const float* B, float* C,
                 int M, int N, int K) {
    // Simple 1D row partitioning example:
    int rows_per = (M + num_workers - 1) / num_workers;
    int row_start = rank * rows_per;
    int row_end   = std::min(M, row_start + rows_per);

    for (int i = row_start; i < row_end; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}



int main(int argc, char** argv) {
// 1. Open A and B as read-only binary dense matrices
int fdA = open("A.bin", O_RDONLY);
int fdB = open("B.bin", O_RDONLY);

size_t bytesA = M * K * sizeof(float);
size_t bytesB = K * N * sizeof(float);

float* A = static_cast<float*>(mmap(nullptr, bytesA, PROT_READ, MAP_PRIVATE, fdA, 0));
float* B = static_cast<float*>(mmap(nullptr, bytesB, PROT_READ, MAP_PRIVATE, fdB, 0));

// 2. Create C as a shared output file
int fdC = open("C.bin", O_CREAT | O_RDWR, 0666);
size_t bytesC = M * N * sizeof(float);
ftruncate(fdC, bytesC);

float* C = static_cast<float*>(mmap(nullptr, bytesC,
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED,
                                    fdC, 0));

// 3. Fork worker processes
int num_workers = 4;  // example
for (int rank = 0; rank < num_workers; ++rank) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);
    }
    if (pid == 0) {
        // child
        worker_main(rank, num_workers, A, B, C, M, N, K);
        _exit(0);
    }
}

// 4. Parent waits for all children
for (int i = 0; i < num_workers; ++i) {
    int status = 0;
    waitpid(-1, &status, 0);
}

}
