// compute.cpp
//
// Usage example:
//   ./compute \
//     --a A.bin        --a_meta A.meta \
//     --b B.bin        --b_meta B.meta
//
// Where A.meta contains: "M K" on the first line
//       B.meta contains: "K N" on the first line
// and A.bin is M*K floats (row-major), B.bin is K*N floats (row-major).
//
// This program:
//   - mmaps A.bin and B.bin as read-only
//   - creates C.bin as M*N floats, mmaps it MAP_SHARED
//   - forks kNumWorkers processes, each computing a subset of rows of C

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstddef>
#include <algorithm>
#include <cerrno>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

struct Dim2 {
    std::size_t rows;
    std::size_t cols;
};

bool parse_dims_from_meta(const std::string& path, Dim2& dims) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: could not open meta file: " << path << "\n";
        return false;
    }
    std::size_t r, c;
    if (!(in >> r >> c)) {
        std::cerr << "Error: failed to read dims from meta file: " << path << "\n";
        return false;
    }
    dims.rows = r;
    dims.cols = c;
    return true;
}

bool file_size_matches(const std::string& path, std::size_t expected_bytes) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        std::cerr << "Error: stat failed for " << path << ": " << std::strerror(errno) << "\n";
        return false;
    }
    if (static_cast<std::size_t>(st.st_size) != expected_bytes) {
        std::cerr << "Error: file size mismatch for " << path
                  << " (expected " << expected_bytes
                  << " bytes, got " << static_cast<std::size_t>(st.st_size) << ")\n";
        return false;
    }
    return true;
}

void worker_main(int rank,
                 int num_workers,
                 const float* A,
                 const float* B,
                 float* C,
                 std::size_t M,
                 std::size_t N,
                 std::size_t K)
{
    // Simple 1D row partition among workers
    std::size_t rows_per = (M + num_workers - 1) / num_workers;
    std::size_t row_start = static_cast<std::size_t>(rank) * rows_per;
    std::size_t row_end   = std::min(M, row_start + rows_per);

    for (std::size_t i = row_start; i < row_end; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            const std::size_t a_row_offset = i * K;
            const std::size_t c_idx = i * N + j;
            for (std::size_t k = 0; k < K; ++k) {
                const float a_ik = A[a_row_offset + k];
                const float b_kj = B[k * N + j];
                acc += a_ik * b_kj;
            }
            C[c_idx] = acc;
        }
    }
}

int main(int argc, char** argv) {
    // Command-line arguments
    std::string a_path;
    std::string b_path;
    std::string a_meta_path;
    std::string b_meta_path;

    // Very simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_arg = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing value after " << arg << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--a") {
            a_path = next_arg();
        } else if (arg == "--b") {
            b_path = next_arg();
        } else if (arg == "--a_meta") {
            a_meta_path = next_arg();
        } else if (arg == "--b_meta") {
            b_meta_path = next_arg();
        } else {
            std::cerr << "Warning: unrecognized argument ignored: " << arg << "\n";
        }
    }

    if (a_path.empty() || b_path.empty() ||
        a_meta_path.empty() || b_meta_path.empty()) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " --a A.bin --a_meta A.meta"
                  << " --b B.bin --b_meta B.meta\n";
        return 1;
    }

    Dim2 dimsA, dimsB;
    if (!parse_dims_from_meta(a_meta_path, dimsA)) return 1;
    if (!parse_dims_from_meta(b_meta_path, dimsB)) return 1;

    std::size_t M = dimsA.rows;
    std::size_t K = dimsA.cols;
    std::size_t K2 = dimsB.rows;
    std::size_t N = dimsB.cols;

    if (K != K2) {
        std::cerr << "Error: inner dimensions mismatch: A is "
                  << M << "x" << K << ", B is "
                  << K2 << "x" << N << "\n";
        return 1;
    }

    std::cout << "A: " << M << " x " << K << "\n";
    std::cout << "B: " << K << " x " << N << "\n";

    const std::size_t bytesA = M * K * sizeof(float);
    const std::size_t bytesB = K * N * sizeof(float);
    const std::size_t bytesC = M * N * sizeof(float);

    if (!file_size_matches(a_path, bytesA)) return 1;
    if (!file_size_matches(b_path, bytesB)) return 1;

    // Open and mmap A
    int fdA = open(a_path.c_str(), O_RDONLY);
    if (fdA < 0) {
        std::cerr << "Error: open failed for " << a_path << ": "
                  << std::strerror(errno) << "\n";
        return 1;
    }

    void* a_map = mmap(nullptr, bytesA, PROT_READ, MAP_SHARED, fdA, 0);
    if (a_map == MAP_FAILED) {
        std::cerr << "Error: mmap failed for A: " << std::strerror(errno) << "\n";
        close(fdA);
        return 1;
    }
    const float* A = static_cast<const float*>(a_map);

    // Open and mmap B
    int fdB = open(b_path.c_str(), O_RDONLY);
    if (fdB < 0) {
        std::cerr << "Error: open failed for " << b_path << ": "
                  << std::strerror(errno) << "\n";
        munmap(a_map, bytesA);
        close(fdA);
        return 1;
    }

    void* b_map = mmap(nullptr, bytesB, PROT_READ, MAP_SHARED, fdB, 0);
    if (b_map == MAP_FAILED) {
        std::cerr << "Error: mmap failed for B: " << std::strerror(errno) << "\n";
        munmap(a_map, bytesA);
        close(fdA);
        close(fdB);
        return 1;
    }
    const float* B = static_cast<const float*>(b_map);

    // Create and mmap C (shared output)
    const std::string c_path = "C.bin";
    int fdC = open(c_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (fdC < 0) {
        std::cerr << "Error: open failed for " << c_path << ": "
                  << std::strerror(errno) << "\n";
        munmap(a_map, bytesA);
        munmap(b_map, bytesB);
        close(fdA);
        close(fdB);
        return 1;
    }

    if (ftruncate(fdC, static_cast<off_t>(bytesC)) != 0) {
        std::cerr << "Error: ftruncate failed for C: " << std::strerror(errno) << "\n";
        munmap(a_map, bytesA);
        munmap(b_map, bytesB);
        close(fdA);
        close(fdB);
        close(fdC);
        return 1;
    }

    void* c_map = mmap(nullptr, bytesC,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED,
                       fdC, 0);
    if (c_map == MAP_FAILED) {
        std::cerr << "Error: mmap failed for C: " << std::strerror(errno) << "\n";
        munmap(a_map, bytesA);
        munmap(b_map, bytesB);
        close(fdA);
        close(fdB);
        close(fdC);
        return 1;
    }
    float* C = static_cast<float*>(c_map);

    // std::memset(C, 0, bytesC);

    // Number of worker processes
    const int kNumWorkers = 4;  // adjust as desired

    std::cout << "Spawning " << kNumWorkers << " worker processes...\n";

    for (int rank = 0; rank < kNumWorkers; ++rank) {
        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "Error: fork failed: " << std::strerror(errno) << "\n";
            // Attempt best-effort wait for already-forked children
            for (int j = 0; j < rank; ++j) {
                int status = 0;
                waitpid(-1, &status, 0);
            }
            munmap(a_map, bytesA);
            munmap(b_map, bytesB);
            munmap(c_map, bytesC);
            close(fdA);
            close(fdB);
            close(fdC);
            return 1;
        }

        if (pid == 0) {
            // Child process
            worker_main(rank, kNumWorkers, A, B, C, M, N, K);
            _exit(0);
        }
        // Parent continues looping
    }

    // Parent: wait for all children
    for (int i = 0; i < kNumWorkers; ++i) {
        int status = 0;
        pid_t w = waitpid(-1, &status, 0);
        if (w < 0) {
            std::cerr << "Warning: waitpid failed: " << std::strerror(errno) << "\n";
        }
    }

    std::cout << "All workers finished. Output written to " << c_path << "\n";

    // Cleanup (parent)
    munmap(a_map, bytesA);
    munmap(b_map, bytesB);
    munmap(c_map, bytesC);
    close(fdA);
    close(fdB);
    close(fdC);

    return 0;
}
